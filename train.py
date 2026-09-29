"""ChessGPT baseline and two-axis recurrent training, adapted from Karvonen/nanoGPT.

    uv run python train.py experiments/smoke/configs/baseline.py
    uv run python -m experiments.run_serious pair --billions 1
    uv run torchrun --standalone --nproc_per_node=2 train.py --gradient_accumulation_steps=2

Gradient accumulation is a global count split across DDP ranks, as in upstream.
"""
import json
import math
import os
from pathlib import Path
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import ChessData, file_hash
from inference.live import create_live_state, decode_live_step, validate_live_inference_spec
from model import GPT, GPTConfig
from moves.data import OBJECTIVES, MoveData, training_batches
from moves.vocab import PAD
from models.recurrent_2d import (Recurrent2DGPT, RecurrentGPTConfig,
                                 validate_recurrence_counts, validate_update_probability_distribution,
                                 validate_recurrence_mode)
from recurrence.schedule import (RecurrenceScheduleSampler, normalize_update_config,
                                 sample_schedule, update_probabilities_at_step)
from training_utils import append_json, atomic_save, capture_rng, provenance, restore_rng

DEFAULTS = dict(
    out_dir='experiments/manual/results', dataset='chess_v1', init_from='scratch',
    eval_interval=4000, eval_iters=100, log_interval=50, eval_only=False,
    n_layer=8, n_head=8, n_embd=512, block_size=1023, bias=False, dropout=0.0,
    batch_size=100, gradient_accumulation_steps=1, learning_rate=3e-4,
    max_iters=600000, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
    decay_lr=True, lr_schedule=None, warmup_iters=2000, lr_decay_start=None,
    lr_decay_iters=600000, min_lr=3e-5,
    backend='nccl', device='cuda', dtype='bfloat16', compile=True, seed=1337,
    num_threads=4, architecture='baseline', n_prelude=1, n_core=4, n_coda=1, n_buffer=1, n_source=1,
    update_support=[], update_probabilities=[], recurrence_seed=1729, recurrence_mode='hybrid',
    update_probability_schedule=None,
    temporal_memory_gate_init=0.1,
    eval_u_t=0, eval_u_d=0, keep_checkpoints=False, checkpoint_steps=None,
    checkpoint_interval=0,
    eval_panel_path='', deep_supervision=False, deep_supervision_lambda=0.25,
    training_budget_seconds=0.0,
    # Warm start (temporal mode): this fraction of each update's microbatches first settles the
    # temporal memory with gradient-free passes, then trains its sampled schedule from it.
    warm_start_fraction=0.0, warm_start_max_passes=64, warm_start_tolerance=0.01,
    # Move tokens (docs/engine_policy_plan.md): data_format='moves' reads a moves.prepare dataset
    # and trains one objective: 'human' (next ply), 'legal' or 'engine'. Rows come from random
    # games with probability random_game_fraction. init_from='continue' starts from the weights
    # of continue_from with a fresh output layer, optimizer and step count.
    data_format='characters', objective='next_token', random_game_fraction=0.0,
    value_budget='deep', continue_from='',
    # Live (cached, token-by-token) evaluation of recurrent move runs on dev games, at these core
    # iteration counts; None means [1] for temporal and [1, 2, 4] for depth and hybrid.
    live_eval_iters=1, live_eval_depths=None,
    # Processes building move training batches ahead of the training loop (0: build in-process).
    # Batches depend only on seed, rank and index, so this never changes what is trained on.
    loader_workers=0,
)


def get_lr(step, config):
    """Resolve learning rate for an optimizer-step index.

    ``lr_schedule=None`` preserves historical ``decay_lr`` behavior. For WSD,
    ``lr_decay_iters`` is the number of optimizer updates: the final update at
    index ``lr_decay_iters - 1`` uses ``min_lr`` exactly.
    """
    schedule = config.get('lr_schedule')
    if schedule is None:
        schedule = 'cosine' if config['decay_lr'] else 'constant'
    if schedule == 'constant':
        return config['learning_rate']
    if schedule not in {'cosine', 'wsd'}:
        raise ValueError("lr_schedule must be one of None, 'cosine', 'constant', or 'wsd'")
    if step < config['warmup_iters']:
        return config['learning_rate'] * step / config['warmup_iters']
    if schedule == 'wsd':
        decay_start = config['lr_decay_start']
        if step < decay_start:
            return config['learning_rate']
        decay_end = config['lr_decay_iters'] - 1
        if step >= decay_end:
            return config['min_lr']
        ratio = ((step - decay_start) /
                 (decay_end - decay_start))
        return config['learning_rate'] + ratio * (config['min_lr'] - config['learning_rate'])
    if step >= config['lr_decay_iters']:
        return config['min_lr']
    ratio = (step - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    return config['min_lr'] + 0.5 * (1 + math.cos(math.pi * ratio)) * (config['learning_rate'] - config['min_lr'])


def aggregate_gradient_stats(model, scaler, optimizer, grad_clip, step):
    """Unscale, validate, and clip one accumulated optimizer update."""
    if grad_clip or scaler.is_enabled():
        scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError(f'Non-finite gradient before optimizer step at step {step}')
    if grad_clip:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm_value = float(grad_norm.detach().cpu())
        clip_coefficient = min(1.0, grad_clip / (grad_norm_value + 1e-6))
        clipped = grad_norm_value > grad_clip
    else:
        squared_norm = sum(gradient.detach().float().square().sum().item() for gradient in gradients)
        grad_norm_value = math.sqrt(squared_norm)
        clip_coefficient, clipped = 1.0, False
    if not math.isfinite(grad_norm_value):
        raise FloatingPointError(f'Non-finite aggregate gradient norm before optimizer step at step {step}')
    return dict(grad_norm_pre_clip=grad_norm_value,
                applied_clip_coefficient=clip_coefficient, gradients_clipped=clipped)


def train(config):
    config = {**DEFAULTS, **normalize_update_config(config)}
    if config['architecture'] not in ['baseline', 'recurrent']:
        raise ValueError('architecture must be baseline or recurrent')
    budget = config['training_budget_seconds']
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError('training_budget_seconds must be finite and nonnegative')
    recurrent = config['architecture'] == 'recurrent'
    if config['deep_supervision'] and not recurrent:
        raise ValueError('Deep supervision is currently supported for recurrent training only')
    if (isinstance(config['deep_supervision_lambda'], bool) or
            not isinstance(config['deep_supervision_lambda'], (int, float)) or
            not math.isfinite(config['deep_supervision_lambda']) or
            config['deep_supervision_lambda'] < 0):
        raise ValueError('deep_supervision_lambda must be a finite nonnegative number')
    if recurrent and config['compile']:
        raise ValueError('Use compile=False for variable recurrent schedules in the MVP')
    if recurrent:
        validate_recurrence_mode(config['recurrence_mode'])
        if not config['update_support']:
            raise ValueError('update_support must be explicitly provided for recurrent training')
        initial_probabilities = update_probabilities_at_step(config, 0)
        sampler = RecurrenceScheduleSampler(config['update_support'],
                                             None if config['update_probability_schedule'] is not None
                                             else config['update_probabilities'],
                                             config['recurrence_seed'])
        validate_update_probability_distribution(config['recurrence_mode'], config['update_support'],
                                                 initial_probabilities)
        validate_recurrence_counts(config['recurrence_mode'], config['eval_u_t'], config['eval_u_d'])
        sample_schedule(config['eval_u_t'], config['eval_u_d'], random.Random(0))
    else:
        sampler = None
    fraction = config['warm_start_fraction']
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError('warm_start_fraction must be between zero and one')
    if fraction and not (recurrent and config['recurrence_mode'] == 'temporal'):
        raise ValueError('Warm-start batches are implemented for the temporal mode only')
    if type(config['warm_start_max_passes']) is not int or config['warm_start_max_passes'] < 1:
        raise ValueError('warm_start_max_passes must be a positive integer')
    if not config['warm_start_tolerance'] >= 0:
        raise ValueError('warm_start_tolerance must be nonnegative')
    if config['init_from'] not in ['scratch', 'resume', 'continue']:
        raise ValueError('init_from must be scratch, resume or continue')
    moves = config['data_format'] == 'moves'
    if config['data_format'] not in ['characters', 'moves']:
        raise ValueError("data_format must be 'characters' or 'moves'")
    if not moves and (config['objective'] != 'next_token' or config['random_game_fraction'] or
                      config['init_from'] == 'continue' or config['live_eval_depths'] is not None or
                      config['loader_workers']):
        raise ValueError('Objectives, random games, continued runs, live evaluation and loader workers '
                         'need data_format=moves')
    live_depths = []
    if moves and recurrent and config['live_eval_iters']:
        mode = config['recurrence_mode']
        live_depths = (list(config['live_eval_depths']) if config['live_eval_depths'] is not None
                       else [1] if mode == 'temporal' else [1, 2, 4])
        if any(type(depth) is not int or depth < 1 for depth in live_depths) or (mode == 'temporal' and live_depths != [1]):
            raise ValueError('live_eval_depths must be positive integers, and [1] for temporal models')
    elif moves and config['live_eval_depths']:
        raise ValueError('Live evaluation is for recurrent models; a transformer runs live exactly as trained')
    live_strategy = 'final_depth' if config['recurrence_mode'] == 'temporal' else 'depth_specialized'
    if moves:
        if config['objective'] not in OBJECTIVES:
            raise ValueError(f'objective must be one of {OBJECTIVES} for move data')
        random_fraction = config['random_game_fraction']
        if (isinstance(random_fraction, bool) or not isinstance(random_fraction, (int, float)) or
                not 0 <= random_fraction <= 1):
            raise ValueError('random_game_fraction must be between zero and one')
        if config['objective'] == 'engine' and random_fraction:
            raise ValueError('The engine objective trains on engine-labelled human games only')
        if config['eval_panel_path']:
            raise ValueError('Evaluation panels are defined for character data only')
        if type(config['live_eval_iters']) is not int or config['live_eval_iters'] < 0:
            raise ValueError('live_eval_iters must be a nonnegative integer')
        if type(config['loader_workers']) is not int or config['loader_workers'] < 0:
            raise ValueError('loader_workers must be a nonnegative integer')
        # A resumed continuation keeps continue_from as provenance; resume checks it matches.
        if (config['init_from'] == 'continue' and not config['continue_from'] or
                config['init_from'] == 'scratch' and config['continue_from']):
            raise ValueError("init_from='continue' needs continue_from; a scratch run can't have one")
    for key in ['batch_size', 'gradient_accumulation_steps', 'eval_interval', 'eval_iters', 'log_interval', 'num_threads']:
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if config['max_iters'] < 0 or config['warmup_iters'] < 0:
        raise ValueError('Iteration counts must be nonnegative')
    if type(config['checkpoint_interval']) is not int or config['checkpoint_interval'] < 0:
        raise ValueError('checkpoint_interval must be a nonnegative integer (zero disables it)')
    schedule = config['lr_schedule']
    if schedule not in (None, 'cosine', 'constant', 'wsd'):
        raise ValueError("lr_schedule must be one of None, 'cosine', 'constant', or 'wsd'")
    effective_schedule = schedule or ('cosine' if config['decay_lr'] else 'constant')
    if effective_schedule in ('cosine', 'wsd') and config['lr_decay_iters'] <= config['warmup_iters']:
        raise ValueError('lr_decay_iters must exceed warmup_iters')
    if effective_schedule == 'wsd':
        if any(type(config[key]) is not int or config[key] < 0
               for key in ('warmup_iters', 'lr_decay_iters')):
            raise ValueError('WSD warmup_iters and lr_decay_iters must be nonnegative integers')
        decay_start = config['lr_decay_start']
        if (type(decay_start) is not int or decay_start < config['warmup_iters'] or
                decay_start >= config['lr_decay_iters'] - 1):
            raise ValueError('WSD requires warmup_iters <= lr_decay_start < lr_decay_iters - 1')
        rates = (config['learning_rate'], config['min_lr'])
        if (any(isinstance(rate, bool) or not isinstance(rate, (int, float)) or
                not math.isfinite(rate) or rate < 0 for rate in rates) or
                config['min_lr'] > config['learning_rate']):
            raise ValueError('WSD requires finite nonnegative rates with min_lr <= learning_rate')
    if config['checkpoint_steps'] is not None:
        if (not config['checkpoint_steps'] or
                any(type(step) is not int or step < 0 for step in config['checkpoint_steps']) or
                len(set(config['checkpoint_steps'])) != len(config['checkpoint_steps'])):
            raise ValueError('checkpoint_steps must be a nonempty list of distinct nonnegative integers')
    torch.set_num_threads(config['num_threads'])
    ddp = int(os.environ.get('RANK', -1)) >= 0
    rank, world_size = 0, 1
    device = config['device']
    if ddp:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if device.startswith('cuda'):
            device = f"cuda:{os.environ['LOCAL_RANK']}"
            torch.cuda.set_device(device)
        elif device != 'cpu':
            raise ValueError('DDP is supported on CPU or CUDA')
        dist.init_process_group(backend=config['backend'])
    if budget and ddp:
        raise ValueError('Time-budget training currently requires a single process')
    master = rank == 0
    if config['gradient_accumulation_steps'] % world_size:
        raise ValueError('Global gradient_accumulation_steps must be divisible by world size')
    accumulation = config['gradient_accumulation_steps'] // world_size
    device_type = device.split(':')[0]
    if device_type != 'cuda' and config['dtype'] != 'float32':
        raise ValueError('CPU/MPS runs use float32; select --dtype=float32')
    if config['dtype'] not in ['float32', 'float16', 'bfloat16']:
        raise ValueError('Unsupported dtype')
    if config['compile'] and device_type == 'mps':
        raise ValueError('Use --compile=False on MPS')
    seed = config['seed'] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_rng = torch.Generator().manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if moves:
        data = MoveData(Path('data') / config['dataset'], config['block_size'],
                        value_budget=config['value_budget'] if config['objective'] == 'engine' else None)
    else:
        data = ChessData(Path('data') / config['dataset'], config['block_size'])
    panel = None
    panel_indices = None
    if config['eval_panel_path']:
        from evaluation.panels import load_panel
        panel = load_panel(config['eval_panel_path'], data, split='selection')
        panel_indices = panel['row_indices']
    out = Path(config['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if config['init_from'] != 'resume' and (out / 'ckpt.pt').exists():
        raise FileExistsError('Checkpoint exists: choose a new out_dir or init_from=resume')
    model_args = {key: config[key] for key in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout']}
    model_args['vocab_size'] = data.meta['vocab_size']
    if moves:
        # Moves in, a score per move out: the readout is untied and one move shorter than the vocabulary.
        model_args.update(output_size=data.meta['output_size'], tie_weights=False)
    if recurrent:
        model_args.update({key: config[key] for key in ['n_prelude', 'n_buffer', 'n_core', 'n_source', 'n_coda',
                                                        'recurrence_mode', 'temporal_memory_gate_init']})
    checkpoint = None
    step, best_val, last_eval_step = 0, float('inf'), -1
    training_seconds = 0.0
    move_counts = dict(human_plies=0, random_plies=0, supervised_positions=0, row_tokens=0)
    continued = None
    if config['init_from'] == 'resume':
        checkpoint = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=False)
        saved_args = checkpoint['model_args']
        saved_config = normalize_update_config(checkpoint['config'])
        same_model = (RecurrentGPTConfig.from_checkpoint(saved_args) == RecurrentGPTConfig(**model_args)
                      if recurrent else saved_args == model_args)
        if not same_model:
            raise ValueError('Resume model configuration differs from checkpoint')
        if checkpoint['manifest_hash'] != data.manifest_hash:
            raise ValueError('Resume dataset differs from checkpoint')
        if checkpoint['world_size'] != world_size:
            raise ValueError('Exact resume requires the same world size')
        if checkpoint.get('eval_panel_sha256') != (panel['sha256'] if panel else None):
            raise ValueError('Resume evaluation panel differs from checkpoint')
        mutable = {'out_dir', 'max_iters', 'init_from', 'eval_only', 'eval_interval', 'eval_iters', 'log_interval', 'keep_checkpoints', 'checkpoint_steps', 'checkpoint_interval', 'eval_panel_path',
                   # A warm start changes how batches are processed, not the model or optimizer state.
                   'warm_start_fraction', 'warm_start_max_passes', 'warm_start_tolerance',
                   'live_eval_iters', 'live_eval_depths', 'loader_workers'}
        # Layout equivalence was checked from model_args, including legacy omissions.
        layout_keys = {'n_prelude', 'n_buffer', 'n_core', 'n_source', 'n_coda'} if recurrent else set()
        for key in DEFAULTS.keys() - mutable - layout_keys:
            if config[key] != saved_config.get(key, DEFAULTS[key]):
                raise ValueError(f'Resume changes {key}; use a new run for changed training settings')
        step, best_val = checkpoint['iter_num'], checkpoint['best_val_loss']
        last_eval_step = checkpoint['last_eval_step']
        training_seconds = checkpoint.get('training_seconds', 0.0)
        move_counts = checkpoint.get('move_counts', move_counts)
        continued = checkpoint.get('continued_from')
    model = (Recurrent2DGPT(RecurrentGPTConfig(**model_args)) if recurrent
             else GPT(GPTConfig(**model_args))).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        if sampler is not None:
            sampler.load_state_dict(checkpoint['recurrence_sampler'])
    if config['init_from'] == 'continue':
        # Stage 2 of the move-target plan: the trunk of an earlier run, a fresh output layer.
        source = torch.load(config['continue_from'], map_location='cpu', weights_only=False)
        same_trunk = (RecurrentGPTConfig.from_checkpoint(source['model_args']) == RecurrentGPTConfig(**model_args)
                      if recurrent else source['model_args'] == model_args)
        if not same_trunk:
            raise ValueError('continue_from has a different model configuration')
        trunk = {key: value for key, value in source['model'].items() if not key.startswith('lm_head.')}
        missing, unexpected = model.load_state_dict(trunk, strict=False)
        if unexpected or missing != ['lm_head.weight']:
            raise ValueError(f'continue_from does not match the model: missing {missing}, unexpected {unexpected}')
        continued = dict(path=str(config['continue_from']), sha256=file_hash(config['continue_from']),
                         iter_num=source['iter_num'], objective=source['config'].get('objective'),
                         dataset_manifest_hash=source['manifest_hash'])
        del source
    optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'],
                                          (config['beta1'], config['beta2']), device_type)
    scaler = torch.amp.GradScaler('cuda', enabled=(device_type == 'cuda' and config['dtype'] == 'float16'))
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
    raw_model = model
    if config['compile']:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])] if device_type == 'cuda' else None,
                    find_unused_parameters=recurrent)
    if checkpoint:
        restore_rng(checkpoint['rng_by_rank'][rank], train_rng, device)
    del checkpoint
    run_info = provenance()
    if panel:
        run_info.update(eval_panel_path=panel['path'], eval_panel_sha256=panel['sha256'],
                        eval_panel_split=panel['split'], eval_panel_row_count=panel['row_count'])
    effective_batch = config['batch_size'] * accumulation * world_size
    if master:
        record = dict(config=config, model_args=model_args, provenance=run_info, world_size=world_size,
                      manifest_hash=data.manifest_hash, effective_batch_size=effective_batch,
                      parameter_count=sum(p.numel() for p in raw_model.parameters()),
                      **({'row_tokens_per_step': effective_batch * config['block_size'],
                          'continued_from': continued} if moves else
                         {'characters_per_step': effective_batch * config['block_size']}),
                      resume_step=step,
                      eval_panel_path=panel['path'] if panel else '',
                      eval_panel_sha256=panel['sha256'] if panel else None,
                      eval_panel_split=panel['split'] if panel else None,
                      eval_panel_row_indices=panel['row_indices'] if panel else None)
        (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
        append_json(out / 'events.jsonl', dict(event='start', **record))

    def context():
        if device_type == 'cuda' and config['dtype'] != 'float32':
            return torch.autocast('cuda', dtype=getattr(torch, config['dtype']))
        return nullcontext()

    def move_batch(split, generator, random_fraction, one_game_per_row=False):
        return data.batch(split, config['batch_size'], device, generator, objective=config['objective'],
                          random_fraction=random_fraction, one_game_per_row=one_game_per_row)

    def live_logits(x, depth_steps):
        """Cached token-by-token live execution: the deployed model, not the training graph."""
        spec = validate_live_inference_spec(raw_model.config, depth_steps, live_strategy)
        state = create_live_state(raw_model, spec.depth_steps, spec.kv_strategy, batch_size=x.shape[0])
        # Rows hold one game each; nothing after the longest game is supervised.
        steps = int((x != PAD).sum(1).max())
        logits = torch.stack([decode_live_step(raw_model, x[:, t], state, spec) for t in range(steps)], 1)
        return F.pad(logits, (0, 0, 0, x.shape[1] - steps))

    def objective_loss(logits, targets):
        if isinstance(targets, torch.Tensor):
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return targets.loss(logits)

    def move_metrics(name, batches):
        """Pooled loss and objective metrics over (logits, loss, batch) triples."""
        loss_sum, positions, sums = 0.0, 0, {}
        for logits, loss, batch in batches:
            count = batch.counts['supervised_positions']
            loss_sum += loss.item() * count
            positions += count
            if config['objective'] == 'human':
                valid = batch.targets >= 0
                sums['correct'] = sums.get('correct', 0) + (logits.argmax(-1)[valid] == batch.targets[valid]).sum().item()
            else:
                for key, value in batch.targets.metrics(logits).items():
                    sums[key] = sums.get(key, 0.0) + value
        result = {name + '_loss': loss_sum / positions}
        if config['objective'] == 'human':
            result[name + '_accuracy'] = sums['correct'] / positions
        elif config['objective'] == 'legal':
            result.update({f'{name}_{key}': sums[key] / sums['positions']
                           for key in ('legal_bce', 'illegal_bce', 'constant_bce', 'exact_set')})
            result[name + '_precision'] = sums['true_positives'] / max(sums['predicted_positives'], 1)
            result[name + '_recall'] = sums['true_positives'] / sums['legal_moves']
        else:
            result.update({f'{name}_{key}': sums[key] / sums['positions'] for key in ('regret', 'best_move')})
        return result

    @torch.no_grad()
    def evaluate_moves():
        """Training-graph metrics on training rows, dev games and random games; live metrics on dev games.

        Dev and random games are evaluated one game per row, so no game reads another's state.
        """
        result = {}
        sets = [('train', 'train', config['random_game_fraction'], False, 10000), ('val', 'dev', 0.0, True, 20000)]
        if config['objective'] != 'engine':
            sets.append(('random', 'dev', 1.0, True, 30000))
        for name, split, random_fraction, alone, offset in sets:
            generator = torch.Generator().manual_seed(config['seed'] + offset)
            schedule_rng = random.Random(config['recurrence_seed'] + offset)
            scored = []
            for _ in range(config['eval_iters']):
                batch = move_batch(split, generator, random_fraction, alone)
                kwargs = dict(schedule=sample_schedule(config['eval_u_t'], config['eval_u_d'], schedule_rng)) if recurrent else {}
                with context():
                    logits, loss = raw_model(batch.x, batch.targets, **kwargs)
                scored.append((logits, loss, batch))
            result.update(move_metrics(name, scored))
        for depth_steps in live_depths:
            # The same dev games as the training-graph 'val' metrics, so the two are paired.
            generator = torch.Generator().manual_seed(config['seed'] + 20000)
            scored = []
            for _ in range(config['live_eval_iters']):
                batch = move_batch('dev', generator, 0.0, True)
                with context():
                    logits = live_logits(batch.x, depth_steps)
                scored.append((logits, objective_loss(logits, batch.targets), batch))
            result.update(move_metrics(f'live_J{depth_steps}_val', scored))
        return result

    @torch.no_grad()
    def evaluate():
        # Independent fixed batches make evaluations comparable and do not advance training RNG.
        raw_model.eval()
        if moves:
            result = evaluate_moves()
            raw_model.train()
            return result
        result = {}
        for split in ['train', 'val']:
            generator = torch.Generator().manual_seed(config['seed'] + (10000 if split == 'train' else 20000))
            schedule_rng = random.Random(config['recurrence_seed'] + (10000 if split == 'train' else 20000))
            total_loss, correct, total = 0.0, 0, 0
            for _ in range(config['eval_iters']):
                x, y = data.batch(split, config['batch_size'], device, generator,
                                  allowed_indices=panel_indices if split == 'val' else None)
                kwargs = dict(schedule=sample_schedule(config['eval_u_t'], config['eval_u_d'], schedule_rng)) if recurrent else {}
                with context():
                    logits, loss = raw_model(x, y, **kwargs)
                total_loss += loss.item() * y.numel()
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.numel()
            result[split + '_nll'] = total_loss / total
            result[split + '_accuracy'] = correct / total
        raw_model.train()
        return result

    def save_checkpoint(retain=True):
        state = capture_rng(train_rng, device)
        states = [None] * world_size
        if ddp:
            dist.all_gather_object(states, state)
        else:
            states[0] = state
        if master:
            payload = dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                             scaler=scaler.state_dict(), model_args=model_args, config=config,
                             iter_num=step, best_val_loss=best_val, last_eval_step=last_eval_step,
                             training_seconds=training_seconds,
                             rng_by_rank=states, world_size=world_size, manifest_hash=data.manifest_hash,
                             meta=data.meta, provenance=run_info,
                             eval_panel_path=panel['path'] if panel else '',
                             eval_panel_sha256=panel['sha256'] if panel else None,
                             eval_panel_split=panel['split'] if panel else None,
                             eval_panel_row_indices=panel['row_indices'] if panel else None,
                             recurrence_sampler=sampler.state_dict() if sampler else None)
            if moves:
                payload.update(move_counts=dict(move_counts), continued_from=continued)
            atomic_save(payload, out / 'ckpt.pt')
            if (retain and config['keep_checkpoints'] and
                    (config['checkpoint_steps'] is None or step in config['checkpoint_steps'])):
                atomic_save(payload, out / f'ckpt-step{step:06d}.pt')

    def next_schedule(probabilities):
        # Only rank zero advances the sampler. Its checkpoint state is authoritative.
        selected = [sampler.sample(probabilities) if master else None]
        if ddp:
            dist.broadcast_object_list(selected, src=0)
        return selected[0]

    if moves and not config['eval_only']:
        # Batch i of this rank is the same whatever the step it's built at, so resume just skips ahead.
        train_batches = training_batches(data, config['batch_size'], objective=config['objective'],
                                         random_fraction=config['random_game_fraction'], seed=config['seed'],
                                         rank=rank, start=step * accumulation, workers=config['loader_workers'])
    model.train()
    while True:
        exhausted = bool(budget and training_seconds >= budget)
        if config['eval_only'] or ((step % config['eval_interval'] == 0 or step == config['max_iters'] or exhausted) and step != last_eval_step):
            # All ranks participate; raw-model evaluation avoids DDP synchronization asymmetry.
            metrics = evaluate()
            best_val = min(best_val, metrics['val_loss' if moves else 'val_nll'])
            last_eval_step = step
            if master:
                if moves:
                    print(f'step {step}: ' + ', '.join(f'{key} {value:.4f}' for key, value in metrics.items()), flush=True)
                else:
                    print(f"step {step}: train {metrics['train_nll']:.4f}, val {metrics['val_nll']:.4f}, accuracy {metrics['val_accuracy']:.3f}", flush=True)
                label = dict(execution='training_graph', u_t=config['eval_u_t'], u_d=config['eval_u_d']) if recurrent else {}
                if recurrent:
                    active_matrix = update_probabilities_at_step(config, step)
                    label.update(next_update_probability_step=step,
                                 next_update_probability_matrix=[list(row) for row in active_matrix],
                                 last_update_probability_matrix=(
                                     [list(row) for row in update_probabilities_at_step(config, step - 1)]
                                     if step > 0 else None))
                append_json(out / 'metrics.jsonl', dict(event='evaluation', step=step, **label, **metrics))
            if not config['eval_only']:
                save_checkpoint()
        if config['eval_only'] or step >= config['max_iters'] or exhausted:
            break
        # A time-matched run decays by consumed training budget, not update count.
        lr_step = config['lr_decay_iters'] * training_seconds / budget if budget else step
        lr = get_lr(lr_step, config)
        for group in optimizer.param_groups:
            group['lr'] = lr
        if device_type == 'mps':
            torch.mps.synchronize()
        elif device_type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        loss_sum = 0.0
        final_loss_sum = 0.0
        intermediate_loss_sum = 0.0
        intermediate_weight = 0.0
        schedules = []
        probabilities = update_probabilities_at_step(config, step) if recurrent else None
        warm_microbatches = round(config['warm_start_fraction'] * accumulation)
        update_counts = dict.fromkeys(move_counts, 0)
        for micro_step in range(accumulation):
            if ddp:
                model.require_backward_grad_sync = micro_step == accumulation - 1
            if moves:
                batch = next(train_batches).to(device)
                x, y = batch.x, batch.targets
                for key, value in batch.counts.items():
                    update_counts[key] += value
            else:
                x, y = data.batch('train', config['batch_size'], device, train_rng)
            kwargs = {}
            if recurrent:
                schedule = next_schedule(probabilities)
                kwargs['schedule'] = schedule
                record = dict(u_t=schedule.u_t, u_d=schedule.u_d, rounds=schedule.rounds)
                if micro_step < warm_microbatches:
                    # Deterministic choice: the same microbatches, data and schedules as without warm start.
                    with context():
                        memory, record['warm_passes'] = raw_model.settle_temporal_memory(
                            x, config['warm_start_max_passes'], config['warm_start_tolerance'])
                    kwargs['initial_temporal_state'] = memory
                schedules.append(record)
            with context():
                if recurrent:
                    _, loss, components = model(
                        x, y, deep_supervision=config['deep_supervision'],
                        deep_supervision_lambda=config['deep_supervision_lambda'],
                        return_components=True, **kwargs)
                else:
                    _, loss = model(x, y, **kwargs)
                    components = dict(final_loss=loss, intermediate_loss=None)
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            loss_sum += loss.detach().item() / accumulation
            final_loss_sum += components['final_loss'].detach().item() / accumulation
            if components['intermediate_loss'] is not None:
                intermediate_loss_sum += components['intermediate_loss'].detach().item() / accumulation
                intermediate_weight += 1 / accumulation
            scaler.scale(scaled_loss).backward()
        gradient_stats = aggregate_gradient_stats(model, scaler, optimizer, config['grad_clip'], step)
        if moves:
            if ddp:
                totals = torch.tensor(list(update_counts.values()), dtype=torch.float64, device=device)
                dist.all_reduce(totals)
                update_counts = dict(zip(update_counts, (int(value) for value in totals.tolist())))
            for key, value in update_counts.items():
                move_counts[key] += value
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device_type == 'mps':
            torch.mps.synchronize()
        elif device_type == 'cuda':
            torch.cuda.synchronize()
        step += 1
        elapsed = time.perf_counter() - started
        training_seconds += elapsed
        retained_step = (config['keep_checkpoints'] and config['checkpoint_steps'] is not None and
                         step in config['checkpoint_steps'])
        recovery_step = bool(config['checkpoint_interval'] and step % config['checkpoint_interval'] == 0)
        if ((retained_step or recovery_step) and step % config['eval_interval'] != 0 and
                step != config['max_iters'] and not (budget and training_seconds >= budget)):
            # Save recovery/curve snapshots without evaluation; evaluation boundaries save above.
            # All ranks participate because save_checkpoint gathers RNG state under DDP.
            save_checkpoint(retain=retained_step)
        if master and (step % config['log_interval'] == 0 or step == 1):
            print(f'step {step}: loss {loss_sum:.4f}, {elapsed:.3f}s', flush=True)
            if moves:
                volume = dict(**move_counts, **{key + '_this_update': value for key, value in update_counts.items()},
                              row_tokens_per_second=update_counts['row_tokens'] / elapsed)
            else:
                volume = dict(characters_processed=step * effective_batch * config['block_size'],
                              characters_this_update=effective_batch * config['block_size'],
                              characters_per_second=effective_batch * config['block_size'] / elapsed)
            append_json(out / 'metrics.jsonl', dict(event='train', step=step, nll=loss_sum, lr=lr,
                                                   final_nll=final_loss_sum,
                                                   intermediate_nll=(intermediate_loss_sum / intermediate_weight
                                                                     if intermediate_weight else None),
                                                   deep_supervision=config['deep_supervision'],
                                                   deep_supervision_lambda=config['deep_supervision_lambda'],
                                                   seconds=elapsed, training_seconds=training_seconds, statistics='accumulated_update',
                                                   **gradient_stats, **volume,
                                                   schedules=schedules, microbatch_schedules=schedules))
    if ddp:
        dist.destroy_process_group()
    return out / 'ckpt.pt'


if __name__ == '__main__':
    namespace = dict(DEFAULTS)
    exec(Path('configurator.py').read_text(), namespace)
    train({key: namespace[key] for key in DEFAULTS})
