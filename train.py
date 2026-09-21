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
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import ChessData
from model import GPT, GPTConfig
from models.recurrent_2d import (Recurrent2DGPT, RecurrentGPTConfig,
                                 validate_recurrence_counts, validate_recurrence_distribution,
                                 validate_recurrence_mode)
from recurrence.schedule import (RecurrenceScheduleSampler, probabilities_at_step,
                                 sample_schedule)
from training_utils import append_json, atomic_save, capture_rng, provenance, restore_rng

DEFAULTS = dict(
    out_dir='experiments/manual/results', dataset='chess_v1', init_from='scratch',
    eval_interval=4000, eval_iters=100, log_interval=50, eval_only=False,
    n_layer=8, n_head=8, n_embd=512, block_size=1023, bias=False, dropout=0.0,
    batch_size=100, gradient_accumulation_steps=1, learning_rate=3e-4,
    max_iters=600000, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
    decay_lr=True, warmup_iters=2000, lr_decay_iters=600000, min_lr=3e-5,
    backend='nccl', device='cuda', dtype='bfloat16', compile=True, seed=1337,
    num_threads=4, architecture='baseline', n_prelude=1, n_core=4, n_coda=1, n_buffer=1, n_source=1,
    recurrence_support=[], recurrence_probabilities=[], recurrence_seed=1729, recurrence_mode='hybrid',
    recurrence_probability_schedule=None,
    eval_u_t=0, eval_u_d=0, keep_checkpoints=False, checkpoint_steps=None,
    eval_panel_path='', deep_supervision=False, deep_supervision_lambda=0.25,
    training_budget_seconds=0.0,
)


def get_lr(step, config):
    if not config['decay_lr']:
        return config['learning_rate']
    if step < config['warmup_iters']:
        return config['learning_rate'] * step / config['warmup_iters']
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
    config = {**DEFAULTS, **config}
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
        initial_probabilities = probabilities_at_step(config, 0)
        sampler = RecurrenceScheduleSampler(config['recurrence_support'],
                                             None if config['recurrence_probability_schedule'] is not None
                                             else config['recurrence_probabilities'],
                                             config['recurrence_seed'])
        validate_recurrence_distribution(config['recurrence_mode'], config['recurrence_support'],
                                         initial_probabilities)
        validate_recurrence_counts(config['recurrence_mode'], config['eval_u_t'], config['eval_u_d'])
        sample_schedule(config['eval_u_t'], config['eval_u_d'], random.Random(0))
    else:
        sampler = None
    if config['init_from'] not in ['scratch', 'resume']:
        raise ValueError('init_from must be scratch or resume')
    for key in ['batch_size', 'gradient_accumulation_steps', 'eval_interval', 'eval_iters', 'log_interval', 'num_threads']:
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if config['max_iters'] < 0 or config['warmup_iters'] < 0:
        raise ValueError('Iteration counts must be nonnegative')
    if config['checkpoint_steps'] is not None:
        if (not config['checkpoint_steps'] or
                any(type(step) is not int or step < 0 for step in config['checkpoint_steps']) or
                len(set(config['checkpoint_steps'])) != len(config['checkpoint_steps'])):
            raise ValueError('checkpoint_steps must be a nonempty list of distinct nonnegative integers')
    if config['decay_lr'] and config['lr_decay_iters'] <= config['warmup_iters']:
        raise ValueError('lr_decay_iters must exceed warmup_iters')
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
    data = ChessData(Path('data') / config['dataset'], config['block_size'])
    panel = None
    panel_indices = None
    if config['eval_panel_path']:
        from evaluation.panels import load_panel
        panel = load_panel(config['eval_panel_path'], data, split='selection')
        panel_indices = panel['row_indices']
    out = Path(config['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if config['init_from'] == 'scratch' and (out / 'ckpt.pt').exists():
        raise FileExistsError('Checkpoint exists: choose a new out_dir or init_from=resume')
    model_args = {key: config[key] for key in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout']}
    model_args['vocab_size'] = data.meta['vocab_size']
    if recurrent:
        model_args.update({key: config[key] for key in ['n_prelude', 'n_buffer', 'n_core', 'n_source', 'n_coda',
                                                        'recurrence_mode']})
    checkpoint = None
    step, best_val, last_eval_step = 0, float('inf'), -1
    training_seconds = 0.0
    if config['init_from'] == 'resume':
        checkpoint = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=False)
        saved_args = checkpoint['model_args']
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
        mutable = {'out_dir', 'max_iters', 'init_from', 'eval_only', 'eval_interval', 'eval_iters', 'log_interval', 'keep_checkpoints', 'checkpoint_steps', 'eval_panel_path'}
        # Layout equivalence was checked from model_args, including legacy omissions.
        layout_keys = {'n_prelude', 'n_buffer', 'n_core', 'n_source', 'n_coda'} if recurrent else set()
        for key in DEFAULTS.keys() - mutable - layout_keys:
            if config[key] != checkpoint['config'].get(key, DEFAULTS[key]):
                raise ValueError(f'Resume changes {key}; use a new run for changed training settings')
        step, best_val = checkpoint['iter_num'], checkpoint['best_val_loss']
        last_eval_step = checkpoint['last_eval_step']
        training_seconds = checkpoint.get('training_seconds', 0.0)
    model = (Recurrent2DGPT(RecurrentGPTConfig(**model_args)) if recurrent
             else GPT(GPTConfig(**model_args))).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        if sampler is not None:
            sampler.load_state_dict(checkpoint['recurrence_sampler'])
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
                      characters_per_step=effective_batch * config['block_size'], resume_step=step,
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

    @torch.no_grad()
    def evaluate():
        # Independent fixed batches make evaluations comparable and do not advance training RNG.
        raw_model.eval()
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

    def save_checkpoint():
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
            atomic_save(payload, out / 'ckpt.pt')
            if (config['keep_checkpoints'] and
                    (config['checkpoint_steps'] is None or step in config['checkpoint_steps'])):
                atomic_save(payload, out / f'ckpt-step{step:06d}.pt')

    def next_schedule(probabilities):
        # Only rank zero advances the sampler. Its checkpoint state is authoritative.
        selected = [sampler.sample(probabilities) if master else None]
        if ddp:
            dist.broadcast_object_list(selected, src=0)
        return selected[0]

    model.train()
    while True:
        exhausted = bool(budget and training_seconds >= budget)
        if config['eval_only'] or ((step % config['eval_interval'] == 0 or step == config['max_iters'] or exhausted) and step != last_eval_step):
            # All ranks participate; raw-model evaluation avoids DDP synchronization asymmetry.
            metrics = evaluate()
            best_val = min(best_val, metrics['val_nll'])
            last_eval_step = step
            if master:
                print(f"step {step}: train {metrics['train_nll']:.4f}, val {metrics['val_nll']:.4f}, accuracy {metrics['val_accuracy']:.3f}", flush=True)
                label = dict(execution='training_graph', u_t=config['eval_u_t'], u_d=config['eval_u_d']) if recurrent else {}
                if recurrent:
                    active_matrix = probabilities_at_step(config, step)
                    label.update(training_probability_step=step,
                                 training_probability_matrix=[list(row) for row in active_matrix])
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
        probabilities = probabilities_at_step(config, step) if recurrent else None
        for micro_step in range(accumulation):
            if ddp:
                model.require_backward_grad_sync = micro_step == accumulation - 1
            x, y = data.batch('train', config['batch_size'], device, train_rng)
            kwargs = {}
            if recurrent:
                schedule = next_schedule(probabilities)
                kwargs['schedule'] = schedule
                schedules.append(dict(u_t=schedule.u_t, u_d=schedule.u_d, rounds=schedule.rounds))
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
        if (config['keep_checkpoints'] and config['checkpoint_steps'] is not None and
                step in config['checkpoint_steps'] and step != last_eval_step):
            # Retain requested curve snapshots even when they are not eval_interval boundaries.
            # All ranks participate because save_checkpoint gathers RNG state under DDP.
            save_checkpoint()
        if master and (step % config['log_interval'] == 0 or step == 1):
            print(f'step {step}: loss {loss_sum:.4f}, {elapsed:.3f}s', flush=True)
            append_json(out / 'metrics.jsonl', dict(event='train', step=step, nll=loss_sum, lr=lr,
                                                   final_nll=final_loss_sum,
                                                   intermediate_nll=(intermediate_loss_sum / intermediate_weight
                                                                     if intermediate_weight else None),
                                                   deep_supervision=config['deep_supervision'],
                                                   deep_supervision_lambda=config['deep_supervision_lambda'],
                                                   seconds=elapsed, training_seconds=training_seconds, statistics='accumulated_update',
                                                   **gradient_stats,
                                                   characters_processed=step * effective_batch * config['block_size'],
                                                   characters_this_update=effective_batch * config['block_size'],
                                                   schedules=schedules, microbatch_schedules=schedules,
                                                   characters_per_second=effective_batch * config['block_size'] / elapsed))
    if ddp:
        dist.destroy_process_group()
    return out / 'ckpt.pt'


if __name__ == '__main__':
    namespace = dict(DEFAULTS)
    exec(Path('configurator.py').read_text(), namespace)
    train({key: namespace[key] for key in DEFAULTS})
