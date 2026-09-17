"""ChessGPT baseline and two-axis recurrent training, adapted from Karvonen/nanoGPT.

    uv run python train.py configs/smoke.py
    uv run python train.py configs/baseline_chessgpt.py --batch_size=16
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
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import RecurrenceScheduleSampler, sample_schedule
from training_utils import append_json, atomic_save, capture_rng, provenance, restore_rng

DEFAULTS = dict(
    out_dir='out-baseline', dataset='chess_v1', init_from='scratch',
    eval_interval=4000, eval_iters=100, log_interval=50, eval_only=False,
    n_layer=8, n_head=8, n_embd=512, block_size=1023, bias=False, dropout=0.0,
    batch_size=100, gradient_accumulation_steps=1, learning_rate=3e-4,
    max_iters=600000, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
    decay_lr=True, warmup_iters=2000, lr_decay_iters=600000, min_lr=3e-5,
    backend='nccl', device='cuda', dtype='bfloat16', compile=True, seed=1337,
    num_threads=4, architecture='baseline', n_prelude=2, n_core=4, n_coda=1,
    recurrence_support=[], recurrence_probabilities=[], recurrence_seed=1729,
    eval_u_t=0, eval_u_d=0,
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


def train(config):
    config = {**DEFAULTS, **config}
    if config['architecture'] not in ['baseline', 'recurrent']:
        raise ValueError('architecture must be baseline or recurrent')
    recurrent = config['architecture'] == 'recurrent'
    if recurrent and config['compile']:
        raise ValueError('Use compile=False for variable recurrent schedules in the MVP')
    sampler = RecurrenceScheduleSampler(config['recurrence_support'], config['recurrence_probabilities'],
                                         config['recurrence_seed']) if recurrent else None
    if recurrent:
        sample_schedule(config['eval_u_t'], config['eval_u_d'], random.Random(0))
    if config['init_from'] not in ['scratch', 'resume']:
        raise ValueError('init_from must be scratch or resume')
    for key in ['batch_size', 'gradient_accumulation_steps', 'eval_interval', 'eval_iters', 'log_interval', 'num_threads']:
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if config['max_iters'] < 0 or config['warmup_iters'] < 0:
        raise ValueError('Iteration counts must be nonnegative')
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
    out = Path(config['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if config['init_from'] == 'scratch' and (out / 'ckpt.pt').exists():
        raise FileExistsError('Checkpoint exists: choose a new out_dir or init_from=resume')
    model_args = {key: config[key] for key in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout']}
    model_args['vocab_size'] = data.meta['vocab_size']
    if recurrent:
        model_args.update({key: config[key] for key in ['n_prelude', 'n_core', 'n_coda']})
    checkpoint = None
    step, best_val, last_eval_step = 0, float('inf'), -1
    if config['init_from'] == 'resume':
        checkpoint = torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=False)
        if checkpoint['model_args'] != model_args:
            raise ValueError('Resume model configuration differs from checkpoint')
        if checkpoint['manifest_hash'] != data.manifest_hash:
            raise ValueError('Resume dataset differs from checkpoint')
        if checkpoint['world_size'] != world_size:
            raise ValueError('Exact resume requires the same world size')
        mutable = {'out_dir', 'max_iters', 'init_from', 'eval_only', 'eval_interval', 'eval_iters', 'log_interval'}
        for key in DEFAULTS.keys() - mutable:
            if config[key] != checkpoint['config'].get(key, DEFAULTS[key]):
                raise ValueError(f'Resume changes {key}; use a new run for changed training settings')
        step, best_val = checkpoint['iter_num'], checkpoint['best_val_loss']
        last_eval_step = checkpoint['last_eval_step']
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
    effective_batch = config['batch_size'] * accumulation * world_size
    if master:
        record = dict(config=config, model_args=model_args, provenance=run_info, world_size=world_size,
                      manifest_hash=data.manifest_hash, effective_batch_size=effective_batch,
                      parameter_count=sum(p.numel() for p in raw_model.parameters()),
                      characters_per_step=effective_batch * config['block_size'], resume_step=step)
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
                x, y = data.batch(split, config['batch_size'], device, generator)
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
            atomic_save(dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                             scaler=scaler.state_dict(), model_args=model_args, config=config,
                             iter_num=step, best_val_loss=best_val, last_eval_step=last_eval_step,
                             rng_by_rank=states, world_size=world_size, manifest_hash=data.manifest_hash,
                             meta=data.meta, provenance=run_info,
                             recurrence_sampler=sampler.state_dict() if sampler else None), out / 'ckpt.pt')

    def next_schedule():
        # Only rank zero advances the sampler. Its checkpoint state is authoritative.
        selected = [sampler.sample() if master else None]
        if ddp:
            dist.broadcast_object_list(selected, src=0)
        return selected[0]

    model.train()
    while True:
        if config['eval_only'] or ((step % config['eval_interval'] == 0 or step == config['max_iters']) and step != last_eval_step):
            # All ranks participate; raw-model evaluation avoids DDP synchronization asymmetry.
            metrics = evaluate()
            best_val = min(best_val, metrics['val_nll'])
            last_eval_step = step
            if master:
                print(f"step {step}: train {metrics['train_nll']:.4f}, val {metrics['val_nll']:.4f}, accuracy {metrics['val_accuracy']:.3f}", flush=True)
                label = dict(execution='training_graph', u_t=config['eval_u_t'], u_d=config['eval_u_d']) if recurrent else {}
                append_json(out / 'metrics.jsonl', dict(event='evaluation', step=step, **label, **metrics))
            if not config['eval_only']:
                save_checkpoint()
        if config['eval_only'] or step >= config['max_iters']:
            break
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group['lr'] = lr
        if device_type == 'mps':
            torch.mps.synchronize()
        elif device_type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        loss_sum = 0.0
        schedules = []
        for micro_step in range(accumulation):
            if ddp:
                model.require_backward_grad_sync = micro_step == accumulation - 1
            x, y = data.batch('train', config['batch_size'], device, train_rng)
            kwargs = {}
            if recurrent:
                schedule = next_schedule()
                kwargs['schedule'] = schedule
                schedules.append(dict(u_t=schedule.u_t, u_d=schedule.u_d, rounds=schedule.rounds))
            with context():
                _, loss = model(x, y, **kwargs)
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            loss_sum += loss.detach().item() / accumulation
            scaler.scale(scaled_loss).backward()
        if config['grad_clip']:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device_type == 'mps':
            torch.mps.synchronize()
        elif device_type == 'cuda':
            torch.cuda.synchronize()
        step += 1
        elapsed = time.perf_counter() - started
        if master and (step % config['log_interval'] == 0 or step == 1):
            print(f'step {step}: loss {loss_sum:.4f}, {elapsed:.3f}s', flush=True)
            append_json(out / 'metrics.jsonl', dict(event='train', step=step, nll=loss_sum, lr=lr,
                                                   seconds=elapsed, schedules=schedules, characters_per_second=effective_batch * config['block_size'] / elapsed))
    if ddp:
        dist.destroy_process_group()
    return out / 'ckpt.pt'


if __name__ == '__main__':
    namespace = dict(DEFAULTS)
    exec(Path('configurator.py').read_text(), namespace)
    train({key: namespace[key] for key in DEFAULTS})
