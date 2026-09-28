"""Post-hoc live alignment: retrain the temporal mixer on settled memory.

The procedure is fixed before the confirmation evaluation, exactly as tested on
the 128-row selection panel. It trains only ``temporal_mixer``; every
transformer block stays frozen. Each microbatch settles the temporal memory with
``--passes`` gradient-free training-graph passes, using the current mixer, then
backpropagates the next-character loss of one final pass that reads that memory.

A pass runs the buffer, ``depth_steps`` core iterations (the live depth loop,
equivalent to live execution with ``depth_specialized`` caches) and the
T-source. Temporal alignment uses one core iteration. The hybrid alternates one
and four iterations across the two microbatches of every update, covering both
deployed live settings.

Example:
    python -m experiments.ablations.live_warm_start.align --arm temporal \
        --checkpoint experiments/long_runs/20B_recurrence/temporal_20B/results/ckpt-step195504.pt \
        --output experiments/ablations/live_warm_start/results/temporal_aligned.pt --device cuda
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import ChessData, file_hash
from evaluation.live_inference import load_checkpoint_model
from models.recurrent_2d import shift_right

# The fixed procedure. Changing any value makes a different experiment.
PROCEDURE = dict(updates=200, microbatches=2, rows_per_microbatch=8, passes=64,
                 learning_rate=3e-4, warmup_updates=20, betas=(0.9, 0.95), weight_decay=0.0,
                 grad_clip=1.0, row_seed=4242, trainable='temporal_mixer')
DEPTH_STEPS = {'temporal': (1, 1), 'hybrid': (1, 4)}  # per microbatch within an update


def prelude(model, idx):
    p = model.embed(idx)
    for index in range(model.config.n_prelude):
        p = model.transformer.h[index](p)
    return p


def one_pass(model, p, memory, depth_steps=1):
    """One training-graph pass that reads ``memory`` (unshifted) and loops the core."""
    config, blocks = model.config, model.transformer.h
    anchor = p if memory is None else model.temporal_mixer(p, shift_right(memory))
    for index in range(config.n_prelude, config.core_start):
        anchor = blocks[index](anchor)
    h = anchor
    for depth in range(depth_steps):
        if depth:
            h = model.depth_mixer(h, anchor)
        for index in range(config.core_start, config.core_stop):
            h = blocks[index](h)
    return h


def settle(model, p, passes, depth_steps=1, memory=None):
    """Temporal memory after ``passes`` writes; exact live memory at positions below ``passes``."""
    for _ in range(passes):
        memory = model.temporal_source(one_pass(model, p, memory, depth_steps))
    return memory


def final_losses(model, p, memory, targets, depth_steps=1):
    """Per-position next-character NLL of the final pass reading ``memory``."""
    h = one_pass(model, p, memory, depth_steps)
    logits, _ = model.readout(model.coda(model.temporal_source(h)), targets)
    return F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')


def align(model, rows, arm, device, *, log=print):
    """Train ``model.temporal_mixer`` in place with the fixed procedure; return the loss history."""
    settings = PROCEDURE
    for param in model.parameters():
        param.requires_grad_(False)
    trainable = list(model.temporal_mixer.parameters())
    for param in trainable:
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=settings['learning_rate'], betas=settings['betas'],
                                  weight_decay=settings['weight_decay'])
    rng = np.random.default_rng(settings['row_seed'])
    context = model.config.block_size
    history = []
    started = time.monotonic()
    for update in range(1, settings['updates'] + 1):
        for group in optimizer.param_groups:
            group['lr'] = settings['learning_rate'] * min(1.0, update / settings['warmup_updates'])
        total = 0.0
        for micro in range(settings['microbatches']):
            depth_steps = DEPTH_STEPS[arm][micro % len(DEPTH_STEPS[arm])]
            indices = np.sort(rng.integers(0, len(rows), settings['rows_per_microbatch']))
            block = torch.from_numpy(np.array(rows[indices, :context + 1], dtype=np.int64))
            x, y = block[:, :-1].to(device), block[:, 1:].to(device)
            with torch.no_grad():
                memory = settle(model, prelude(model, x), settings['passes'], depth_steps)
            loss = final_losses(model, prelude(model, x), memory, y, depth_steps).mean()
            (loss / settings['microbatches']).backward()
            total += float(loss.detach()) / settings['microbatches']
        torch.nn.utils.clip_grad_norm_(trainable, settings['grad_clip'])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(total)
        if update % 20 == 0:
            log(f'{arm} update {update}: settled-memory loss {sum(history[-20:]) / 20:.4f}, '
                f'{time.monotonic() - started:.0f}s')
    for param in model.parameters():
        param.requires_grad_(True)
    return history


# Training-resume state that no longer matches the aligned weights.
RESUME_KEYS = ('optimizer', 'scaler', 'recurrence_sampler', 'rng_by_rank', 'training_seconds',
               'last_eval_step', 'best_val_loss')


def evaluation_checkpoint(source, state_dict, alignment):
    """An evaluation-only export: aligned weights, provenance, and no resume state."""
    checkpoint = {key: value for key, value in source.items() if key not in RESUME_KEYS}
    checkpoint['model'] = {key: value.detach().cpu() for key, value in state_dict.items()}
    checkpoint['alignment'] = alignment
    checkpoint['evaluation_only'] = True
    return checkpoint


def weights_sha256(state_dict):
    """Digest of the model tensors alone, stable across container changes."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', choices=sorted(DEPTH_STEPS), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f'{output} exists; refusing to overwrite')
    model, checkpoint = load_checkpoint_model(args.checkpoint, args.device)
    if model.config.recurrence_mode != args.arm:
        raise SystemExit(f'Checkpoint recurrence mode is {model.config.recurrence_mode}, not {args.arm}')
    data = ChessData(Path('data') / checkpoint['config']['dataset'], model.config.block_size,
                     verify_hashes=False)
    if data.manifest_hash != checkpoint['manifest_hash']:
        raise SystemExit('Dataset manifest differs from the checkpoint')
    model.train()
    history = align(model, data.rows['train'], args.arm, args.device)
    model.eval()
    alignment = dict(
        procedure={**PROCEDURE, 'depth_steps_per_microbatch': DEPTH_STEPS[args.arm]},
        source_checkpoint=str(Path(args.checkpoint)), source_sha256=file_hash(args.checkpoint),
        characters=PROCEDURE['updates'] * PROCEDURE['microbatches'] * PROCEDURE['rows_per_microbatch'] *
        model.config.block_size,
        loss_history=history, device=args.device, torch=torch.__version__,
        weights_sha256=weights_sha256(model.state_dict()))
    checkpoint = evaluation_checkpoint(checkpoint, model.state_dict(), alignment)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.rename(output)
    output.with_suffix('.json').write_text(json.dumps(
        {key: value for key, value in checkpoint['alignment'].items() if key != 'loss_history'}
        | dict(final_loss=sum(history[-20:]) / 20, output_sha256=file_hash(output)), indent=2) + '\n')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
