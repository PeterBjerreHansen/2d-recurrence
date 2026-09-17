"""Evaluate a recurrent checkpoint on fixed data and distinct write placements.

Run with: python -m evaluation.recurrence_grid --checkpoint ... --output ...
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

import torch

from data_loader import ChessData, file_hash
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import RecurrenceSchedule
from training_utils import provenance

PILOT_SUPPORT = (0, 1, 3)


def distinct_schedules(u_t, u_d, seeds):
    """Sample placements without replacement; deterministic cells run only once."""
    slots = max(u_t, u_d)
    masks = lambda count: [tuple(i in chosen for i in range(slots))
                           for chosen in itertools.combinations(range(slots), count)]
    remaining = [RecurrenceSchedule(t, d) for t in masks(u_t) for d in masks(u_d)]
    selected = []
    for seed in seeds:
        if not remaining:
            break
        selected.append((seed, remaining.pop(random.Random(seed).randrange(len(remaining)))))
    return selected


def compute_estimate(config, schedule, length):
    """Forward matrix-multiply FLOPs per sequence; multiply-add counts as two."""
    d = config.n_embd
    blocks = config.n_prelude + schedule.rounds * config.n_core + schedule.u_t + 1 + config.n_coda
    # A write after pass i is consumed from pass i+1 onward, even while held.
    def reads(mask):
        return len(mask) - mask.index(True) if any(mask) else 0
    temporal_reads, depth_reads = reads(schedule.temporal_write_mask), reads(schedule.depth_write_mask)
    flops = (blocks * (24 * length * d * d + 4 * length * length * d)
             + temporal_reads * 12 * length * d * d
             + depth_reads * 4 * length * d * d
             + 2 * length * d * config.vocab_size)
    return dict(block_applications=blocks, temporal_reads=temporal_reads, depth_reads=depth_reads,
                estimated_forward_matmul_flops_per_sequence=flops)


def trajectory_diagnostics(model, x, y, schedule):
    """One eval-mode backward probe; autograd.grad leaves parameter .grad untouched."""
    core_end = model.config.n_prelude + model.config.n_core - 1
    norms = {'core_output_rms': [], 'source_output_rms': []}
    handles = []
    for index, key in [(core_end, 'core_output_rms'), (core_end + 1, 'source_output_rms')]:
        def record(module, inputs, output, key=key):
            if not torch.isfinite(output).all():
                raise FloatingPointError(f'Non-finite {key}')
            norms[key].append(output.detach().float().square().mean().sqrt().item())
        handles.append(model.transformer.h[index].register_forward_hook(record))
    try:
        with torch.enable_grad():
            _, loss = model(x, y, schedule=schedule)
            named = list(model.named_parameters())
            gradients = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
        groups = {}
        for (name, _), gradient in zip(named, gradients):
            if name.startswith('transformer.h.'):
                index = int(name.split('.')[2])
                group = ('prelude' if index < model.config.n_prelude else
                         'core' if index <= core_end else 'source' if index == core_end + 1 else 'coda')
            else:
                group = name.split('.')[0] if 'mixer' in name else 'embedding_and_head'
            groups.setdefault(group, None)
            if gradient is not None:
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f'Non-finite gradient: {name}')
                groups[group] = (groups[group] or 0.0) + gradient.float().square().sum().item()
        return dict(**norms, gradient_l2={k: None if v is None else v ** .5 for k, v in groups.items()},
                    loss=loss.item(), mode='eval', batch_count=1)
    finally:
        for handle in handles:
            handle.remove()


def evaluate_grid(model, data, *, batches=8, batch_size=2, data_seed=2027,
                  mask_seeds=(11, 23, 37), training_probabilities=None, diagnostics=True):
    if batches < 1 or batch_size < 1 or not mask_seeds or len(set(mask_seeds)) != len(mask_seeds):
        raise ValueError('Positive batch counts and nonempty distinct mask seeds are required')
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(data_seed)
    fixed = [data.batch('val', batch_size, 'cpu', generator) for _ in range(batches)]
    digest = hashlib.sha256()
    for x, y in fixed:
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    was_training = model.training
    model.eval()
    cells, baseline_predictions = [], []
    try:
        for u_t, u_d in itertools.product(PILOT_SUPPORT, repeat=2):
            placements = []
            for seed, schedule in distinct_schedules(u_t, u_d, mask_seeds):
                total_loss, correct, changed, count = 0., 0, 0, 0
                with torch.no_grad():
                    for index, (cpu_x, cpu_y) in enumerate(fixed):
                        x, y = cpu_x.to(device), cpu_y.to(device)
                        logits, loss = model(x, y, schedule=schedule)
                        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                            raise FloatingPointError(f'Non-finite grid cell {(u_t, u_d)}')
                        predictions = logits.argmax(-1).cpu()
                        if (u_t, u_d) == (0, 0):
                            baseline_predictions.append(predictions)
                        changed += (predictions != baseline_predictions[index]).sum().item()
                        total_loss += loss.item() * y.numel()
                        correct += (predictions == cpu_y).sum().item()
                        count += y.numel()
                placements.append(dict(mask_seed=seed, temporal_write_mask=schedule.temporal_write_mask,
                                       depth_write_mask=schedule.depth_write_mask, nll=total_loss / count,
                                       accuracy=correct / count, prediction_change_rate=changed / count,
                                       **compute_estimate(model.config, schedule, data.context_length)))
            stats = {}
            for key in ['nll', 'accuracy', 'prediction_change_rate', 'estimated_forward_matmul_flops_per_sequence']:
                values = [p[key] for p in placements]
                stats[key + '_mean'] = statistics.mean(values)
                stats[key + '_std'] = statistics.pstdev(values)
            cell = dict(u_t=u_t, u_d=u_d, core_passes=max(u_t, u_d) + 1,
                        training_probability=training_probabilities.get((u_t, u_d), 0.) if training_probabilities is not None else None,
                        possible_placements=math.comb(max(u_t, u_d), u_t) * math.comb(max(u_t, u_d), u_d),
                        placement_count=len(placements), evaluated_characters_per_placement=count,
                        **stats, placements=placements)
            cell['all_placements_evaluated'] = len(placements) == cell['possible_placements']
            cell['nll_delta_vs_00'] = stats['nll_mean'] - (cells[0]['nll_mean'] if cells else stats['nll_mean'])
            if diagnostics:
                first = placements[0]
                schedule = RecurrenceSchedule(first['temporal_write_mask'], first['depth_write_mask'])
                cell['diagnostics'] = dict(mask_seed=first['mask_seed'], **trajectory_diagnostics(
                    model, fixed[0][0].to(device), fixed[0][1].to(device), schedule))
            cells.append(cell)
    finally:
        model.train(was_training)
    return dict(execution='training_graph', split='val', data_seed=data_seed, mask_seeds=list(mask_seeds),
                sampling='fixed row-aligned batches sampled with replacement', batch_size=batch_size,
                batches=batches, batch_sha256=digest.hexdigest(), context_length=data.context_length,
                variation='population standard deviation across distinct mask placements, not training seeds or batches',
                compute_convention='Forward matmul estimate per sequence, multiply-add=2; dense T-by-T attention. '
                'Includes transformer projections/MLPs, attention, temporal gates/values, depth values, LM head. '
                'Excludes normalization, softmax, activations, elementwise operations, backward, and kernel overhead; '
                'not measured hardware FLOPs or a complete compute comparison.', cells=cells)


def evaluate_checkpoint(checkpoint_path, *, device='cpu', dataset=None, **kwargs):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = file_hash(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['config'].get('architecture') != 'recurrent':
        raise ValueError('The recurrence grid requires a recurrent checkpoint')
    config = checkpoint['config']
    data = ChessData(Path('data') / (dataset or config['dataset']), checkpoint['model_args']['block_size'])
    if data.manifest_hash != checkpoint['manifest_hash'] or data.meta != checkpoint['meta']:
        raise ValueError('Evaluation data or vocabulary differs from the training checkpoint')
    model = Recurrent2DGPT(RecurrentGPTConfig(**checkpoint['model_args'])).to(device)
    model.load_state_dict(checkpoint['model'])
    probabilities = {(t, d): config['recurrence_probabilities'][i][j]
                     for i, t in enumerate(config['recurrence_support'])
                     for j, d in enumerate(config['recurrence_support'])}
    report = evaluate_grid(model, data, training_probabilities=probabilities, **kwargs)
    if file_hash(checkpoint_path) != checkpoint_hash:
        raise ValueError('Checkpoint changed during evaluation; use a retained step checkpoint')
    report.update(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_hash,
                  checkpoint_step=checkpoint['iter_num'], manifest_hash=data.manifest_hash,
                  model_args=checkpoint['model_args'], training_seed=config['seed'],
                  training_schedule_seed=config['recurrence_seed'], device=device,
                  dtype='float32', provenance=provenance())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='JSON report; a flat CSV is written alongside it')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dataset', help='Optional relocated data directory, checked against the saved manifest')
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--data-seed', type=int, default=2027)
    parser.add_argument('--mask-seeds', type=int, nargs='+', default=[11, 23, 37])
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--no-diagnostics', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix != '.json':
        parser.error('--output must end with .json')
    if output.exists() or output.with_suffix('.csv').exists():
        parser.error('Output exists; choose a new report path')
    torch.set_num_threads(args.num_threads)
    report = evaluate_checkpoint(args.checkpoint, device=args.device, dataset=args.dataset,
                                 batches=args.batches, batch_size=args.batch_size, data_seed=args.data_seed,
                                 mask_seeds=args.mask_seeds, diagnostics=not args.no_diagnostics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    rows = [{k: v for k, v in cell.items() if k not in ['placements', 'diagnostics']} for cell in report['cells']]
    with output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for cell in report['cells']:
        print(f"({cell['u_t']},{cell['u_d']}): NLL {cell['nll_mean']:.5f} ± {cell['nll_std']:.5f}, "
              f"accuracy {cell['accuracy_mean']:.3f}, ΔNLL {cell['nll_delta_vs_00']:+.5f}")


if __name__ == '__main__':
    main()
