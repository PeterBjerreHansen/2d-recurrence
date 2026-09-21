"""Evaluate checkpoints on every stored validation row exactly once.

Recurrent checkpoints receive the full nine-cell pilot grid. Ordinary
checkpoints receive the equivalent one-pass evaluation on the same batches.
Each ``--checkpoint`` argument is ``label=path``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_loader import ChessData, file_hash
from evaluation.recurrence_grid import evaluate_grid
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import probability_map_at_step, probabilities_at_step
from training_utils import provenance


def exact_validation_batches(data, batch_size):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    indices = list(range(len(data.rows['val'])))
    batches = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        block = np.array(data.rows['val'][selected, :data.context_length + 1], dtype=np.int64)
        batches.append((torch.from_numpy(block[:, :-1].copy()),
                        torch.from_numpy(block[:, 1:].copy())))
    return indices, batches


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = file_hash(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    data = ChessData(Path('data') / config['dataset'], checkpoint['model_args']['block_size'])
    if data.manifest_hash != checkpoint['manifest_hash'] or data.meta != checkpoint['meta']:
        raise ValueError(f'{checkpoint_path}: evaluation data or vocabulary differs from checkpoint')
    recurrent = config.get('architecture') == 'recurrent'
    if recurrent:
        model = Recurrent2DGPT(RecurrentGPTConfig.from_checkpoint(checkpoint['model_args']))
    elif config.get('architecture') in (None, 'baseline'):
        model = GPT(GPTConfig(**checkpoint['model_args']))
    else:
        raise ValueError(f'{checkpoint_path}: unsupported checkpoint architecture')
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return checkpoint, checkpoint_hash, data, model, recurrent


def evaluate_baseline(model, fixed_batches, device):
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for cpu_x, cpu_y in fixed_batches:
            x, y = cpu_x.to(device), cpu_y.to(device)
            logits, loss = model(x, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError('Non-finite ordinary baseline evaluation')
            total_loss += loss.item() * y.numel()
            total_correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
    return dict(nll=total_loss / total, accuracy=total_correct / total,
                evaluated_characters=total)


def training_probabilities(config, step=0):
    return probability_map_at_step(config, step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True,
                        help='label=checkpoint path; repeat for each checkpoint')
    parser.add_argument('--output', required=True, help='JSON report; a flat CSV is written alongside it')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--mask-seeds', type=int, nargs='+', default=[11, 23, 37])
    parser.add_argument('--num-threads', type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix != '.json' or output.exists() or output.with_suffix('.csv').exists():
        parser.error('--output must be a new .json path')
    specs = []
    for raw in args.checkpoint:
        if '=' not in raw:
            parser.error('--checkpoint must use label=path')
        label, path = raw.split('=', 1)
        if not label or not path:
            parser.error('--checkpoint must use nonempty label=path')
        specs.append((label, path))
    if len(set(label for label, _ in specs)) != len(specs):
        parser.error('checkpoint labels must be distinct')
    torch.set_num_threads(args.num_threads)

    results = []
    fixed_metadata = None
    for label, path in specs:
        checkpoint, checkpoint_hash, data, model, recurrent = load_model(path, args.device)
        row_indices, fixed = exact_validation_batches(data, args.batch_size)
        if fixed_metadata is None:
            fixed_digest = hashlib.sha256()
            for x, y in fixed:
                fixed_digest.update(x.numpy().tobytes())
                fixed_digest.update(y.numpy().tobytes())
            fixed_metadata = dict(
                split='val', row_indices=row_indices, row_count=len(row_indices),
                batch_size=args.batch_size, batch_count=len(fixed),
                batch_sha256=fixed_digest.hexdigest(),
                context_length=data.context_length,
                manifest_hash=data.manifest_hash,
                sampling='every stored validation row exactly once in canonical row order',
            )
        elif data.manifest_hash != fixed_metadata['manifest_hash']:
            raise ValueError('Checkpoints do not share the same dataset manifest')

        if recurrent:
            active_matrix = probabilities_at_step(checkpoint['config'], checkpoint['iter_num'])
            report = evaluate_grid(
                model, data, fixed_batches=fixed, data_seed=None, batch_size=args.batch_size,
                mask_seeds=args.mask_seeds,
                training_probabilities=training_probabilities(checkpoint['config'], checkpoint['iter_num']),
                diagnostics=False,
                sampling=fixed_metadata['sampling'],
            )
            report['row_indices'] = row_indices
            report['training_probability_step'] = checkpoint['iter_num']
            report['training_probability_matrix'] = [list(row) for row in active_matrix]
            result = dict(label=label, checkpoint=str(Path(path).resolve()),
                          checkpoint_sha256=checkpoint_hash,
                          checkpoint_step=checkpoint['iter_num'], architecture='recurrent',
                          training_seed=checkpoint['config']['seed'],
                          recurrence_mode=model.config.recurrence_mode, report=report)
        else:
            metrics = evaluate_baseline(model, fixed, args.device)
            result = dict(label=label, checkpoint=str(Path(path).resolve()),
                          checkpoint_sha256=checkpoint_hash,
                          checkpoint_step=checkpoint['iter_num'], architecture='baseline',
                          training_seed=checkpoint['config']['seed'], metrics=metrics)
        if file_hash(path) != checkpoint_hash:
            raise ValueError(f'{path}: checkpoint changed during evaluation')
        results.append(result)
        print(f'{label}: {result.get("architecture")} step {result["checkpoint_step"]}')

    report = dict(
        execution='training_graph' if any(r['architecture'] == 'recurrent' for r in results) else 'baseline',
        fixed_validation=fixed_metadata,
        mask_seeds=args.mask_seeds,
        device=args.device,
        dtype='float32',
        provenance=provenance(),
        checkpoints=results,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    rows = []
    for result in results:
        if result['architecture'] == 'baseline':
            rows.append(dict(label=result['label'], checkpoint_step=result['checkpoint_step'],
                             architecture='baseline', u_t=0, u_d=0, placement_count=1,
                             nll=result['metrics']['nll'], accuracy=result['metrics']['accuracy'],
                             nll_delta_vs_00=0.0))
        else:
            for cell in result['report']['cells']:
                rows.append(dict(label=result['label'], checkpoint_step=result['checkpoint_step'],
                                 architecture='recurrent', u_t=cell['u_t'], u_d=cell['u_d'],
                                 placement_count=cell['placement_count'], nll=cell['nll_mean'],
                                 accuracy=cell['accuracy_mean'],
                                 nll_delta_vs_00=cell['nll_delta_vs_00']))
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
