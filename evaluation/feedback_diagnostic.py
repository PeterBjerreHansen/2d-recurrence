"""Diagnostic swap of temporal feedback across independent validation rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import file_hash
from evaluation.all_rows import exact_validation_batches, load_model
from recurrence.schedule import RecurrenceSchedule
from training_utils import provenance


SCHEDULE = RecurrenceSchedule((True,), (False,))


def donor_permutation(target_count):
    """Return [donor0, donor1, ..., target0, target1, ...] for batch swapping."""
    if target_count < 1:
        raise ValueError('target_count must be positive')
    return list(range(target_count, 2 * target_count)) + list(range(target_count))


def row_metrics(logits, targets):
    batch, length, vocab = logits.shape
    losses = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction='none')
    losses = losses.reshape(batch, length).mean(dim=1)
    accuracy = (logits.argmax(-1) == targets).float().mean(dim=1)
    return losses.detach().cpu(), accuracy.detach().cpu()


def run_batch(model, x, y, device, permutation=None):
    handle = None
    call_count = {'value': 0}
    if permutation is not None:
        source_index = model.config.source_index
        source = model.transformer.h[source_index]

        def swap_first_source_output(module, inputs, output):
            if call_count['value'] == 0:
                output = output.index_select(0, permutation.to(output.device))
            call_count['value'] += 1
            return output

        handle = source.register_forward_hook(swap_first_source_output)
    try:
        with torch.no_grad():
            logits, _ = model(x.to(device), y.to(device), schedule=SCHEDULE)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite feedback diagnostic logits')
            return row_metrics(logits, y.to(device))
    finally:
        if handle is not None:
            handle.remove()


def evaluate_intervention(model, data, device):
    target_indices, _ = exact_validation_batches(data, 1)
    normal_rows = []
    corrupt_rows = []
    target_count = 2
    for start in range(0, len(target_indices), target_count):
        targets = target_indices[start:start + target_count]
        donors = [(index + 1) % len(target_indices) for index in targets]
        indices = targets + donors
        block = np.array(data.rows['val'][indices, :data.context_length + 1], dtype=np.int64)
        x = torch.from_numpy(block[:, :-1].copy())
        y = torch.from_numpy(block[:, 1:].copy())
        # Samples are [target0, target1, donor0, donor1]. Each target reads
        # feedback from the donor at the same sequence positions.
        permutation = torch.tensor(donor_permutation(len(targets)), dtype=torch.long)
        normal_loss, normal_accuracy = run_batch(model, x, y, device)
        corrupt_loss, corrupt_accuracy = run_batch(model, x, y, device, permutation)
        for offset, (target, donor) in enumerate(zip(targets, donors)):
            normal_rows.append(dict(row_index=target, donor_row_index=donor,
                                    nll=normal_loss[offset].item(),
                                    accuracy=normal_accuracy[offset].item()))
            corrupt_rows.append(dict(row_index=target, donor_row_index=donor,
                                     nll=corrupt_loss[offset].item(),
                                     accuracy=corrupt_accuracy[offset].item()))
    normal_by_row = {row['row_index']: row for row in normal_rows}
    corrupt_by_row = {row['row_index']: row for row in corrupt_rows}
    rows = []
    for index in target_indices:
        normal = normal_by_row[index]
        corrupt = corrupt_by_row[index]
        rows.append(dict(
            row_index=index,
            donor_row_index=normal['donor_row_index'],
            normal_nll=normal['nll'],
            corrupted_nll=corrupt['nll'],
            delta_nll=corrupt['nll'] - normal['nll'],
            normal_accuracy=normal['accuracy'],
            corrupted_accuracy=corrupt['accuracy'],
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True,
                        help='label=checkpoint path; repeat for each checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-threads', type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix('.csv').exists():
        parser.error('Output exists; choose a new report path')
    torch.set_num_threads(args.num_threads)
    specs = []
    for raw in args.checkpoint:
        if '=' not in raw:
            parser.error('--checkpoint must use label=path')
        specs.append(tuple(raw.split('=', 1)))

    reports = []
    fixed_metadata = None
    for label, path in specs:
        checkpoint, checkpoint_hash, data, model, recurrent = load_model(path, args.device)
        if not recurrent:
            raise ValueError('Feedback corruption requires a recurrent checkpoint')
        indices, _ = exact_validation_batches(data, 2)
        if fixed_metadata is None:
            fixed_metadata = dict(split='val', row_indices=indices, row_count=len(indices),
                                 manifest_hash=data.manifest_hash,
                                 context_length=data.context_length,
                                 sampling='each validation row is a target exactly once; donor is the next row modulo 41')
        elif data.manifest_hash != fixed_metadata['manifest_hash']:
            raise ValueError('Checkpoints do not share the same dataset manifest')
        rows = evaluate_intervention(model, data, args.device)
        for row in rows:
            row['label'] = label
        normal = sum(row['normal_nll'] for row in rows) / len(rows)
        corrupt = sum(row['corrupted_nll'] for row in rows) / len(rows)
        report = dict(label=label, checkpoint=str(Path(path).resolve()),
                      checkpoint_sha256=checkpoint_hash, checkpoint_step=checkpoint['iter_num'],
                      training_seed=checkpoint['config']['seed'], cell=[1, 0],
                      normal_feedback='source output from the same row',
                      corrupted_feedback='first temporal source output batch-swapped from an independent row at the same sequence positions',
                      rows=rows, summary=dict(normal_nll=normal, corrupted_nll=corrupt,
                                               delta_nll=corrupt - normal,
                                               normal_accuracy=sum(r['normal_accuracy'] for r in rows) / len(rows),
                                               corrupted_accuracy=sum(r['corrupted_accuracy'] for r in rows) / len(rows)))
        if file_hash(path) != checkpoint_hash:
            raise ValueError(f'{path}: checkpoint changed during evaluation')
        reports.append(report)
        print(f'{label}: normal {normal:.6f}, corrupted {corrupt:.6f}, delta {corrupt - normal:+.6f}')

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(execution='training_graph', intervention='diagnostic_feedback_swap',
                   fixed_validation=fixed_metadata, device=args.device, dtype='float32',
                   provenance=provenance(), checkpoints=reports)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    flat = []
    for report in reports:
        for row in report['rows']:
            flat.append(row)
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


if __name__ == '__main__':
    main()
