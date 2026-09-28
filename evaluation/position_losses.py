"""Per-position training-graph losses for any checkpoint and any recurrence cells.

Saves ``cell_<U_T>_<U_D>.npz`` (and a ``.json`` summary) per cell with per-target losses and
correctness for every selected validation row, so later analysis can compare
arms row by row and stratify by character role or ply. Ordinary checkpoints
support only ``(0, 0)``. Cells with several possible write placements average
the per-position losses over the distinct placements drawn from the mask seeds.

Example (every validation row, hybrid primary and single-axis cells):
    python -m evaluation.position_losses --checkpoint .../ckpt-step195504.pt \
        --output-dir .../battery/hybrid --cell 3,3 --cell 3,0 --cell 0,3 --device cuda
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import ChessData, file_hash
from evaluation.live_inference import load_checkpoint_model
from evaluation.recurrence_grid import compute_estimate, distinct_schedules, _parse_cell
from evaluation.row_selection import row_arrays, select_validation_rows
from models.recurrent_2d import validate_recurrence_counts
from recurrence.schedule import RecurrenceSchedule
from training_utils import provenance


def load_evaluation_target(checkpoint_path, device, *, data_dir=None, verify_data=False):
    """Load a checkpoint and its dataset, refusing a mismatched manifest or vocabulary."""
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    directory = Path(data_dir) if data_dir else Path('data') / checkpoint['config']['dataset']
    data = ChessData(directory, checkpoint['model_args']['block_size'], verify_hashes=verify_data)
    if data.manifest_hash != checkpoint['manifest_hash'] or data.meta != checkpoint['meta']:
        raise ValueError(f'{checkpoint_path}: evaluation data or vocabulary differs from checkpoint')
    return model, checkpoint, data


def checkpoint_identity(checkpoint_path, checkpoint, model):
    return dict(checkpoint=str(Path(checkpoint_path).resolve()), checkpoint_sha256=file_hash(checkpoint_path),
                checkpoint_step=checkpoint['iter_num'], training_seed=checkpoint['config'].get('seed'),
                architecture=checkpoint['config'].get('architecture', 'baseline'),
                recurrence_mode=getattr(model.config, 'recurrence_mode', 'baseline'),
                model_args=checkpoint['model_args'], manifest_hash=checkpoint['manifest_hash'])


def save_npz_atomic(path, **arrays):
    """Write an .npz beside ``path`` and move it into place, so readers never see a partial file."""
    path = Path(path)
    temporary = path.with_name(path.name[:-len('.npz')] + '.tmp.npz')
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def write_text_atomic(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def validate_cells(model, cells):
    mode = getattr(model.config, 'recurrence_mode', 'baseline')
    for u_t, u_d in cells:
        if mode == 'baseline':
            if (u_t, u_d) != (0, 0):
                raise ValueError('Ordinary checkpoints support only the (0, 0) cell')
        else:
            validate_recurrence_counts(mode, u_t, u_d)
    if len(set(cells)) != len(cells):
        raise ValueError('Evaluation cells must be distinct')


@torch.no_grad()
def cell_losses(model, inputs, targets, cell, *, batch_size, mask_seeds, device, progress=None):
    """Return per-position losses [rows, T] float32, correctness, and the placements used."""
    recurrent = hasattr(model.config, 'recurrence_mode')
    schedules = distinct_schedules(*cell, mask_seeds) if recurrent else [(None, None)]
    losses = np.zeros(targets.shape, dtype=np.float64)
    correct = np.zeros(targets.shape, dtype=np.float64)
    for _, schedule in schedules:
        for start in range(0, len(inputs), batch_size):
            x = torch.from_numpy(inputs[start:start + batch_size]).to(device)
            y = torch.from_numpy(targets[start:start + batch_size]).to(device)
            logits, _ = model(x, y, schedule=schedule) if recurrent else model(x, y)
            loss = F.cross_entropy(logits.transpose(1, 2).float(), y, reduction='none')
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f'Non-finite loss in cell {cell}')
            losses[start:start + len(y)] += loss.cpu().numpy()
            correct[start:start + len(y)] += (logits.argmax(-1) == y).cpu().numpy()
            if progress:
                progress(start + len(y))
    placements = [dict(mask_seed=seed, temporal_write_mask=schedule.temporal_write_mask,
                       depth_write_mask=schedule.depth_write_mask)
                  for seed, schedule in schedules if schedule is not None]
    return (losses / len(schedules)).astype(np.float32), correct / len(schedules), placements


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cell', action='append', type=_parse_cell, required=True,
                        help='U_T,U_D; repeat for several cells')
    parser.add_argument('--panel-file', help='Frozen panel; omit to use every validation row')
    parser.add_argument('--panel-split', choices=['selection', 'confirmation'], default='confirmation')
    parser.add_argument('--limit', type=int, help='Seeded subset of the selected rows')
    parser.add_argument('--subset-seed', type=int, default=0)
    parser.add_argument('--data-dir', help='Dataset directory; defaults to the checkpoint dataset')
    parser.add_argument('--verify-data', action='store_true', help='Hash train.bin/val.bin before use')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--mask-seeds', type=int, nargs='+', default=[11, 23, 37])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    output_dir = Path(args.output_dir)
    paths = {cell: output_dir / f'cell_{cell[0]}_{cell[1]}.npz' for cell in args.cell}
    if any(path.exists() for path in paths.values()):
        parser.error('An output cell file exists; choose a new --output-dir')

    model, checkpoint, data = load_evaluation_target(args.checkpoint, args.device, data_dir=args.data_dir,
                                                     verify_data=args.verify_data)
    validate_cells(model, args.cell)
    indices, selection = select_validation_rows(data, panel_file=args.panel_file, panel_split=args.panel_split,
                                                limit=args.limit, seed=args.subset_seed)
    inputs, targets = row_arrays(data, indices)
    identity = checkpoint_identity(args.checkpoint, checkpoint, model)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = dict(execution='training_graph', rows=selection, device=args.device, dtype='float32',
                  batch_size=args.batch_size, mask_seeds=args.mask_seeds, provenance=provenance(), **identity)
    for cell in args.cell:
        started = time.monotonic()
        last = [started]

        def progress(done):
            if time.monotonic() - last[0] > 60:
                last[0] = time.monotonic()
                print(f'  cell {cell}: {done}/{len(inputs)} rows', flush=True)

        losses, correct, placements = cell_losses(model, inputs, targets, cell, batch_size=args.batch_size,
                                                  mask_seeds=args.mask_seeds, device=args.device,
                                                  progress=progress)
        meta = dict(execution='training_graph', u_t=cell[0], u_d=cell[1], placements=placements,
                    rows=selection, **identity)
        summary = dict(u_t=cell[0], u_d=cell[1], file=paths[cell].name, nll=float(losses.mean(dtype=np.float64)),
                       accuracy=float(correct.mean()), placements=placements,
                       target_count=int(losses.size), seconds=time.monotonic() - started,
                       **(compute_estimate(model.config, _schedule(placements), data.context_length)
                          if placements else {}), **common)
        write_text_atomic(paths[cell].with_suffix('.json'), json.dumps(summary, indent=2, allow_nan=False) + '\n')
        # The .npz is published last: its existence marks a completed cell.
        save_npz_atomic(paths[cell], row_indices=np.asarray(indices, dtype=np.int64), losses=losses,
                        correct=correct.astype(np.float32), meta=json.dumps(meta))
        print(f"({cell[0]},{cell[1]}): NLL {summary['nll']:.5f} accuracy {summary['accuracy']:.4f} "
              f"[{summary['seconds']:.0f}s]", flush=True)


def _schedule(placements):
    first = placements[0]
    return RecurrenceSchedule(tuple(first['temporal_write_mask']), tuple(first['depth_write_mask']))


if __name__ == '__main__':
    main()
