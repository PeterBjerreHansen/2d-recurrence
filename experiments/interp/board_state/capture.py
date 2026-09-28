"""Label probe points and capture every site's residual stream at them.

Writes ``results/labels.npz`` once and ``results/activations/<arm>/<site>.npy``
(float16 [points, width]) per arm, plus ``<arm>/run.json`` with the fixed-point
NLL over all targets of the rows and the settling statistics.

Example:
    python -m experiments.interp.board_state.capture --arms transformer temporal depth hybrid random_init
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from interp.boards import KINDS, encode, probe_points
from .common import (ARMS, CONTEXT, RESULTS, SPACE_WHITE_KEEP, TRAIN_ROWS, decode, device, load_arm,
                     load_rows)


def build_labels(rows, selection, meta):
    columns = {key: [] for key in ('row', 'index', 'kind', 'ply', 'game', 'turn', 'board_abs', 'board_rel')}
    fens, next_sans = [], []
    kept = 0
    for r, row in enumerate(rows):
        for point in probe_points(decode(row, meta), limit=CONTEXT):
            if point.kind == 'space_white':
                kept += 1
                if kept % SPACE_WHITE_KEEP:
                    continue
            columns['row'].append(r)
            columns['index'].append(point.index)
            columns['kind'].append(KINDS.index(point.kind))
            columns['ply'].append(point.ply)
            columns['game'].append(point.game)
            columns['turn'].append(int(point.board.turn))
            columns['board_abs'].append(encode(point.board, False))
            columns['board_rel'].append(encode(point.board, True))
            fens.append(point.board.fen())
            next_sans.append(point.next_san or '')
    labels = {key: np.array(value) for key, value in columns.items()}
    labels.update(fen=np.array(fens), next_san=np.array(next_sans), selection=selection,
                  train=labels['row'] < TRAIN_ROWS)
    return labels


@torch.no_grad()
def capture(name, rows, labels, device_name, batch_rows):
    runner, _ = load_arm(name, device_name)
    out = RESULTS / 'activations' / name
    out.mkdir(parents=True, exist_ok=True)
    width = runner.model.config.n_embd
    total = len(labels['row'])
    arrays = {site: np.lib.format.open_memmap(out / f'{site}.npy', mode='w+', dtype=np.float16,
                                              shape=(total, width)) for site in runner.site_names}
    nll_sum, targets, passes, changes = 0.0, 0, [], []
    started = time.monotonic()
    for start in range(0, len(rows), batch_rows):
        block = torch.from_numpy(rows[start:start + batch_rows, :CONTEXT + 1].astype(np.int64)).to(device_name)
        x, y = block[:, :-1], block[:, 1:]
        selected = np.flatnonzero((labels['row'] >= start) & (labels['row'] < start + len(block)))
        index = (torch.from_numpy(labels['row'][selected] - start).to(device_name),
                 torch.from_numpy(labels['index'][selected]).to(device_name))
        run = runner.run(x, capture=runner.site_names, index=index)
        for site, values in run.captures.items():
            arrays[site][selected] = values.numpy()
        nll_sum += float(F.cross_entropy(run.logits.transpose(1, 2), y, reduction='sum'))
        targets += y.numel()
        passes.append(run.passes)
        changes.append(run.memory_change)
        print(f'{name}: rows {start + len(block)}/{len(rows)}, passes {run.passes}, '
              f'{time.monotonic() - started:.0f}s', flush=True)
    for array in arrays.values():
        array.flush()
    report = dict(arm=name, checkpoint=str(ARMS[name].checkpoint), depth_steps=runner.depth_steps,
                  sites=[vars(site) for site in runner.sites], nll=nll_sum / targets, targets=targets,
                  passes=passes, max_memory_change=max((c for c in changes if c is not None), default=None),
                  tolerance=runner.tolerance, seconds=time.monotonic() - started)
    (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{name}: fixed-point NLL {report["nll"]:.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    parser.add_argument('--batch-rows', type=int, default=4)
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    rows, selection, meta = load_rows()
    RESULTS.mkdir(parents=True, exist_ok=True)
    labels_path = RESULTS / 'labels.npz'
    if labels_path.exists():
        labels = dict(np.load(labels_path))
        if not np.array_equal(labels['selection'], selection):
            raise SystemExit('labels.npz was built from different rows')
    else:
        labels = build_labels(rows, selection, meta)
        np.savez(labels_path, **labels)
    print(f'{len(labels["row"])} points, {int(labels["train"].sum())} for training')
    for name in args.arms:
        capture(name, rows, labels, args.device, args.batch_rows)


if __name__ == '__main__':
    main()
