"""When is the most recent move applied to the board?

At each pre-move point (``dot``, ``space_black``) whose previous pre-move point
in the same game is also labelled, two probes are fit per site in the absolute
encoding: one for the board *after* the most recent move (the current board)
and one for the board *before* it. Both are scored only on the squares that
move changed, where the two boards disagree.

A site that holds a stale board reads "before" better than "after"; the layer
where "after" overtakes "before" is where the move is applied. Probes are fit
separately for each kind: the board is represented relative to the side to
move, so one absolute-encoding probe cannot read both colours. Uses the
activations from ``capture.py``; writes ``results/update/<arm>.json``.

Example:
    python -m experiments.interp.board_state.update --arms transformer temporal hybrid
"""

import argparse
import json

import numpy as np
import torch

from interp.boards import KINDS
from interp.probes import fit
from .common import ARMS, RESULTS, device

# Per-colour subsets are about half the points; 36 epochs gives roughly 2,500 steps per probe.
TRAINING = dict(epochs=36, batch=1024, lr=3e-3)


def paired_points(labels):
    """Pre-move points and the previous pre-move point of the same game (ply - 1)."""
    pre_move = np.isin(labels['kind'], [KINDS.index('dot'), KINDS.index('space_black')])
    key = {(int(r), int(g), int(p)): i for i, (r, g, p) in
           enumerate(zip(labels['row'], labels['game'], labels['ply'])) if pre_move[i]}
    current, previous = [], []
    for (row, game, ply), point in key.items():
        before = key.get((row, game, ply - 1))
        if before is not None:
            current.append(point)
            previous.append(before)
    return np.array(current), np.array(previous)


def analyse(name, labels, device_name):
    run = json.loads((RESULTS / 'activations' / name / 'run.json').read_text())
    current, previous = paired_points(labels)
    after = labels['board_abs'][current].astype(np.int64)
    before = labels['board_abs'][previous].astype(np.int64)
    changed = after != before
    train = labels['train'][current]
    kinds = labels['kind'][current]
    result = dict(arm=name, points=len(current), sites=[])
    for site in run['sites']:
        x = np.load(RESULTS / 'activations' / name / f'{site["name"]}.npy', mmap_mode='r')[current]
        entry = dict(site)
        for kind in ('dot', 'space_black'):
            rows = kinds == KINDS.index(kind)
            x_train = np.asarray(x[rows & train], dtype=np.float32)
            x_test = torch.from_numpy(np.asarray(x[rows & ~train], dtype=np.float32))
            for key, y in (('after', after), ('before', before)):
                probe = fit(x_train, y[rows & train], 13, device=device_name, **TRAINING)
                predicted = probe.predict(x_test).numpy()
                entry[f'{kind}_{key}_changed_accuracy'] = float(
                    (predicted == y[rows & ~train])[changed[rows & ~train]].mean())
        result['sites'].append(entry)
        print(f'{name} {site["name"]:>7}: changed squares after / before: '
              + ', '.join(f'{kind} {entry[f"{kind}_after_changed_accuracy"]:.3f} / '
                          f'{entry[f"{kind}_before_changed_accuracy"]:.3f}' for kind in ('dot', 'space_black')),
              flush=True)
    out = RESULTS / 'update'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=['transformer', 'temporal', 'hybrid'], choices=list(ARMS))
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    labels = dict(np.load(RESULTS / 'labels.npz'))
    for name in args.arms:
        analyse(name, labels, args.device)


if __name__ == '__main__':
    main()
