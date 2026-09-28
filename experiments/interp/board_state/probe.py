"""Train linear probes at every captured site and score them on held-out rows.

Three probes per site:

``board``      64 squares x 13 classes, side to move first ("mine"/"theirs"), trained on
               both pre-move kinds (``dot`` for white, ``space_black`` for black), so one
               probe must read both colours' boards
``board_karvonen``  Karvonen's setting: ``dot`` points of each row's first game within
               its first 365 characters (his probe games are cut to 365 characters)
``turn``       side to move on the two space kinds (the same input token), class-balanced

Karvonen's 8-layer model reaches 0.980 with ``board_karvonen`` at L6 (0.991 in
his paper with about ten times more probe games); a random-init model 0.754.

Board accuracy is reported over all squares (Karvonen's metric) and over
*changed* squares, whose class differs from the starting position in the same
encoding. Most squares never change, so the changed-square accuracy is the more
informative number. The per-square majority class of the training rows is the
trivial baseline.

Writes ``results/probes/<arm>/<site>.pt`` and ``results/probes/<arm>/metrics.json``.

Example:
    python -m experiments.interp.board_state.probe --arms transformer temporal depth hybrid random_init
"""

import argparse
import json

import chess
import numpy as np
import torch

from interp.boards import KINDS, encode
from interp.probes import fit
from .common import ARMS, RESULTS, device

INITIAL = encode(chess.Board(), False)
_SWAP = np.array([0] + list(range(7, 13)) + list(range(1, 7)), dtype=np.int8)
DOT, SPACE_WHITE, SPACE_BLACK = (KINDS.index(kind) for kind in KINDS)
PROBES = ('board', 'board_karvonen', 'turn')
# The Karvonen subset is about a sixth of the board points, so it takes more, smaller steps.
TRAINING = dict(board=dict(epochs=12, batch=1024, lr=3e-3), board_karvonen=dict(epochs=30, batch=256, lr=3e-3),
                turn=dict(epochs=12, batch=1024, lr=3e-3))
KARVONEN_CHARACTERS = 365


def initial_boards(labels):
    """Starting-position classes per point, side to move first."""
    initial = np.broadcast_to(INITIAL, labels['board_rel'].shape)
    return np.where(labels['turn'][:, None] == 1, initial, _SWAP[initial])


def selection(labels, probe):
    """Points, integer targets [n, heads] and class count of one probe."""
    kind = labels['kind']
    if probe == 'turn':
        rows = np.isin(kind, [SPACE_WHITE, SPACE_BLACK])
        # Balance the two classes inside each split with a fixed subsample.
        rng = np.random.default_rng(0)
        for split in (labels['train'], ~labels['train']):
            white, black = (np.flatnonzero(rows & split & (kind == k)) for k in (SPACE_WHITE, SPACE_BLACK))
            larger, smaller = (white, black) if len(white) > len(black) else (black, white)
            rows[rng.choice(larger, len(larger) - len(smaller), replace=False)] = False
        return rows, labels['turn'][:, None].astype(np.int64), 2
    if probe == 'board_karvonen':
        rows = (kind == DOT) & (labels['game'] == 0) & (labels['index'] < KARVONEN_CHARACTERS)
    else:
        rows = np.isin(kind, [DOT, SPACE_BLACK])
    return rows, labels['board_rel'].astype(np.int64), 13


def board_scores(predicted, truth, initial, kinds):
    changed = truth != initial
    correct = predicted == truth
    scores = dict(accuracy=float(correct.mean()), changed_accuracy=float(correct[changed].mean()))
    for code in np.unique(kinds):
        rows = kinds == code
        scores[f'accuracy_{KINDS[code]}'] = float(correct[rows].mean())
        scores[f'changed_accuracy_{KINDS[code]}'] = float(correct[rows][changed[rows]].mean())
    return scores


def majority_baseline(labels, probe):
    rows, y, _ = selection(labels, probe)
    train, test = rows & labels['train'], rows & ~labels['train']
    majority = np.array([np.bincount(y[train][:, square], minlength=13).argmax() for square in range(64)])
    return board_scores(np.broadcast_to(majority, y[test].shape), y[test], initial_boards(labels)[test],
                        labels['kind'][test])


def probe_arm(name, labels, device_name):
    run = json.loads((RESULTS / 'activations' / name / 'run.json').read_text())
    out = RESULTS / 'probes' / name
    out.mkdir(parents=True, exist_ok=True)
    initial = initial_boards(labels)
    metrics = dict(arm=name, training=TRAINING, sites=[])
    for site in run['sites']:
        x = np.load(RESULTS / 'activations' / name / f'{site["name"]}.npy', mmap_mode='r')
        states, entry = {}, dict(site)
        for probe_name in PROBES:
            rows, y, classes = selection(labels, probe_name)
            train, test = rows & labels['train'], rows & ~labels['train']
            probe = fit(np.asarray(x[train], dtype=np.float32), y[train], classes, device=device_name,
                        **TRAINING[probe_name])
            predicted = probe.predict(torch.from_numpy(np.asarray(x[test], dtype=np.float32))).numpy()
            if probe_name == 'turn':
                entry['turn_accuracy'] = float((predicted == y[test]).mean())
            else:
                scores = board_scores(predicted, y[test], initial[test], labels['kind'][test])
                entry.update({f'{probe_name}_{key}': value for key, value in scores.items()})
            states[probe_name] = probe.state()
        torch.save(states, out / f'{site["name"]}.pt')
        metrics['sites'].append(entry)
        print(f'{name} {site["name"]:>7}: board {entry["board_accuracy"]:.4f} '
              f'(changed {entry["board_changed_accuracy"]:.3f}), karvonen {entry["board_karvonen_accuracy"]:.4f}, '
              f'turn {entry["turn_accuracy"]:.3f}', flush=True)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    labels = dict(np.load(RESULTS / 'labels.npz'))
    baselines = {probe: majority_baseline(labels, probe) for probe in ('board', 'board_karvonen')}
    (RESULTS / 'probes').mkdir(parents=True, exist_ok=True)
    (RESULTS / 'probes' / 'baselines.json').write_text(json.dumps(baselines, indent=2) + '\n')
    print('majority baseline: ' + ', '.join(f'{k} {v["accuracy"]:.4f}' for k, v in baselines.items()))
    for name in args.arms:
        probe_arm(name, labels, args.device)


if __name__ == '__main__':
    main()
