"""Where does the move decision form? Next-move readouts at every site.

At the two pre-move kinds (``dot``, ``space_black``) with a complete next move:

``from`` / ``to``  linear probes for the from- and to-square of the move actually played
``lens``           logit lens: the model's own final norm and unembedding applied to the
                   site, scored on the move's first character (top-1 accuracy)

Probes are fit separately for White's (``dot``) and Black's (``space_black``)
decisions, since the model may represent moves relative to the mover.

The played move is a human's, so the ceiling is how often the model predicts it
(about 0.6 top-1 for the move). What matters is where each readout rises
relative to where the board probe plateaus. Uses the activations from
``capture.py``; writes ``results/decision/<arm>.json``.

Example:
    python -m experiments.interp.board_state.decision --arms transformer temporal depth hybrid karvonen
"""

import argparse
import json

import chess
import numpy as np
import torch

from interp.boards import KINDS
from interp.probes import fit
from .common import ARMS, RESULTS, device, load_arm, load_rows

# Per-colour subsets are about half the points; 36 epochs gives roughly 2,500 steps per probe.
TRAINING = dict(epochs=36, batch=1024, lr=3e-3)


def move_labels(labels, meta):
    """Points with a parseable next move: indices, from-square, to-square, first-character id."""
    pre_move = np.isin(labels['kind'], [KINDS.index('dot'), KINDS.index('space_black')]) & (labels['next_san'] != '')
    points, origin, target, first, kinds = [], [], [], [], []
    for point in np.flatnonzero(pre_move):
        san = str(labels['next_san'][point])
        try:
            move = chess.Board(str(labels['fen'][point])).parse_san(san)
        except ValueError:
            continue
        points.append(point)
        origin.append(move.from_square)
        target.append(move.to_square)
        first.append(meta['stoi'][san[0]])
        kinds.append(labels['kind'][point])
    return np.array(points), np.array(origin), np.array(target), np.array(first), np.array(kinds)


@torch.no_grad()
def lens_accuracy(model, x, first, device_name, batch=8192):
    correct = 0
    for start in range(0, len(x), batch):
        h = torch.from_numpy(np.asarray(x[start:start + batch], dtype=np.float32)).to(device_name)
        predicted = model.lm_head(model.transformer.ln_f(h)).argmax(-1).cpu().numpy()
        correct += int((predicted == first[start:start + batch]).sum())
    return correct / len(x)


def analyse(name, labels, meta, device_name):
    runner, _ = load_arm(name, device_name)
    run = json.loads((RESULTS / 'activations' / name / 'run.json').read_text())
    points, origin, target, first, kinds = move_labels(labels, meta)
    train = labels['train'][points]
    result = dict(arm=name, points=len(points), sites=[])
    for site in run['sites']:
        x = np.load(RESULTS / 'activations' / name / f'{site["name"]}.npy', mmap_mode='r')[points]
        entry = dict(site)
        for key, y in (('from', origin), ('to', target)):
            correct = []
            for kind in ('dot', 'space_black'):
                rows = kinds == KINDS.index(kind)
                probe = fit(np.asarray(x[rows & train], dtype=np.float32), y[rows & train, None], 64,
                            device=device_name, **TRAINING)
                predicted = probe.predict(torch.from_numpy(np.asarray(x[rows & ~train], dtype=np.float32)))
                correct.append(predicted[:, 0].numpy() == y[rows & ~train])
                entry[f'{key}_accuracy_{kind}'] = float(correct[-1].mean())
            entry[f'{key}_accuracy'] = float(np.concatenate(correct).mean())
        entry['lens_first_char_accuracy'] = lens_accuracy(runner.model, x[~train], first[~train], device_name)
        result['sites'].append(entry)
        print(f'{name} {site["name"]:>7}: from {entry["from_accuracy"]:.3f}, to {entry["to_accuracy"]:.3f}, '
              f'lens first char {entry["lens_first_char_accuracy"]:.3f}', flush=True)
    out = RESULTS / 'decision'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=['transformer', 'temporal', 'depth', 'hybrid', 'karvonen'],
                        choices=list(ARMS))
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    _, _, meta = load_rows()
    labels = dict(np.load(RESULTS / 'labels.npz'))
    for name in args.arms:
        analyse(name, labels, meta, args.device)


if __name__ == '__main__':
    main()
