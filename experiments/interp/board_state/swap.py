"""Memory swap: is the early board carried by the temporal memory?

At a white-to-move ``dot`` of game A, the temporal mixer reads the memory of
the previous character. For a window of the last ``k`` characters of A's
prefix, we replace the memory each one reads with the memory game B's run wrote
at the same offset before its own ``dot`` at the same ply, on every settling
pass. We then read the ``board_karvonen`` probe at every site of A's dot. On
squares where A and B differ, we count how often the probe reports A's piece
and how often B's.

With ``k = 1`` only the dot's own memory is swapped; every earlier character
still carries A's board in its memory, and attention can read it. Wider windows
push A's memories further back. If early sites report B, the board there comes
from memory. If later sites return to A even for wide windows, the model
re-derives the board from the text.

The next-character distribution at the dot is also scored: probability on
characters that begin a legal move only in B, and only in A.

Controls: no swap, and a self-swap (B = A), which must reproduce no swap.
Writes ``results/swap/<arm>.json``.

Example:
    python -m experiments.interp.board_state.swap --arm temporal
"""

import argparse
import json
import time

import chess
import numpy as np
import torch

from interp.boards import KINDS
from interp.probes import Probe
from interp.steering import pad, valid_positions
from .common import RESULTS, device, load_arm, load_rows
from .probe import KARVONEN_CHARACTERS


def pairs(labels, count, seed, min_index):
    """(A, B) dot points of different held-out rows at the same ply."""
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(~labels['train'] & (labels['kind'] == KINDS.index('dot')) & (labels['game'] == 0) &
                          (labels['index'] >= min_index) & (labels['index'] < KARVONEN_CHARACTERS))
    by_ply = {}
    for point in pool:
        by_ply.setdefault(int(labels['ply'][point]), []).append(point)
    chosen = []
    for point in rng.permutation(pool):
        others = [p for p in by_ply[int(labels['ply'][point])] if labels['row'][p] != labels['row'][point]]
        if others:
            chosen.append((point, others[rng.integers(len(others))]))
        if len(chosen) == count:
            break
    return np.array(chosen)


@torch.no_grad()
def run_points(runner, sequences, device_name, capture, edits=None):
    idx, lengths = pad(sequences, device_name)
    rows = torch.arange(len(sequences), device=device_name)
    run = runner.run(idx, capture=capture, index=(rows, lengths - 1), edits=edits,
                     valid=valid_positions(lengths, idx.shape[1]))
    probabilities = torch.softmax(run.logits[rows, lengths - 1].float(), -1).cpu()
    return run.captures, probabilities


def window(sequences, width, offset, device_name):
    """(rows, positions) of the ``width`` characters ending ``offset`` before each sequence's end."""
    rows = torch.arange(len(sequences)).repeat_interleave(width)
    ends = torch.tensor([len(s) - 1 - offset for s in sequences])
    positions = (ends[:, None] - torch.arange(width - 1, -1, -1)[None, :]).reshape(-1)
    return rows.to(device_name), positions.to(device_name)


@torch.no_grad()
def capture_at(runner, sequences, device_name, site, index):
    idx, lengths = pad(sequences, device_name)
    run = runner.run(idx, capture=[site], index=index, valid=valid_positions(lengths, idx.shape[1]))
    return run.captures[site].float().to(device_name)


def swap_edit(model, prelude, memory, index):
    """Replace the mixer output at ``index`` with the mix of A's ``prelude`` and donor ``memory``."""
    mixed = model.temporal_mixer._mix(prelude[:, None], memory[:, None])[:, 0]

    def edit(h):
        h = h.clone()
        h[index] = mixed.to(h.dtype)
        return h
    return edit


def first_characters(board):
    return {board.san(move)[0] for move in board.legal_moves}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', required=True, choices=['temporal', 'hybrid'])
    parser.add_argument('--pairs', type=int, default=300)
    parser.add_argument('--windows', nargs='+', type=int, default=[1, 4, 16, 64])
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    started = time.monotonic()
    runner, checkpoint = load_arm(args.arm, args.device)
    rows, _, meta = load_rows(checkpoint)
    labels = dict(np.load(RESULTS / 'labels.npz'))
    chosen = pairs(labels, args.pairs, args.seed, max(args.windows) + 1)
    sites = runner.site_names
    probes = {site: Probe.from_state(torch.load(RESULTS / 'probes' / args.arm / f'{site}.pt')['board_karvonen'],
                                     args.device) for site in sites}
    stoi = meta['stoi']

    def sequence(point):
        return rows[labels['row'][point], :labels['index'][point] + 1]

    conditions = ['none'] + [f'{kind}{k}' for k in args.windows for kind in ('self', 'swap')]
    totals = {condition: {site: np.zeros(3) for site in sites} for condition in conditions}
    next_char = {condition: np.zeros(3) for condition in totals}
    for start in range(0, len(chosen), args.batch):
        batch = chosen[start:start + args.batch]
        a_seqs, b_seqs = [sequence(a) for a, _ in batch], [sequence(b) for _, b in batch]
        board_a = torch.as_tensor(labels['board_rel'][batch[:, 0]].astype(np.int64))
        board_b = torch.as_tensor(labels['board_rel'][batch[:, 1]].astype(np.int64))
        differ = board_a != board_b
        edits = {'none': None}
        for k in args.windows:
            # The mixer at A's last k characters reads memory written one character earlier.
            target = window(a_seqs, k, 0, args.device)
            prelude = capture_at(runner, a_seqs, args.device, 'L1', target)
            for kind, donor_seqs in (('self', a_seqs), ('swap', b_seqs)):
                memory = capture_at(runner, donor_seqs, args.device, 'L7', window(donor_seqs, k, 1, args.device))
                edits[f'{kind}{k}'] = {'Tmix': swap_edit(runner.model, prelude, memory, target)}
        a_only, b_only = [], []
        for a, b in batch:
            first_a = first_characters(chess.Board(str(labels['fen'][a])))
            first_b = first_characters(chess.Board(str(labels['fen'][b])))
            a_only.append([stoi[c] for c in first_a - first_b])
            b_only.append([stoi[c] for c in first_b - first_a])
        for condition in totals:
            captures, probabilities = run_points(runner, a_seqs, args.device, sites, edits[condition])
            for site in sites:
                predicted = probes[site].predict(captures[site].float())
                totals[condition][site] += [int(((predicted == board_a) & differ).sum()),
                                            int(((predicted == board_b) & differ).sum()), int(differ.sum())]
            for row in range(len(batch)):
                next_char[condition] += [float(probabilities[row, a_only[row]].sum()),
                                         float(probabilities[row, b_only[row]].sum()), 1]
        print(f'{args.arm}: pairs {start + len(batch)}/{len(chosen)}, {time.monotonic() - started:.0f}s', flush=True)

    result = dict(arm=args.arm, pairs=len(chosen), windows=args.windows, probe='board_karvonen', conditions={})
    for condition in totals:
        per_site = {site: dict(reads_a=float(t[0] / t[2]), reads_b=float(t[1] / t[2]))
                    for site, t in totals[condition].items()}
        n = next_char[condition]
        result['conditions'][condition] = dict(sites=per_site, p_first_char_only_a=float(n[0] / n[2]),
                                               p_first_char_only_b=float(n[1] / n[2]))
        print(f'{condition}: next char P(only-A first chars) {n[0] / n[2]:.3f}, '
              f'P(only-B first chars) {n[1] / n[2]:.3f}')
        for site in sites:
            print(f'  {site:>7}: reads A {per_site[site]["reads_a"]:.3f}, reads B {per_site[site]["reads_b"]:.3f}')
    result['seconds'] = time.monotonic() - started
    out = RESULTS / 'swap'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.arm}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
