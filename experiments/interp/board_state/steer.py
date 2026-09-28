"""Causal checks of the probes: edit the residual stream along probe directions.

``turn``   At a space, push the side-to-move probe to the other player. A space
           after black's move is followed by a move number (a digit); a space
           after white's move by black's SAN (a letter). A working edit swaps
           which kind of character the model predicts.

``board``  Karvonen's board intervention. Decode the model's move greedily,
           delete the moved piece from the probe's board ("mine" piece to empty on
           its square), decode again. A working edit makes the new move legal on
           the modified board although the PGN text is unchanged.

Both compare the probe direction with a random direction of the same norm at
each site. Examples come from held-out rows. Writes ``results/steer/<test>-<arm>.json``
(``--tag`` adds a suffix for supporting runs).

Examples:
    python -m experiments.interp.board_state.steer turn --arm transformer
    python -m experiments.interp.board_state.steer board --arm hybrid --examples 100
"""

import argparse
import json
import time

import chess
import numpy as np
import torch

from interp.boards import EMPTY, KINDS
from interp.probes import Probe
from interp.steering import (additive_edit, greedy_moves, next_char_probabilities, pad, probe_coefficients,
                             random_like, valid_positions)
from .common import RESULTS, device, load_arm, load_rows

MARGIN = 5.0          # target (target - source) probe logit margin at the edited point
MAX_INDEX = 400       # probed character index cap, to bound sequence length
DIGITS = '0123456789'


def default_groups(runner):
    """Single sites without the embedding, plus each core layer at every iteration of a looped core."""
    names = [site.name for site in runner.sites if site.kind != 'embed']
    if runner.depth_steps == 1:
        return [[name] for name in names]
    looped = [[name] for name in names if '@' not in name or name.endswith(('@1', f'@{runner.depth_steps}'))]
    layers = sorted({name.split('@')[0] for name in names if '@' in name and name.startswith('L')})
    return looped + [[f'{layer}@{i}' for i in range(1, runner.depth_steps + 1)] for layer in layers]


def load_probes(arm, sites, probe_name, device_name):
    return {site: Probe.from_state(torch.load(RESULTS / 'probes' / arm / f'{site}.pt')[probe_name], device_name)
            for site in sites}


@torch.no_grad()
def point_activations(runner, sequences, sites, device_name):
    idx, lengths = pad(sequences, device_name)
    rows = torch.arange(len(sequences), device=device_name)
    run = runner.run(idx, capture=sites, index=(rows, lengths - 1), valid=valid_positions(lengths, idx.shape[1]))
    return {site: value.float().to(device_name) for site, value in run.captures.items()}


def edit_factory(group_vectors, starts):
    """Edits for a batch subset: ``group_vectors`` maps site -> [n, width]."""
    def edits(rows):
        rows_t = torch.as_tensor(rows, device=starts.device)
        return {site: additive_edit(vectors[rows_t], starts[rows_t]) for site, vectors in group_vectors.items()}
    return edits


def chunks(n, size):
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


def turn_test(args, runner, labels, rows, meta):
    kinds = labels['kind']
    rng = np.random.default_rng(args.seed)
    test = ~labels['train'] & (labels['index'] <= MAX_INDEX)
    picked = np.concatenate([rng.choice(np.flatnonzero(test & (kinds == KINDS.index(kind))), args.examples // 2,
                                        replace=False) for kind in ('space_white', 'space_black')])
    sequences = [rows[labels['row'][p], :labels['index'][p] + 1] for p in picked]
    white = torch.tensor(labels['turn'][picked] == 1)
    digit_ids = [meta['stoi'][c] for c in DIGITS]
    groups = args.groups or default_groups(runner)
    sites = sorted({site for group in groups for site in group})
    probes = load_probes(args.arm, sites, 'turn', args.device)
    generator = torch.Generator().manual_seed(args.seed)
    results = dict(test='turn', arm=args.arm, examples=len(picked), margin=MARGIN, conditions=[])

    def evaluate(label, group, multiplier, direction):
        digit = []
        for batch in chunks(len(picked), args.batch):
            seqs = [sequences[i] for i in batch]
            if group is None:
                edits = None
            else:
                acts = point_activations(runner, seqs, group, args.device)
                current = white[batch].long().to(args.device)
                vectors = {}
                for site in group:
                    zero = torch.zeros_like(current)
                    coefficients, directions = probe_coefficients(probes[site], acts[site], zero, current,
                                                                  1 - current, MARGIN)
                    if direction == 'random':
                        directions = random_like(directions, generator)
                    vectors[site] = multiplier * coefficients[:, None] * directions
                starts = torch.tensor([len(s) - 1 for s in seqs], device=args.device)
                edits = edit_factory(vectors, starts)
            probs = next_char_probabilities(runner, seqs, args.device, edits)
            digit.append(probs[:, digit_ids].sum(-1))
        digit = torch.cat(digit)
        predicted_digit = digit > 0.5
        entry = dict(label=label, group=group, multiplier=multiplier, direction=direction,
                     p_digit_white=float(digit[white].mean()), p_digit_black=float(digit[~white].mean()),
                     flipped=float((predicted_digit != white).float().mean()))
        results['conditions'].append(entry)
        print(f'{label:>28}: P(digit) white {entry["p_digit_white"]:.3f} black {entry["p_digit_black"]:.3f} '
              f'flipped {entry["flipped"]:.3f}', flush=True)

    evaluate('none', None, 0, None)
    for group in groups:
        for multiplier in args.multipliers:
            for direction in ('probe', 'random'):
                evaluate(f'{"+".join(group)} x{multiplier} {direction}', group, multiplier, direction)
    return results


def board_examples(args, runner, labels, rows, meta):
    """Points where the model's greedy move is legal and deleting its piece makes that move illegal."""
    kinds = labels['kind']
    rng = np.random.default_rng(args.seed)
    pool = np.flatnonzero(~labels['train'] & np.isin(kinds, [KINDS.index('dot'), KINDS.index('space_black')]) &
                          (labels['game'] == 0) & (labels['ply'] >= 10) & (labels['ply'] <= 60) &
                          (labels['index'] <= MAX_INDEX))
    candidates = rng.choice(pool, min(len(pool), 3 * args.examples), replace=False)
    sequences = [rows[labels['row'][p], :labels['index'][p] + 1] for p in candidates]
    moves = []
    for batch in chunks(len(candidates), args.batch):
        moves += greedy_moves(runner, [sequences[i] for i in batch], meta['itos'], args.device)
    examples = []
    for point, sequence, san in zip(candidates, sequences, moves):
        board = chess.Board(str(labels['fen'][point]))
        try:
            move = board.parse_san(san)
        except ValueError:
            continue
        piece = board.piece_at(move.from_square)
        if piece.piece_type == chess.KING:
            continue
        modified = board.copy(stack=False)
        modified.remove_piece_at(move.from_square)
        modified.castling_rights = modified.clean_castling_rights()
        if not modified.is_valid() or not any(modified.legal_moves) or _legal(modified, san):
            continue
        examples.append(dict(point=int(point), sequence=sequence, san=san, square=move.from_square,
                             piece_class=piece.piece_type, board=board, modified=modified))
        if len(examples) == args.examples:
            break
    return examples


def _legal(board, san):
    try:
        board.parse_san(san)
    except ValueError:
        return False
    return True


def board_test(args, runner, labels, rows, meta):
    examples = board_examples(args, runner, labels, rows, meta)
    print(f'{len(examples)} examples', flush=True)
    groups = args.groups or default_groups(runner)
    sites = sorted({site for group in groups for site in group})
    probes = load_probes(args.arm, sites, 'board', args.device)
    generator = torch.Generator().manual_seed(args.seed)
    squares = torch.tensor([e['square'] for e in examples], device=args.device)
    pieces = torch.tensor([e['piece_class'] for e in examples], device=args.device)  # "mine" classes are 1-6
    results = dict(test='board', arm=args.arm, examples=len(examples), margin=MARGIN,
                   example_moves=[dict(point=e['point'], san=e['san'], square=chess.square_name(e['square']))
                                  for e in examples], conditions=[])

    def evaluate(label, group, multiplier, direction):
        written = []
        for batch in chunks(len(examples), args.batch):
            seqs = [examples[i]['sequence'] for i in batch]
            edits = None
            if group is not None:
                acts = point_activations(runner, seqs, group, args.device)
                vectors = {}
                for site in group:
                    b = torch.as_tensor(batch, device=args.device)
                    coefficients, directions = probe_coefficients(probes[site], acts[site], squares[b], pieces[b],
                                                                  torch.full_like(b, EMPTY), MARGIN)
                    if direction == 'random':
                        directions = random_like(directions, generator)
                    vectors[site] = multiplier * coefficients[:, None] * directions
                starts = torch.tensor([len(s) - 1 for s in seqs], device=args.device)
                edits = edit_factory(vectors, starts)
            written += greedy_moves(runner, seqs, meta['itos'], args.device, edits)
        legal_modified = np.mean([_legal(e['modified'], san) for e, san in zip(examples, written)])
        legal_original = np.mean([_legal(e['board'], san) for e, san in zip(examples, written)])
        moved_deleted = np.mean([_legal(e['board'], san) and e['board'].parse_san(san).from_square == e['square']
                                 for e, san in zip(examples, written)])
        entry = dict(label=label, group=group, multiplier=multiplier, direction=direction,
                     legal_modified=float(legal_modified), legal_original=float(legal_original),
                     moves_deleted_piece=float(moved_deleted), moves=written)
        results['conditions'].append(entry)
        print(f'{label:>40}: legal on modified {legal_modified:.3f}, on original {legal_original:.3f}, '
              f'moves deleted piece {moved_deleted:.3f}', flush=True)

    evaluate('none', None, 0, None)
    for group in groups:
        for multiplier in args.multipliers:
            for direction in ('probe', 'random'):
                evaluate(f'{"+".join(group)} x{multiplier} {direction}', group, multiplier, direction)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('test', choices=['turn', 'board'])
    parser.add_argument('--arm', required=True)
    parser.add_argument('--groups', nargs='+', type=lambda s: s.split('+'),
                        help='site groups edited together, e.g. L5 L4+L5+L6+L7')
    parser.add_argument('--multipliers', nargs='+', type=float, default=[1.0, 2.0])
    parser.add_argument('--examples', type=int, default=200)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default='', help='suffix for the output file, for supporting runs')
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    runner, checkpoint = load_arm(args.arm, args.device)
    rows, _, meta = load_rows(checkpoint)
    labels = dict(np.load(RESULTS / 'labels.npz'))
    started = time.monotonic()
    results = (turn_test if args.test == 'turn' else board_test)(args, runner, labels, rows, meta)
    results['seconds'] = time.monotonic() - started
    out = RESULTS / 'steer'
    out.mkdir(parents=True, exist_ok=True)
    suffix = f'-{args.tag}' if args.tag else ''
    (out / f'{args.test}-{args.arm}{suffix}.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
