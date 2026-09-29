"""Forcing lines in real games: data, validation and a behavioural baseline.

``build``      Scan decision points (the ``.`` before White's move and the space
               before Black's) in held-out validation rows that the probes never
               train on, and classify each position:

               ``mate1``     the side to move can mate in one
               ``mate2``     no mate in one, but a checking move after which every legal
                             reply allows mate in one (a forced three-ply line)
               ``control``   neither, with at least one checking move available

               Only the first ``mate1`` and ``mate2`` position per game is kept, so
               that one missed mate recurring over several plies counts once; matched
               controls also use each game at most once.

               Every checking move is stored with its number of legal replies, so a
               preference for forcing checks can be separated from a preference for
               checks that leave few replies (visible one ply ahead). Each stored
               line is re-verified independently. Controls are matched to the
               ``mate2`` positions on side to move, ply and number of checking moves.
               Mates that start with a quiet move are not searched.

``behaviour``  For each model: the teacher-forced probability of every checking
               move, and the greedy move. Within a position, the share of
               probability on the forcing (or mating) checks against the uniform
               share over checks shows whether the model sees the line.

Writes ``results/forcing/positions.jsonl``, ``results/forcing/build.json`` and
``results/forcing/behaviour-<arm>.json``.

Examples:
    python -m experiments.interp.board_state.forcing build --rows 20000
    python -m experiments.interp.board_state.forcing behaviour --arms transformer temporal depth hybrid karvonen
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random

import chess
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import ChessData
from interp.boards import probe_points
from interp.steering import greedy_moves, pad, valid_positions
from .common import CONTEXT, RESULTS, ROW_SEED, ROWS, decode, device, load_arm

OUT = RESULTS / 'forcing'
MOVE_ROOM = 8  # characters after the decision point that a written move can need
PLY_BUCKET = 10
CONTROL_POOL = 200  # control candidates kept per stratum, by reservoir sampling


def _mates(board, move):
    board.push(move)
    mate = board.is_checkmate()
    board.pop()
    return mate


def mating_moves(board):
    """Moves that give checkmate (only checking moves can)."""
    return [m for m in board.legal_moves if board.gives_check(m) and _mates(board, m)]


def checks_with_replies(board):
    """Every checking move and its number of legal replies."""
    out = []
    for move in board.legal_moves:
        if board.gives_check(move):
            board.push(move)
            out.append((move, board.legal_moves.count()))
            board.pop()
    return out


def forcing_mates_in_two(board):
    """Checking first moves, not themselves mate, after which every reply allows mate in one.

    Returns ``[(first, {reply: [mating moves]})]``.
    """
    lines = []
    for move in list(board.legal_moves):
        if not board.gives_check(move):
            continue
        board.push(move)
        if not board.is_checkmate():
            answers = {}
            for reply in list(board.legal_moves):
                board.push(reply)
                mates = mating_moves(board)
                board.pop()
                if not mates:
                    answers = None
                    break
                answers[reply] = mates
            if answers:
                lines.append((move, answers))
        board.pop()
    return lines


def verify(record):
    """Independent re-check of a stored position's label; raises on any mismatch."""
    board = chess.Board(record['fen'])
    checks = {board.san(m) for m in board.legal_moves if board.gives_check(m)}
    assert checks == set(record['checks']), 'checking moves differ'
    mates = set()
    for san in checks:
        b = board.copy()
        b.push_san(san)
        if b.is_checkmate():
            mates.add(san)
    assert mates == set(record['mates1']), 'mate-in-one moves differ'
    for line in record['lines']:
        b = board.copy()
        b.push_san(line['first'])
        assert b.is_check() and not b.is_checkmate(), 'first move must check without mating'
        assert {b.san(r) for r in b.legal_moves} == set(line['replies']), 'replies are not all covered'
        for reply, answers in line['replies'].items():
            c = b.copy()
            c.push_san(reply)
            assert answers, 'a reply without a mate'
            for answer in answers:
                d = c.copy()
                d.push_san(answer)
                assert d.is_checkmate(), 'answer is not mate'
    expected = 'mate1' if record['mates1'] else 'mate2' if record['lines'] else 'control'
    assert record['class'] == expected, 'class label differs'


def classify(board):
    checks = checks_with_replies(board)
    if not checks:
        return None
    mates1 = [m for m, _ in checks if _mates(board, m)]
    lines = [] if mates1 else forcing_mates_in_two(board)
    record = dict(fen=board.fen(), checks=[board.san(m) for m, _ in checks],
                  check_replies={board.san(m): n for m, n in checks},
                  mates1=[board.san(m) for m in mates1],
                  lines=[dict(first=board.san(first),
                              replies={_after(board, first).san(r): [_after(_after(board, first), r).san(a) for a in mates]
                                       for r, mates in answers.items()})
                         for first, answers in lines],
                  legal=board.legal_moves.count())
    record['class'] = 'mate1' if mates1 else 'mate2' if lines else 'control'
    return record


def _after(board, move):
    b = board.copy(stack=False)
    b.push(move)
    return b


def scan_row(args):
    row_id, text = args
    out = []
    for point in probe_points(text, limit=CONTEXT):
        if point.kind == 'space_white':
            continue
        record = classify(point.board)
        if record is None:
            continue
        record.update(row=int(row_id), index=point.index, kind=point.kind, ply=point.ply, game=point.game,
                      human=point.next_san)
        out.append(record)
    return out


def held_out_rows(limit):
    """Validation rows outside the probe selection, in a fixed order."""
    data = ChessData(Path('data/chess_8M_v1'), CONTEXT, verify_hashes=False)
    val = data.rows['val']
    probe_rows = set(np.random.default_rng(ROW_SEED).choice(len(val), ROWS, replace=False).tolist())
    ids = [i for i in range(len(val)) if i not in probe_rows][:limit]
    return ids, val, data.meta


def build(args):
    ids, val, meta = held_out_rows(args.rows)
    tasks = [(i, decode(val[i], meta)) for i in ids]
    by_class, controls, seen, offered = defaultdict(list), defaultdict(list), set(), Counter()
    rng = random.Random(0)
    with ProcessPoolExecutor(args.workers) as pool:
        for records in pool.map(scan_row, tasks, chunksize=32):
            for record in records:
                if record['class'] == 'control':
                    stratum = _stratum(record)
                    offered[stratum] += 1
                    if len(controls[stratum]) < CONTROL_POOL:
                        controls[stratum].append(record)
                    elif rng.random() < CONTROL_POOL / offered[stratum]:
                        controls[stratum][rng.randrange(CONTROL_POOL)] = record
                    continue
                key = (record['row'], record['game'], record['class'])
                if key not in seen:
                    seen.add(key)
                    by_class[record['class']].append(record)
    # Controls matched to the mate-in-two positions on side to move, ply bucket and number
    # of checks, using each game at most once.
    matched, unmatched, used_games = [], 0, set()
    for record in by_class['mate2']:
        pool_ = [c for c in controls.get(_stratum(record), []) if (c['row'], c['game']) not in used_games]
        if pool_:
            choice = pool_[rng.randrange(len(pool_))]
            controls[_stratum(record)].remove(choice)
            used_games.add((choice['row'], choice['game']))
            matched.append(choice)
        else:
            unmatched += 1
    by_class['control'] = matched
    for records in by_class.values():
        for record in records:
            verify(record)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'positions.jsonl', 'w') as stream:
        for cls in ('mate1', 'mate2', 'control'):
            for record in by_class[cls]:
                stream.write(json.dumps(record) + '\n')
    stats = dict(rows_scanned=len(ids), unmatched_mate2=unmatched,
                 counts={cls: len(v) for cls, v in by_class.items()})
    for cls, records in by_class.items():
        stats[cls] = dict(
            mean_ply=float(np.mean([r['ply'] for r in records])) if records else None,
            mean_checks=float(np.mean([len(r['checks']) for r in records])) if records else None,
            white_to_move=float(np.mean([r['kind'] == 'dot' for r in records])) if records else None,
            human_played_forcing=_human_rate(records, cls))
    two = [r for r in by_class['mate2'] if len(r['lines']) >= 2]
    stats['mate2_lines'] = dict(Counter(len(r['lines']) for r in by_class['mate2']))
    stats['mate2_single_reply_lines'] = sum(all(len(l['replies']) == 1 for l in r['lines']) for r in by_class['mate2'])
    stats['mate2_with_two_or_more_lines'] = len(two)
    (OUT / 'build.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


def _stratum(record):
    return record['kind'], record['ply'] // PLY_BUCKET, min(len(record['checks']), 8)


def forcing_sans(record):
    return set(record['mates1']) if record['class'] == 'mate1' else {l['first'] for l in record['lines']}


def _human_rate(records, cls):
    if cls == 'control':
        return None
    known = [r for r in records if r['human']]
    return float(np.mean([r['human'] in forcing_sans(r) for r in known])) if known else None


@torch.no_grad()
def move_log_probs(runner, prefixes, sans, stoi, device_name, batch=32):
    """Teacher-forced log-probability of each SAN string after its prefix."""
    out = []
    for start in range(0, len(prefixes), batch):
        seqs = [list(p) + [stoi[c] for c in s] for p, s in zip(prefixes[start:start + batch], sans[start:start + batch])]
        idx, lengths = pad(seqs, device_name)
        run = runner.run(idx, valid=valid_positions(lengths, idx.shape[1]))
        logp = F.log_softmax(run.logits.float(), -1)
        for i, (p, s) in enumerate(zip(prefixes[start:start + batch], sans[start:start + batch])):
            positions = torch.arange(len(p) - 1, len(p) - 1 + len(s), device=device_name)
            targets = torch.tensor([stoi[c] for c in s], device=device_name)
            out.append(float(logp[i, positions, targets].sum()))
    return out


def behaviour(args):
    records = [json.loads(line) for line in open(OUT / 'positions.jsonl')]
    rng = random.Random(1)
    chosen = []
    for cls, limit in (('mate1', args.mate1), ('mate2', args.mate2), ('control', args.mate2)):
        # The prefix plus the longest written move must fit the context.
        pool_ = [r for r in records if r['class'] == cls and r['index'] + 1 + MOVE_ROOM <= CONTEXT]
        chosen += rng.sample(pool_, min(limit, len(pool_)))
    _, val, meta = held_out_rows(0)
    stoi, itos = meta['stoi'], meta['itos']
    for name in args.arms:
        runner, _ = load_arm(name, args.device)
        prefixes = [np.array(val[r['row']][:r['index'] + 1], dtype=np.int64) for r in chosen]
        greedy = []
        for start in range(0, len(prefixes), 64):
            greedy += greedy_moves(runner, prefixes[start:start + 64], itos, args.device)
        flat = [(i, san) for i, r in enumerate(chosen) for san in r['checks']]
        logps = move_log_probs(runner, [prefixes[i] for i, _ in flat], [s for _, s in flat], stoi, args.device)
        per_position = defaultdict(dict)
        for (i, san), lp in zip(flat, logps):
            per_position[i][san] = lp
        rows = []
        for i, record in enumerate(chosen):
            probs = {san: float(np.exp(lp)) for san, lp in per_position[i].items()}
            forcing = forcing_sans(record)
            total = sum(probs.values())
            rows.append(dict(cls=record['class'], checks=len(record['checks']), forcing=sorted(forcing),
                             greedy=greedy[i], greedy_forcing=greedy[i] in forcing,
                             greedy_check=greedy[i] in probs, p_checks=total,
                             forcing_share=sum(probs[s] for s in forcing) / total if total > 0 and forcing else None,
                             uniform_share=len(forcing) / len(probs) if forcing else None,
                             probs=probs, check_replies=record['check_replies']))
        summary = summarise(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'behaviour-{name}.json').write_text(json.dumps(dict(arm=name, summary=summary, positions=rows),
                                                               indent=1) + '\n')
        print(name, json.dumps(summary, indent=1), flush=True)


def summarise(rows):
    out = {}
    for cls in ('mate1', 'mate2', 'control'):
        sub = [r for r in rows if r['cls'] == cls]
        if not sub:
            continue
        entry = dict(positions=len(sub), greedy_plays_check=float(np.mean([r['greedy_check'] for r in sub])),
                     mean_p_checks=float(np.mean([r['p_checks'] for r in sub])))
        if cls != 'control':
            entry.update(greedy_plays_forcing=float(np.mean([r['greedy_forcing'] for r in sub])),
                         mean_forcing_share=float(np.mean([r['forcing_share'] for r in sub])),
                         mean_uniform_share=float(np.mean([r['uniform_share'] for r in sub])))
            entry['reply_matched'] = reply_matched(sub)
        out[cls] = entry
    return out


def reply_matched(rows):
    """Forcing vs non-forcing checks with the same number of legal replies, within positions."""
    wins, pairs = 0, 0
    for r in rows:
        forcing = set(r['forcing'])
        for f in forcing:
            for other, p in r['probs'].items():
                if other not in forcing and r['check_replies'][other] == r['check_replies'][f]:
                    pairs += 1
                    wins += r['probs'][f] > p
    return dict(pairs=pairs, forcing_more_likely=wins / pairs if pairs else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build', 'behaviour'])
    parser.add_argument('--rows', type=int, default=20000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--arms', nargs='+', default=['transformer', 'temporal', 'depth', 'hybrid', 'karvonen'])
    parser.add_argument('--mate1', type=int, default=300)
    parser.add_argument('--mate2', type=int, default=500)
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    build(args) if args.command == 'build' else behaviour(args)


if __name__ == '__main__':
    main()
