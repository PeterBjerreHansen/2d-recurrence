"""A tactics set from human dev games: positions where a little calculation changes the best move.

    uv run python -m experiments.move_pilot.tactics export --games 5000 --out tactics/games.npz      # laptop
    uv run python -m experiments.move_pilot.tactics scan --games tactics/games.npz --out tactics/scan.npz \
        --stockfish /usr/games/stockfish --nodes 200000 --workers 90                                    # CPU instance
    uv run python -m experiments.move_pilot.tactics label --scan tactics/scan.npz --games tactics/games.npz \
        --out tactics/panel.npz --stockfish /usr/games/stockfish --nodes 200000 --workers 90           # CPU instance
    uv run python -m experiments.move_pilot.tactics score --panel tactics/panel.npz                    # laptop

1. ``export``: the plies of the first human dev games of the full stage-1 dataset.
2. ``scan``: every position is searched by Stockfish twice: at depth 1 (essentially the static
   evaluation and captures: no calculation) and at a fixed node budget (a little calculation). Where the
   two best moves differ, the depth-1 move is searched again at the same budget, so its loss in win
   probability Q is known.
3. ``label``: tactics are positions where the depth-1 move loses at least ``--min-regret`` Q under the
   deep search; each becomes up to three steps along the deep line (plies +0, +2, +4, the same side to
   move), with the game's history plus the line so far. Controls are positions where both searches agree.
   Every step gets the teacher's labels (``moves.teacher``): one deep search per legal move, from the
   mover's side, with history. The labels don't depend on any model.
4. ``score``: each final stage-2 model (``full_engine_{arm}``; depth and hybrid at J = 1, 2, 4, 5)
   picks its top legal move at every step; reports agreement with the deep move and regret.

Q is from the mover's perspective throughout (``moves.values``).
"""

import argparse
import hashlib
import json
import pickle
import random
import shutil
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine
import numpy as np

from moves.values import q_from_centipawns, q_from_mate
from moves.vocab import GAME_START, MOVE_COUNT, PAD, id_move, move_id

CONTEXT = 256
STEP_PLIES = (0, 2, 4)


def root_q(score, mover):
    """Q of the side to move from a root search score."""
    score = score.pov(mover)
    if score.is_mate():
        mate = score.mate()
        return q_from_mate(mate) if mate != 0 else 0.0
    return q_from_centipawns(score.score())


def export(arguments):
    from moves.rows import RowData
    data = RowData('data/stage1_full_v1', CONTEXT, verify_hashes=False)
    tokens = np.asarray(data._split('human_dev')['tokens'][:arguments.games])
    games = [row[1:][row[1:] < MOVE_COUNT].astype(np.uint16) for row in tokens]
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.out, plies=np.concatenate(games), offsets=np.cumsum([0] + [len(g) for g in games]))
    print(f'{len(games)} games, {sum(len(g) for g in games)} positions -> {arguments.out}')


def load_games(path):
    z = np.load(path)
    return [z['plies'][a:b] for a, b in zip(z['offsets'][:-1], z['offsets'][1:])]


_ENGINE = None


def _engine(path, hash_mb=16):
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = chess.engine.SimpleEngine.popen_uci(path)
        _ENGINE.configure({'Threads': 1, 'Hash': hash_mb})
    return _ENGINE


def _scan_game(task):
    """Depth-1 and deep best moves for every position of one game; the deep Q of the depth-1 move."""
    index, plies, path, nodes = task
    engine = _engine(path)
    board, out = chess.Board(), []
    for t, ply in enumerate(plies):
        if not board.is_game_over():
            game = object()   # a new game before each search: no hash carried between searches
            shallow = engine.analyse(board, chess.engine.Limit(depth=1), game=game)
            deep = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=object())
            shallow_move, pv = shallow['pv'][0], deep['pv']
            q_best = root_q(deep['score'], board.turn)
            q_shallow = q_best
            if shallow_move != pv[0]:
                again = engine.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=[shallow_move], game=object())
                q_shallow = root_q(again['score'], board.turn)
            line = [move_id(m) for m in pv[:max(STEP_PLIES) + 1]]
            out.append((index, t, move_id(shallow_move), move_id(pv[0]), q_best, q_shallow, line))
        board.push(id_move(int(ply)))
    return out


def scan(arguments):
    games = load_games(arguments.games)
    tasks = [(i, g, arguments.stockfish, arguments.nodes) for i, g in enumerate(games)]
    rows = []
    with Pool(arguments.workers) as pool:
        for done, result in enumerate(pool.imap_unordered(_scan_game, tasks, chunksize=4), 1):
            rows.extend(result)
            if done % 250 == 0:
                print(f'{done}/{len(games)} games, {len(rows)} positions', flush=True)
    rows.sort()
    line = np.full((len(rows), max(STEP_PLIES) + 1), -1, dtype=np.int32)
    for r, row in enumerate(rows):
        line[r, :len(row[6])] = row[6]
    np.savez_compressed(arguments.out, game=np.array([r[0] for r in rows]), ply=np.array([r[1] for r in rows]),
                        shallow=np.array([r[2] for r in rows]), deep=np.array([r[3] for r in rows]),
                        q_best=np.array([r[4] for r in rows]), q_shallow=np.array([r[5] for r in rows]), line=line,
                        meta=json.dumps(engine_meta(arguments.stockfish, arguments.nodes)))
    regret = np.array([r[4] - r[5] for r in rows])
    print(f'{len(rows)} positions; depth-1 and deep moves differ at {np.mean([r[2] != r[3] for r in rows]):.1%}; '
          f'regret >= 0.10 at {np.mean(regret >= 0.10):.1%}')


def engine_meta(path, nodes):
    resolved = shutil.which(path) or path
    engine = chess.engine.SimpleEngine.popen_uci(resolved)
    name = engine.id.get('name')
    engine.quit()
    return dict(name=name, path=resolved, sha256=hashlib.sha256(Path(resolved).read_bytes()).hexdigest(),
                threads=1, hash_mb=16, tablebases=None, nodes=nodes, shallow='depth 1')


def _label_step(task):
    from moves.teacher import Teacher
    key, history, path, nodes = task
    global _TEACHER
    if '_TEACHER' not in globals() or _TEACHER is None:
        _TEACHER = Teacher.open(path, nodes)
    board = chess.Board()
    for ply in history:
        board.push(id_move(int(ply)))
    ids, labels = _TEACHER.label(board)
    return key, np.asarray(ids, dtype=np.int32), labels['q'].astype(np.float32)


_TEACHER = None


def plan_steps(arguments):
    """The tactics, controls and steps to label; deterministic, so a restart or ``assemble`` sees the same list."""
    scan_data, games = np.load(arguments.scan), load_games(arguments.games)
    regret = scan_data['q_best'] - scan_data['q_shallow']
    differ = scan_data['shallow'] != scan_data['deep']
    rng = np.random.default_rng(0)
    tactic = np.flatnonzero(differ & (regret >= arguments.min_regret))
    control = np.flatnonzero(~differ)
    tactic = np.sort(rng.choice(tactic, min(arguments.max_tactics, len(tactic)), replace=False))
    control = np.sort(rng.choice(control, min(arguments.max_controls or len(tactic), len(control)), replace=False))
    steps = []   # dict per step: kind, source position, step index, history plies, and source features
    for kind, chosen in (('tactic', tactic), ('control', control)):
        for p in chosen:
            g, t, line = int(scan_data['game'][p]), int(scan_data['ply'][p]), scan_data['line'][p]
            board = chess.Board()
            for ply in games[g][:t]:
                board.push(id_move(int(ply)))
            walk, forcing = board.copy(), []
            for x in line:
                if x < 0:
                    break
                move = id_move(int(x))
                forcing.append(walk.is_capture(move) or walk.gives_check(move))
                walk.push(move)
            source = dict(source=int(p), regret=float(regret[p]), q_best=float(scan_data['q_best'][p]),
                          forcing_first=bool(forcing[:1] and forcing[0]), forcing_count=sum(forcing))
            history = [int(x) for x in games[g][:t]]
            for s, offset in enumerate(STEP_PLIES if kind == 'tactic' else (0,)):
                prefix = [int(x) for x in line[:offset]]
                if len(prefix) < offset or -1 in prefix or len(history) + offset >= CONTEXT - 1:
                    break
                step_board = board.copy()
                for x in prefix:
                    step_board.push(id_move(x))
                if step_board.is_game_over():
                    break
                steps.append(dict(kind=kind, step=s, history=history + prefix, **source))
    meta = dict(engine=json.loads(str(scan_data['meta'])), min_regret=arguments.min_regret, step_plies=STEP_PLIES,
                games=arguments.games)
    return steps, len(tactic), len(control), meta


def write_panel(out, steps, meta):
    """Steps (dicts with ``ids`` and ``q`` labels) as one panel file."""
    np.savez_compressed(
        out, kind=np.array([st['kind'] for st in steps]), source=np.array([st['source'] for st in steps]),
        step=np.array([st['step'] for st in steps]),
        history=np.concatenate([np.asarray(st['history'], dtype=np.uint16) for st in steps]),
        history_offsets=np.cumsum([0] + [len(st['history']) for st in steps]),
        ids=np.concatenate([st['ids'] for st in steps]), q=np.concatenate([st['q'] for st in steps]),
        label_offsets=np.cumsum([0] + [len(st['ids']) for st in steps]),
        source_regret=np.array([st['regret'] for st in steps]), source_q_best=np.array([st['q_best'] for st in steps]),
        forcing_first=np.array([st['forcing_first'] for st in steps]),
        forcing_count=np.array([st['forcing_count'] for st in steps]), meta=json.dumps(meta))


def label(arguments):
    steps, n_tactic, n_control, meta = plan_steps(arguments)
    print(f'{n_tactic} tactics, {n_control} controls, {len(steps)} steps to label', flush=True)
    # Progress is saved every 1,000 steps, so a restart (after a spot reclaim) skips finished steps.
    partial = Path(str(arguments.out) + '.partial.pkl')
    labelled = pickle.loads(partial.read_bytes()) if partial.exists() else {}
    if labelled:
        print(f'resuming: {len(labelled)} steps already labelled', flush=True)
    tasks = [(k, st['history'], arguments.stockfish, arguments.nodes) for k, st in enumerate(steps) if k not in labelled]
    with Pool(arguments.workers) as pool:
        for done, (k, ids, q) in enumerate(pool.imap_unordered(_label_step, tasks, chunksize=2), 1):
            labelled[k] = (ids, q)
            if done % 1000 == 0 or done == len(tasks):
                partial.write_bytes(pickle.dumps(labelled))
                print(f'{len(labelled)}/{len(steps)} steps labelled', flush=True)
    for k, st in enumerate(steps):
        st['ids'], st['q'] = labelled[k]
    write_panel(arguments.out, steps, meta)
    print(f'wrote {arguments.out}')


def assemble(arguments):
    """A panel from saved labelling progress: the first ``--tactics`` tactics with every step labelled, plus the
    control steps of ``--controls-from`` (an earlier panel from the same scan and node budget)."""
    steps, _, _, meta = plan_steps(arguments)
    labelled = pickle.loads(Path(arguments.partial).read_bytes())
    by_source = {}
    for k, st in enumerate(steps):
        if st['kind'] == 'tactic':
            by_source.setdefault(st['source'], []).append(k)
    complete = [source for source, keys in by_source.items() if all(k in labelled for k in keys)]
    print(f'{len(complete)} tactics with every step labelled ({len(labelled)} steps saved)', flush=True)
    if arguments.count_only:
        return
    chosen = complete[:arguments.tactics]
    panel = []
    for source in chosen:
        for k in by_source[source]:
            steps[k]['ids'], steps[k]['q'] = labelled[k]
            panel.append(steps[k])
    z = np.load(arguments.controls_from)
    for k in np.flatnonzero(z['kind'] == 'control'):
        a, b = z['history_offsets'][k], z['history_offsets'][k + 1]
        c, d = z['label_offsets'][k], z['label_offsets'][k + 1]
        panel.append(dict(kind='control', source=int(z['source'][k]), step=0, history=z['history'][a:b],
                          ids=z['ids'][c:d], q=z['q'][c:d], regret=float(z['source_regret'][k]),
                          q_best=float(z['source_q_best'][k]), forcing_first=False, forcing_count=0))
    meta.update(assembled_from=str(arguments.partial), controls_from=str(arguments.controls_from), tactics=len(chosen))
    write_panel(arguments.out, panel, meta)
    print(f'wrote {arguments.out}: {len(chosen)} tactics ({sum(st["kind"] == "tactic" for st in panel)} steps), '
          f'{int((z["kind"] == "control").sum())} controls')


def score(arguments):
    import torch
    from experiments.move_pilot.stratify_stage2 import CELLS, load
    from moves.objectives import LegalTargets
    from recurrence.schedule import sample_schedule
    z = np.load(arguments.panel)
    n = len(z['kind'])
    histories = [z['history'][a:b] for a, b in zip(z['history_offsets'][:-1], z['history_offsets'][1:])]
    legal = [(z['ids'][a:b], z['q'][a:b]) for a, b in zip(z['label_offsets'][:-1], z['label_offsets'][1:])]
    best = np.array([ids[np.argmax(q)] for ids, q in legal])
    q_max = np.array([q.max() for _, q in legal])
    results = {}
    for arm, cells in CELLS.items():
        model = load(arm, arguments.device, arguments.run, arguments.checkpoint)
        for name, cell in cells:
            picks = np.zeros(n, dtype=np.int64)
            p_best = np.zeros(n)
            for start in range(0, n, arguments.batch_size):
                batch = range(start, min(n, start + arguments.batch_size))
                x = np.full((len(batch), CONTEXT), PAD, dtype=np.int64)
                positions, owner, moves = [], [], []
                for b, k in enumerate(batch):
                    h = histories[k]
                    x[b, 0] = GAME_START
                    x[b, 1:1 + len(h)] = h
                    positions.append(b * CONTEXT + len(h))
                    owner += [b] * len(legal[k][0])
                    moves += list(legal[k][0])
                targets = LegalTargets(torch.tensor(positions), torch.tensor(owner), torch.tensor(moves))
                kwargs = {} if cell is None else dict(schedule=sample_schedule(*cell, random.Random(0)))
                with torch.no_grad():
                    logits, _ = model(torch.from_numpy(x).to(arguments.device), targets.to(arguments.device), **kwargs)
                logits = logits.float().cpu().reshape(-1, logits.size(-1))
                for b, k in enumerate(batch):
                    ids = legal[k][0]
                    z_legal = logits[positions[b], torch.from_numpy(ids.astype(np.int64))]
                    p = torch.softmax(z_legal, 0).numpy()
                    picks[k] = ids[int(np.argmax(p))]
                    p_best[k] = p[list(ids).index(best[k])]
            q_pick = np.array([legal[k][1][list(legal[k][0]).index(picks[k])] for k in range(n)])
            results[f'{arm}_{name}'] = dict(agree=picks == best, regret=q_max - q_pick, p_best=p_best)
            print(f'scored {arm} {name}', flush=True)
        del model
    np.savez_compressed(Path(arguments.panel).with_name(f'panel_scores_{arguments.run}_{Path(arguments.checkpoint).stem}.npz'),
                        **{f'{k}__{m}': v[m] for k, v in results.items() for m in v})
    groups = {'controls': z['kind'] == 'control'}
    tactic = z['kind'] == 'tactic'
    for s in range(len(STEP_PLIES)):
        groups[f'tactic step {s} (ply +{STEP_PLIES[s]})'] = tactic & (z['step'] == s)
    if 'forcing_first' in z:
        groups['forcing tactics, step 0'] = tactic & (z['step'] == 0) & z['forcing_first']
        groups['forcing tactics, steps 1-2'] = tactic & (z['step'] >= 1) & z['forcing_first']
        groups['quiet tactics, step 0'] = tactic & (z['step'] == 0) & ~z['forcing_first']
        groups['swing >= 0.25, step 0'] = tactic & (z['step'] == 0) & (z['source_regret'] >= 0.25)
        groups['swing >= 0.25, steps 1-2'] = tactic & (z['step'] >= 1) & (z['source_regret'] >= 0.25)
    for metric, label_ in (('agree', 'agreement with the deep move'), ('regret', 'regret of the top legal move (Q)')):
        columns = list(results)
        print(f'\n{label_}')
        print(f"{'group':<26}{'n':>7}" + ''.join(f'{c:>16}' for c in columns))
        for group, mask in groups.items():
            print(f'{group:<26}{mask.sum():>7}' + ''.join(f'{results[c][metric][mask].mean():>16.4f}' for c in columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    e = commands.add_parser('export')
    e.add_argument('--games', type=int, default=5000)
    e.add_argument('--out', required=True)
    s = commands.add_parser('scan')
    s.add_argument('--games', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--stockfish', default='stockfish')
    s.add_argument('--nodes', type=int, default=200_000)
    s.add_argument('--workers', type=int, default=8)
    lab = commands.add_parser('label')
    lab.add_argument('--scan', required=True)
    lab.add_argument('--games', required=True)
    lab.add_argument('--out', required=True)
    lab.add_argument('--stockfish', default='stockfish')
    lab.add_argument('--nodes', type=int, default=200_000)
    lab.add_argument('--workers', type=int, default=8)
    lab.add_argument('--min-regret', type=float, default=0.10)
    lab.add_argument('--max-tactics', type=int, default=3000)
    lab.add_argument('--max-controls', type=int, help='Default: as many as tactics')
    asm = commands.add_parser('assemble')
    asm.add_argument('--scan', required=True)
    asm.add_argument('--games', required=True)
    asm.add_argument('--partial', required=True)
    asm.add_argument('--out')
    asm.add_argument('--controls-from')
    asm.add_argument('--tactics', type=int, default=8000)
    asm.add_argument('--count-only', action='store_true')
    asm.add_argument('--min-regret', type=float, default=0.10)
    asm.add_argument('--max-tactics', type=int, default=20000)
    asm.add_argument('--max-controls', type=int, default=4000)
    sc = commands.add_parser('score')
    sc.add_argument('--panel', required=True)
    sc.add_argument('--run', default='full_engine', help='Run name prefix; the arm is appended')
    sc.add_argument('--checkpoint', default='ckpt.pt')
    sc.add_argument('--batch-size', type=int, default=64)
    sc.add_argument('--device', default='mps')
    arguments = parser.parse_args()
    dict(export=export, scan=scan, label=label, assemble=assemble, score=score)[arguments.command](arguments)


if __name__ == '__main__':
    main()
