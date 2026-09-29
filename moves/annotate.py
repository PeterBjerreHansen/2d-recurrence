"""Label human games with the engine teacher and write an engine-labelled dataset.

    uv run python -m moves.annotate --source data/moves_v1 --out data/moves_v1_engine \\
        --engine "$(which stockfish)" --budgets shallow=2000 deep=20000 \\
        --games train=200000 dev=2000 test=2000 --workers 8

Games are a seeded random sample of each source split. Every position before a
ply is labelled at every budget (``moves.teacher``), each worker running its own
single-threaded engines. Positions early in a game repeat across games, so
labels of positions up to ``CACHE_PLIES`` deep are cached by their history key.
"""

import argparse
import json
from multiprocessing import Pool, util
from pathlib import Path

import chess
import numpy as np

from moves.data import MoveData, SplitWriter, file_hash, write_manifest
from moves.teacher import LABEL_DTYPE, Teacher, engine_metadata, history_key
from moves.vocab import id_move

CACHE_PLIES = 16


class StockfishTeachers:
    """Opens one teacher per budget; picklable, so each worker opens its own engines."""

    def __init__(self, path, budgets, hash_mb=16):
        self.path, self.budgets, self.hash_mb = str(path), dict(budgets), hash_mb

    def __call__(self):
        return {name: Teacher.open(self.path, nodes, self.hash_mb) for name, nodes in self.budgets.items()}

    def metadata(self, teachers):
        return {name: engine_metadata(self.path, teacher, self.hash_mb) for name, teacher in teachers.items()}


def label_game(game, teachers, cache):
    """Legal move ids and labels (one array per budget) for the position before each ply."""
    board, legal, labels = chess.Board(), [], {name: [] for name in teachers}
    for token in game:
        key = history_key(board) if len(board.move_stack) <= CACHE_PLIES else None
        if key is not None and key in cache:
            ids, by_budget = cache[key]
        else:
            by_budget = {}
            for name, teacher in teachers.items():
                budget_ids, by_budget[name] = teacher.label(board)
                if name == next(iter(teachers)):
                    ids = budget_ids
                elif budget_ids != ids:
                    raise RuntimeError('Budgets disagree on the legal moves')
            if key is not None:
                cache[key] = ids, by_budget
        legal.append(np.asarray(ids, dtype=np.uint16))
        for name in teachers:
            labels[name].append(by_budget[name])
        board.push(id_move(int(token)))
    return legal, labels


_TEACHERS, _CACHE = None, None


def _initialise(factory):
    global _TEACHERS, _CACHE
    _TEACHERS, _CACHE = factory(), {}
    # Pool workers skip atexit; a finalizer closes their engines when they exit.
    util.Finalize(None, _close, exitpriority=10)


def _close():
    for teacher in _TEACHERS.values():
        teacher.close()


def _label(game):
    return label_game(game, _TEACHERS, _CACHE)


def annotate(source, out, factory, games, *, seed=0, workers=1):
    """Write ``out``: the sampled games of each split with labels from ``factory()``'s teachers.

    ``games`` maps a split to its number of sampled games.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a new data version')
    data = MoveData(source, json.loads((Path(source) / 'manifest.json').read_text())['max_plies'])
    teachers = factory()
    try:
        metadata = factory.metadata(teachers) if hasattr(factory, 'metadata') else None
        budgets = {name: teacher.nodes for name, teacher in teachers.items()}
    finally:
        for teacher in teachers.values():
            teacher.close()
    splits, values = {}, {}
    rng = np.random.default_rng(seed)
    for split, count in games.items():
        available = len(data.offsets[split]) - 1
        if count > available:
            raise ValueError(f'{split} has {available} games, fewer than {count}')
        indices = np.sort(rng.choice(available, size=count, replace=False))
        selected = [np.asarray(data.game(split, index)) for index in indices]
        if workers > 1:
            with Pool(workers, initializer=_initialise, initargs=(factory,)) as pool:
                results = list(pool.imap(_label, selected, chunksize=8))
                pool.close()
                pool.join()
        else:
            split_teachers, cache = factory(), {}
            try:
                results = [label_game(game, split_teachers, cache) for game in selected]
            finally:
                for teacher in split_teachers.values():
                    teacher.close()
        writer = SplitWriter(out, split)
        for game in selected:
            writer.append(game)
        splits[split] = writer.close()
        legal = [ids for game_legal, _ in results for ids in game_legal]
        legal_offsets = np.concatenate(([0], np.cumsum([len(ids) for ids in legal]))).astype(np.int64)
        np.save(out / f'{split}_legal_offsets.npy', legal_offsets)
        np.concatenate(legal).astype(np.uint16).tofile(out / f'{split}_legal_moves.bin')
        names = [f'{split}_legal_offsets.npy', f'{split}_legal_moves.bin']
        bounded = {}
        for name in budgets:
            labels = np.concatenate([array for _, game_labels in results for array in game_labels[name]])
            np.save(out / f'{split}_labels_{name}.npy', labels.astype(LABEL_DTYPE))
            names.append(f'{split}_labels_{name}.npy')
            bounded[name] = int((labels['bound'] != 0).sum())
        values[split] = dict(positions=len(legal), legal_moves=int(legal_offsets[-1]), bounded_labels=bounded,
                             sha256={name: file_hash(out / name) for name in names})
    return write_manifest(out, dict(
        splits=splits,
        values=dict(budgets=budgets, engine=metadata, splits=values,
                    protocol='docs/engine_policy_plan.md, Engine values'),
        source=dict(path=str(source), manifest_sha256=data.manifest_hash, sample_seed=seed, games=games),
        annotation_script_sha256=file_hash(__file__)))


def _pairs(values, cast):
    pairs = {}
    for value in values:
        name, _, number = value.partition('=')
        pairs[name] = cast(number)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--engine', required=True, help='Path to the pinned Stockfish binary')
    parser.add_argument('--budgets', nargs='+', required=True, help='name=nodes, e.g. shallow=2000 deep=20000')
    parser.add_argument('--games', nargs='+', required=True, help='split=count, e.g. train=200000 dev=2000')
    parser.add_argument('--hash-mb', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    factory = StockfishTeachers(args.engine, _pairs(args.budgets, int), args.hash_mb)
    manifest = annotate(args.source, args.out, factory, _pairs(args.games, int), seed=args.seed,
                        workers=args.workers)
    print(json.dumps(manifest['values']['splits'], indent=2))


if __name__ == '__main__':
    main()
