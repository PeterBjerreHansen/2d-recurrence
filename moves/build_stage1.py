"""Build the stage-1 dataset: Lichess and uniformly random games, with legal moves (plan, Datasets).

    uv run python -m moves.build_stage1 --source data/chess_8M_v1 --out data/stage1_v1 \\
        --positions 500000000 --random-fraction 0.2 --eval-games 20000 --workers 8

1. Lichess games are replayed from the character rows (train, then val), in
   stored order, which the upstream split already shuffled. Each game goes to a
   split by the hash of its plies; rows are read until the training quota of
   human positions is met.
2. Random games (uniform legal plies, capped at the context length, at most 255) are generated from
   ``(seed, split, index)``, so they don't depend on the worker count. Random dev
   and test games that occur in the training games are dropped.
3. Training games of both sources are shuffled together, their legal moves
   computed, and packed into rows. Dev and test files hold one game per row.

A position's targets are the legal moves before the next ply, so positions
equal plies. Games longer than the context are truncated to fit (never needed
for Lichess rows at context 256). The dataset is identical for any number of workers.
"""

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import pickle
import random
import subprocess

import numpy as np

from data_loader import file_hash
from moves.games import MAX_PLIES, legal_sets, parse_row, random_game
from moves.rows import GameRecord, game_hash, pack, split_of, write_dataset_manifest

SOURCES = ('human', 'random')
ROOT = Path(__file__).resolve().parents[1]
_LUT = None


def _initialise(lut):
    global _LUT
    _LUT = lut


def _parse(rows):
    games, failures = [], 0
    for row in rows:
        row_games, row_failures = parse_row(_LUT[row].tobytes().decode('ascii'))
        games.extend(np.asarray(game, dtype=np.uint16) for game in row_games)
        failures += row_failures
    return games, failures, len(rows)


def _random_games(task):
    seed, split, first, count, limit = task
    games = []
    for index in range(first, first + count):
        digest = hashlib.sha256(f'{seed}:{split}:{index}'.encode()).digest()
        game, _ = random_game(random.Random(int.from_bytes(digest[:8], 'little')), limit)
        games.append(np.asarray(game, dtype=np.uint16))
    return games


def _records(games):
    """GameRecords with the legal moves before each ply."""
    records = []
    for source, plies in games:
        sets = legal_sets(plies)
        records.append(GameRecord(source, plies, np.array([len(s) for s in sets], dtype=np.uint8),
                                  np.concatenate(sets).astype(np.uint16)))
    return records


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _windowed(pool, window):
    """A lazy ``map(function, items)`` that keeps at most ``window`` tasks in flight."""
    def run(function, items):
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) == window:
                yield from (pool.map(function, batch) if pool else map(function, batch))
                batch = []
        if batch:
            yield from (pool.map(function, batch) if pool else map(function, batch))
    return run


def _lut(source):
    with open(Path(source) / 'meta.pkl', 'rb') as stream:
        itos = pickle.load(stream)['itos']
    return np.array([ord(itos[index]) for index in range(len(itos))], dtype=np.uint8)


def human_games(source, quota, eval_games, imap, split_rule, limit, chunk_rows=2000):
    """Training games until ``quota`` positions, dev and test games up to ``eval_games`` each."""
    row_size = json.loads((Path(source) / 'manifest.json').read_text())['row_size']
    arrays = [np.memmap(Path(source) / f'{split}.bin', dtype=np.uint8, mode='r').reshape(-1, row_size)
              for split in ('train', 'val')]
    games = dict(train=[], dev=[], test=[])
    stats = dict(rows=0, parse_failures=0, train_positions=0, truncated=0)

    def chunks():
        for rows in arrays:
            for start in range(0, len(rows), chunk_rows):
                yield np.array(rows[start:start + chunk_rows])

    for chunk_games, failures, rows in imap(_parse, chunks()):
        stats['rows'] += rows
        stats['parse_failures'] += failures
        for game in chunk_games:
            stats['truncated'] += len(game) > limit
            game = game[:limit]
            split = split_of(game, *split_rule)
            if split == 'train':
                if stats['train_positions'] < quota:
                    games['train'].append(game)
                    stats['train_positions'] += len(game)
            elif len(games[split]) < eval_games:
                games[split].append(game)
        if stats['train_positions'] >= quota and all(len(games[s]) >= eval_games for s in ('dev', 'test')):
            break
    if stats['train_positions'] < quota:
        raise ValueError(f"The source holds only {stats['train_positions']:,} human training positions, "
                         f'fewer than the {quota:,} requested')
    return games, stats


def random_games(seed, split, imap, limit, *, positions=None, count=None, round_games=4000, chunk=250):
    """Random games in index order, until ``positions`` positions or ``count`` games.

    Games are generated in rounds; the ones past the quota are discarded, so the
    result doesn't depend on the round size or the number of workers.
    """
    games, total, first = [], 0, 0
    done = lambda: (positions is not None and total >= positions) or (count is not None and len(games) >= count)
    while not done():
        # Size the round to what is still needed (random games average well over 100 plies).
        needed = (count - len(games)) if count is not None else -(-(positions - total) // 100)
        size = min(round_games, max(1, needed))
        tasks = [(seed, split, start, min(chunk, first + size - start), limit)
                 for start in range(first, first + size, chunk)]
        first += size
        for batch in imap(_random_games, tasks):
            for game in batch:
                if done():
                    break
                games.append(game)
                total += len(game)
    return games


def build(source, out, *, positions, random_fraction=0.2, eval_games=20000, context=256, seed=0, workers=1,
          chunk_games=256, test_per_mille=5, dev_per_mille=5):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a new data version')
    if not 0 <= random_fraction < 1:
        raise ValueError('random_fraction must be in [0, 1)')
    random_quota = round(positions * random_fraction)
    human_quota = positions - random_quota
    lut = _lut(source)
    pool = Pool(workers, initializer=_initialise, initargs=(lut,)) if workers > 1 else None
    if pool is None:
        _initialise(lut)
    imap = _windowed(pool, 8 * workers)
    limit = min(MAX_PLIES, context)
    try:
        human, stats = human_games(source, human_quota, eval_games, imap, (test_per_mille, dev_per_mille), limit)
        randoms = dict(train=random_games(seed, 'train', imap, limit, positions=random_quota) if random_quota else [])
        train_hashes = {game_hash(game) for game in randoms['train']}
        stats['random_eval_duplicates_dropped'] = 0
        for split in ('dev', 'test'):
            candidates = random_games(seed, split, imap, limit, count=eval_games + 100)
            unique = [game for game in candidates if game_hash(game) not in train_hashes]
            stats['random_eval_duplicates_dropped'] += len(candidates) - len(unique)
            randoms[split] = unique[:eval_games]
        train = [(0, game) for game in human['train']] + [(1, game) for game in randoms['train']]
        order = np.random.default_rng(seed).permutation(len(train))
        train = [train[index] for index in order]

        def records(games):
            for chunk in imap(_records, _chunks(games, chunk_games)):
                yield from chunk

        splits = {'train': pack(out, 'train', records(train), context=context, sources=SOURCES, weighted=False)}
        for split in ('dev', 'test'):
            splits[f'human_{split}'] = pack(out, f'human_{split}', records([(0, g) for g in human[split]]),
                                            context=context, sources=SOURCES, weighted=False, one_game_per_row=True)
            splits[f'random_{split}'] = pack(out, f'random_{split}', records([(1, g) for g in randoms[split]]),
                                             context=context, sources=SOURCES, weighted=False, one_game_per_row=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    return write_dataset_manifest(
        out, context=context, sources=SOURCES, targets='legal', splits=splits,
        eval_splits=['human_dev', 'random_dev'],
        build=dict(script='moves/build_stage1.py', script_sha256=file_hash(__file__), commit=commit,
                   source=str(source), source_manifest_sha256=file_hash(Path(source) / 'manifest.json'),
                   positions=positions, random_fraction=random_fraction, eval_games=eval_games, seed=seed,
                   human_training_positions=stats['train_positions'],
                   random_training_positions=sum(len(g) for g in randoms['train']),
                   source_rows_read=stats['rows'], parse_failures=stats['parse_failures'],
                   game_ply_limit=limit, human_games_truncated=stats['truncated'],
                   random_eval_duplicates_dropped=stats['random_eval_duplicates_dropped'],
                   split_rule=dict(hash='sha256 of the uint16 ply ids; first 8 bytes little-endian, modulo 1000',
                                   test_per_mille=test_per_mille, dev_per_mille=dev_per_mille)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='data/chess_8M_v1')
    parser.add_argument('--out', required=True)
    parser.add_argument('--positions', type=int, required=True, help='Training positions, both sources together')
    parser.add_argument('--random-fraction', type=float, default=0.2)
    parser.add_argument('--eval-games', type=int, default=20000, help='Games in each dev and test file')
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    manifest = build(args.source, args.out, positions=args.positions, random_fraction=args.random_fraction,
                     eval_games=args.eval_games, context=args.context, seed=args.seed, workers=args.workers)
    print(json.dumps({split: {key: info[key] for key in ('rows', 'games', 'positions')}
                      for split, info in manifest['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()
