"""Build the stage-2 dataset from Leela Chess Zero training archives (plan, Datasets).

    cd data/leela/archives && curl -O \
        'https://storage.lczero.org/files/training_data/test80/training-run1-test80-20240401-[00-17]17.tar'
    uv run python -m moves.build_leela --archives data/leela/archives/*.tar --out data/leela_v1 \
        --positions 100000000 --eval-games 20000 --workers 8

Each archive is converted in a worker (``moves.leela.convert_game``: verified
replay; Chess960 and failed games skipped) into game stores per archive and
split, with games longer than the context (at most 255 plies) truncated. The
training games of all archives are then shuffled with ``seed``, taken in that
order until ``positions``, and packed into rows; the stores are deleted. Dev and
test files hold up to ``eval_games`` games each, one game per row. Targets are
Leela's search probabilities, stored raw, with the position value (q, d) and
visit count.
"""

import argparse
from collections import Counter
import json
from multiprocessing import Pool
from pathlib import Path
import shutil
import subprocess

import numpy as np

from data_loader import file_hash
from moves.games import MAX_PLIES
from moves.leela import SkippedGame, convert_game, read_archive
from moves.rows import GameRecord, GameStore, GameStoreWriter, pack, split_of, write_dataset_manifest

SOURCES = ('leela',)
ROOT = Path(__file__).resolve().parents[1]


def game_record(game, limit=MAX_PLIES):
    """A ``GameRecord`` of a converted Leela game, truncated at ``limit`` plies."""
    n = min(len(game.plies), limit)
    legal, probabilities = game.legal[:n], game.probabilities[:n]
    values = np.stack([game.values['root_q'][:n], game.values['root_d'][:n]], axis=1).astype(np.float16)
    return GameRecord(0, game.plies[:n], np.array([len(moves) for moves in legal], dtype=np.uint8),
                      np.concatenate(legal).astype(np.uint16), np.concatenate(probabilities).astype(np.float16),
                      values, np.minimum(game.values['visits'][:n], np.iinfo(np.uint16).max).astype(np.uint16))


def convert_archive(task):
    """Convert one archive into game stores ``{stores}/{archive}/{split}``; return its statistics."""
    archive, stores, split_rule, limit = task
    archive = Path(archive)
    writers = {split: GameStoreWriter(Path(stores) / archive.stem / split, policy=True)
               for split in ('train', 'dev', 'test')}
    stats = dict(archive=archive.name, sha256=file_hash(archive), games=Counter(), positions=Counter(),
                 truncated=0, skipped=Counter())
    for _, records in read_archive(archive):
        try:
            game = convert_game(records)
        except SkippedGame as error:
            stats['skipped'][str(error).split(': ', 1)[-1]] += 1
            continue
        record = game_record(game, limit)
        split = split_of(record.plies, *split_rule)
        writers[split].append(record)
        stats['games'][split] += 1
        stats['positions'][split] += len(record.plies)
        stats['truncated'] += len(game.plies) > limit
    for writer in writers.values():
        writer.close()
    return {key: dict(value) if isinstance(value, Counter) else value for key, value in stats.items()}


def build(archives, out, *, positions, eval_games=20000, context=256, seed=0, workers=1,
          test_per_mille=5, dev_per_mille=5):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a new data version')
    archives = [Path(path) for path in archives]
    stores = out / 'stores'
    limit = min(MAX_PLIES, context)
    tasks = [(str(path), str(stores), (test_per_mille, dev_per_mille), limit) for path in archives]
    if workers > 1:
        with Pool(workers) as pool:
            conversions = pool.map(convert_archive, tasks, chunksize=1)
    else:
        conversions = [convert_archive(task) for task in tasks]
    opened = {split: [GameStore(stores / path.stem / split) for path in archives] for split in ('train', 'dev', 'test')}
    train = [(store, index) for store, games in enumerate(opened['train']) for index in range(len(games))]
    order = np.random.default_rng(seed).permutation(len(train))
    available = sum(conversion['positions'].get('train', 0) for conversion in conversions)
    if available < positions:
        raise ValueError(f'The archives hold {available:,} training positions, fewer than the {positions:,} requested')

    def training_games():
        total = 0
        for position in order:
            if total >= positions:
                return
            store, index = train[position]
            record = opened['train'][store].record(index, 0)
            total += len(record.plies)
            yield record

    def evaluation_games(split):
        taken = 0
        for store in opened[split]:
            for index in range(len(store)):
                if taken == eval_games:
                    return
                taken += 1
                yield store.record(index, 0)

    splits = {'train': pack(out, 'train', training_games(), context=context, sources=SOURCES, policy=True)}
    for split in ('dev', 'test'):
        splits[f'leela_{split}'] = pack(out, f'leela_{split}', evaluation_games(split), context=context,
                                        sources=SOURCES, policy=True, one_game_per_row=True)
    shutil.rmtree(stores)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    skipped = Counter()
    for conversion in conversions:
        skipped.update(conversion['skipped'])
    return write_dataset_manifest(
        out, context=context, sources=SOURCES, targets='engine', splits=splits, eval_splits=['leela_dev'],
        build=dict(script='moves/build_leela.py', script_sha256=file_hash(__file__), commit=commit,
                   archives=[dict(name=c['archive'], sha256=c['sha256']) for c in conversions],
                   positions=positions, eval_games=eval_games, seed=seed, available_training_positions=available,
                   truncated_at_plies=limit, games_truncated=sum(c['truncated'] for c in conversions),
                   games_skipped=dict(skipped), values='root_q, root_d (side to move); visits',
                   split_rule=dict(hash='sha256 of the uint16 ply ids; first 8 bytes little-endian, modulo 1000',
                                   test_per_mille=test_per_mille, dev_per_mille=dev_per_mille)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archives', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--positions', type=int, required=True, help='Training positions')
    parser.add_argument('--eval-games', type=int, default=20000, help='Games in each dev and test file')
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    manifest = build(args.archives, args.out, positions=args.positions, eval_games=args.eval_games,
                     context=args.context, seed=args.seed, workers=args.workers)
    print(json.dumps({split: {key: info[key] for key in ('rows', 'games', 'positions')}
                      for split, info in manifest['splits'].items()}, indent=2))
    print('skipped games:', manifest['build']['games_skipped'])


if __name__ == '__main__':
    main()
