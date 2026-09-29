"""Convert character rows into move-token games, split by game.

    uv run python -m moves.prepare --source data/chess_8M_v1 --out data/moves_v1 --workers 8

The source's train and val rows are pooled. Every game in every row is replayed
with python-chess (``moves.games.parse_row``); the ply cut off by a row's end is
dropped. Each game goes to a split by a hash of its ply ids, so identical games
always share a split: by default 0.5% test, 0.5% dev and the rest train. The
manifest is written last and marks the dataset as complete.
"""

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import pickle
import subprocess

import numpy as np

from moves.data import SplitWriter, file_hash, write_manifest
from moves.games import parse_row

ROOT = Path(__file__).resolve().parents[1]
_LUT = None


def split_of(game, test_per_mille=5, dev_per_mille=5):
    """Split of a game from the sha256 of its ply ids: the first 8 bytes, little-endian, modulo 1000."""
    digest = hashlib.sha256(np.asarray(game, dtype=np.uint16).tobytes()).digest()
    bucket = int.from_bytes(digest[:8], 'little') % 1000
    return 'test' if bucket < test_per_mille else 'dev' if bucket < test_per_mille + dev_per_mille else 'train'


def _initialise(lut):
    global _LUT
    _LUT = lut


def _convert(rows):
    games, failures = [], 0
    for row in rows:
        row_games, row_failures = parse_row(_LUT[row].tobytes().decode('ascii'))
        games.extend(row_games)
        failures += row_failures
    return games, failures


def _chunks(arrays, max_rows, chunk_rows):
    for rows in arrays:
        count = len(rows) if max_rows is None else min(max_rows, len(rows))
        for start in range(0, count, chunk_rows):
            yield np.array(rows[start:min(count, start + chunk_rows)])


def prepare(source, out, *, max_rows=None, workers=1, chunk_rows=10000, test_per_mille=5, dev_per_mille=5):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a new data version')
    with open(source / 'meta.pkl', 'rb') as stream:
        itos = pickle.load(stream)['itos']
    lut = np.array([ord(itos[index]) for index in range(len(itos))], dtype=np.uint8)
    source_manifest = json.loads((source / 'manifest.json').read_text())
    row_size = source_manifest['row_size']
    arrays = [np.memmap(source / f'{split}.bin', dtype=np.uint8, mode='r').reshape(-1, row_size)
              for split in ('train', 'val')]
    writers = {split: SplitWriter(out, split) for split in ('train', 'dev', 'test')}
    failures = 0
    chunks = _chunks(arrays, max_rows, chunk_rows)
    pool = Pool(workers, initializer=_initialise, initargs=(lut,)) if workers > 1 else None
    if pool is None:
        _initialise(lut)
    try:
        # imap keeps the chunk order, so the output doesn't depend on the worker count.
        for games, chunk_failures in (pool.imap(_convert, chunks) if pool else map(_convert, chunks)):
            failures += chunk_failures
            for game in games:
                writers[split_of(game, test_per_mille, dev_per_mille)].append(game)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    splits = {split: writer.close() for split, writer in writers.items()}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return write_manifest(out, dict(
        splits=splits, parse_failures=failures,
        split_rule=dict(hash='sha256 of the uint16 ply ids; first 8 bytes little-endian, modulo 1000',
                        test_per_mille=test_per_mille, dev_per_mille=dev_per_mille),
        source=dict(path=str(source), manifest_sha256=file_hash(source / 'manifest.json'),
                    source_splits=['train', 'val'], max_rows_per_split=max_rows),
        preparation_script_sha256=file_hash(__file__), preprocessing_commit=commit,
        preprocessing_dirty=dirty))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='data/chess_8M_v1')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-rows', type=int, help='Leading rows of each source split (for pilots and tests)')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    manifest = prepare(args.source, args.out, max_rows=args.max_rows, workers=args.workers)
    print(json.dumps({split: {key: info[key] for key in ('games', 'plies', 'max_plies')}
                      for split, info in manifest['splits'].items()}, indent=2))
    print(f"parse failures: {manifest['parse_failures']}")


if __name__ == '__main__':
    main()
