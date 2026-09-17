"""Prepare Karvonen's character blocks, preserving the upstream row split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import subprocess
import tempfile

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(rows, out_dir, *, source, seed=2357, val_fraction=0.01, row_size=1024):
    """Publish verified files only after both splits pass validation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ['train.bin', 'val.bin', 'manifest.json']:
        if (out_dir / name).exists():
            raise FileExistsError(f'{out_dir / name} exists; choose a new data version')
    if len(rows) < 2:
        raise ValueError('At least two rows are needed for a train/validation split')
    meta_path = ROOT / 'data/chess_v1/meta.pkl'
    with open(meta_path, 'rb') as stream:
        meta = pickle.load(stream)
    splits = rows.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
    splits['val'] = splits.pop('test')
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=out_dir) as temp:
        temp = Path(temp)
        db = sqlite3.connect(temp / 'hashes.sqlite')
        db.execute('CREATE TABLE hashes (hash BLOB PRIMARY KEY, split TEXT)')
        stats = {}
        for split, dataset in splits.items():
            info = dict(rows=0, characters=0, rows_with_internal_game_markers=0,
                        internal_game_markers=0, duplicate_rows_within_split=0)
            with open(temp / f'{split}.bin', 'wb') as stream:
                for index, item in enumerate(dataset):
                    text = item['transcript']
                    if not isinstance(text, str) or len(text) != row_size or not text.startswith(';'):
                        raise ValueError(f'{split} row {index}: expected {row_size} characters starting with ;')
                    try:
                        encoded = bytes(meta['stoi'][char] for char in text)
                    except KeyError as error:
                        raise ValueError(f'{split} row {index}: unknown character {error.args[0]!r}') from error
                    digest = hashlib.sha256(encoded).digest()
                    previous = db.execute('SELECT split FROM hashes WHERE hash=?', (digest,)).fetchone()
                    if previous and previous[0] != split:
                        raise ValueError(f'Exact row overlap between train and validation at {split} row {index}')
                    if previous:
                        info['duplicate_rows_within_split'] += 1
                    else:
                        db.execute('INSERT INTO hashes VALUES (?, ?)', (digest, split))
                    stream.write(encoded)
                    internal = text[1:].count(';')
                    info['rows'] += 1
                    info['characters'] += len(text)
                    info['rows_with_internal_game_markers'] += bool(internal)
                    info['internal_game_markers'] += internal
            info['sha256'] = sha256(temp / f'{split}.bin')
            stats[split] = info
        db.close()
        upstream = json.loads((ROOT / 'docs/upstream.json').read_text())
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        manifest = dict(format_version=1, dtype='uint8', row_size=row_size,
                        default_context_length=row_size - 1, vocabulary=meta['stoi'],
                        meta_sha256=sha256(meta_path), source=source, split_seed=seed,
                        validation_fraction=val_fraction, splits=stats, upstream=upstream,
                        preprocessing_commit=commit, preprocessing_dirty=dirty,
                        preparation_script_sha256=sha256(__file__),
                        boundary_policy='Rows are independent; preserve causal context across internal ; markers.',
                        overlap_check='No exact encoded row shared between splits; game-level separation is not asserted.')
        (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        shutil.copyfile(meta_path, temp / 'meta.pkl')
        # Manifest is published last and acts as the completion marker.
        for name in ['train.bin', 'val.bin', 'meta.pkl', 'manifest.json']:
            os.replace(temp / name, out_dir / name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='data/chess_v1')
    parser.add_argument('--dataset', default='adamkarvonen/chess_games')
    parser.add_argument('--file', default='lichess_6gb_blocks.zip')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--input', help='Local CSV or JSONL with a transcript column (offline tests)')
    parser.add_argument('--max-rows', type=int, help='Deterministic leading subset before the upstream shuffled split')
    parser.add_argument('--seed', type=int, default=2357)
    parser.add_argument('--val-fraction', type=float, default=0.01)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 2:
        parser.error('--max-rows must be at least 2')
    if args.input:
        file_path = args.input
        source = dict(local_file=str(Path(file_path).resolve()), sha256=sha256(file_path))
    else:
        revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
        file_path = hf_hub_download(args.dataset, args.file, repo_type='dataset', revision=revision)
        source = dict(dataset=args.dataset, file=args.file, revision=revision, sha256=sha256(file_path))
    # Explicit CSV loading replaces the old repository-script loader; row content is unchanged.
    loader = 'json' if str(file_path).endswith(('.jsonl', '.json')) else 'csv'
    rows = load_dataset(loader, data_files=file_path, split='train')
    if args.max_rows:
        rows = rows.select(range(min(args.max_rows, len(rows))))
    source['max_rows'] = args.max_rows
    result = prepare(rows, args.out_dir, source=source, seed=args.seed, val_fraction=args.val_fraction)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
