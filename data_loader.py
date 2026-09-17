"""Reference row-aligned sampling with storage stride independent of context length."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ChessData:
    def __init__(self, directory, context_length, verify_hashes=True):
        directory = Path(directory)
        self.manifest_hash = file_hash(directory / 'manifest.json')
        self.manifest = json.loads((directory / 'manifest.json').read_text())
        self.row_size = self.manifest['row_size']
        if not 1 <= context_length < self.row_size:
            raise ValueError('Context length must be positive and less than the stored row size')
        self.context_length = context_length
        if file_hash(directory / 'meta.pkl') != self.manifest['meta_sha256']:
            raise ValueError('Vocabulary hash differs from the manifest')
        with open(directory / 'meta.pkl', 'rb') as stream:
            self.meta = pickle.load(stream)
        self.rows = {}
        for split in ['train', 'val']:
            path = directory / f'{split}.bin'
            expected = self.manifest['splits'][split]
            if path.stat().st_size != expected['rows'] * self.row_size or not expected['rows']:
                raise ValueError(f'{split}: invalid row count or byte length')
            if verify_hashes and file_hash(path) != expected['sha256']:
                raise ValueError(f'{split}: file hash differs from the manifest')
            self.rows[split] = np.memmap(path, dtype=np.uint8, mode='r').reshape(-1, self.row_size)

    def batch(self, split, batch_size, device, generator):
        rows = self.rows[split]
        indices = torch.randint(len(rows), (batch_size,), generator=generator).numpy()
        block = np.array(rows[indices, :self.context_length + 1], dtype=np.int64)
        x, y = torch.from_numpy(block[:, :-1].copy()), torch.from_numpy(block[:, 1:].copy())
        if str(device).startswith('cuda'):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
