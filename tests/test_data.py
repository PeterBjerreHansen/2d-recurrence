import json
from pathlib import Path

from datasets import Dataset
import numpy as np
import pytest
import torch

from data.chess_v1.prepare import prepare
from data_loader import ChessData


def test_sampling_stays_on_storage_rows_with_short_context(prepared_data):
    data = ChessData(prepared_data, 23)
    generator = torch.Generator().manual_seed(7)
    indices = torch.randint(len(data.rows['train']), (10,), generator=generator)
    x, y = data.batch('train', 10, 'cpu', torch.Generator().manual_seed(7))
    expected = torch.tensor(np.array(data.rows['train'][indices.numpy(), :24], dtype=np.int64))
    assert torch.equal(x, expected[:, :-1])
    assert torch.equal(y, expected[:, 1:])
    assert (x[:, 0] == data.meta['stoi'][';']).all()


def test_exact_split_overlap_is_rejected_without_publishing(tmp_path):
    row = ';1.e4 e5 ' + ' ' * (1024 - len(';1.e4 e5 '))
    with pytest.raises(ValueError, match='overlap'):
        prepare(Dataset.from_dict({'transcript': [row] * 10}), tmp_path / 'bad', source={}, val_fraction=0.2)
    assert not (tmp_path / 'bad' / 'manifest.json').exists()
    assert not (tmp_path / 'bad' / 'train.bin').exists()


def test_unknown_characters_and_bad_lengths_rejected(tmp_path):
    for index, row in enumerate([';x', ';' + '!' * 1023]):
        with pytest.raises(ValueError):
            prepare(Dataset.from_dict({'transcript': [row] * 3}), tmp_path / str(index), source={})


def test_manifest_protects_against_changed_data(prepared_data):
    path = prepared_data / 'train.bin'
    with open(path, 'r+b') as stream:
        stream.seek(12)
        stream.write(b'\x00')
    with pytest.raises(ValueError, match='hash'):
        ChessData(prepared_data, 16)


def test_prepared_version_is_not_overwritten(prepared_data):
    with pytest.raises(FileExistsError):
        prepare(Dataset.from_dict({'transcript': []}), prepared_data, source={})
