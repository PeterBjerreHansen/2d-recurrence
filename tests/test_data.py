import importlib
import json
from pathlib import Path

from datasets import Dataset
import numpy as np
import pytest
import torch

from data_loader import ChessData


prepare_module = importlib.import_module('data.chess_v1.prepare')
prepare = prepare_module.prepare


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


def test_prepare_records_unavailable_git_provenance_in_clean_transfer_bundle(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise prepare_module.subprocess.CalledProcessError(
            128, ['git', 'rev-parse'], stderr='not a git repository')

    monkeypatch.setattr(prepare_module.subprocess, 'check_output', no_git)
    rows = Dataset.from_dict({'transcript': [';1.e4 e5', ';1.d4 d5']})

    manifest = prepare(rows, tmp_path / 'prepared', source={'fixture': 'bundle test'},
                       val_fraction=0.5, row_size=8)

    assert manifest['preprocessing_commit'] is None
    assert manifest['preprocessing_dirty'] is None
    assert manifest['preparation_script_sha256']
