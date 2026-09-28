from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from model import GPT, GPTConfig

reference = import_module('experiments.evaluation_battery.karvonen_reference')


def _release(vocab_size=32):
    args = dict(n_layer=1, n_head=2, n_embd=16, block_size=8, bias=False, vocab_size=vocab_size, dropout=0.0)
    model = GPT(GPTConfig(**args))
    return model, dict(model={f'_orig_mod.{k}': v for k, v in model.state_dict().items()}, model_args=args,
                       iter_num=600000, best_val_loss=torch.tensor(0.2166), dataset='lichess_6gb_blocks.zip')


def test_adapt_keeps_weights_and_adds_battery_metadata():
    model, release = _release()
    data = SimpleNamespace(meta={'stoi': {str(i): i for i in range(32)}}, manifest_hash='manifest')
    adapted = reference.adapt(release, data, 'file.pt', 'abc')
    loaded = GPT(GPTConfig(**adapted['model_args']))
    loaded.load_state_dict(adapted['model'])
    assert all(torch.equal(loaded.state_dict()[k], v) for k, v in model.state_dict().items())
    assert adapted['config']['architecture'] == 'baseline' and adapted['manifest_hash'] == 'manifest'
    assert adapted['source'] == dict(file='file.pt', sha256='abc', best_val_loss=pytest.approx(0.2166),
                                     dataset='lichess_6gb_blocks.zip')


def test_adapt_refuses_a_different_vocabulary():
    _, release = _release(vocab_size=40)
    with pytest.raises(ValueError, match='vocabulary'):
        reference.adapt(release, SimpleNamespace(meta={'stoi': {}}, manifest_hash='m'), 'f', 's')
