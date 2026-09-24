import json

import torch
import pytest

from train import DEFAULTS, get_lr, train


def wsd_config(**overrides):
    config = {
        **DEFAULTS,
        'lr_schedule': 'wsd',
        'learning_rate': 3e-4,
        'min_lr': 3e-5,
        'warmup_iters': 2,
        'lr_decay_start': 5,
        'lr_decay_iters': 9,
    }
    config.update(overrides)
    return config


def test_wsd_uses_linear_warmup_stable_peak_and_linear_cooldown():
    config = wsd_config()

    assert get_lr(0, config) == 0.0
    assert get_lr(1, config) == 1.5e-4
    assert get_lr(2, config) == 3e-4
    assert get_lr(4, config) == 3e-4
    assert get_lr(5, config) == 3e-4
    assert get_lr(6, config) == pytest.approx(2.1e-4)
    assert get_lr(7, config) == pytest.approx(1.2e-4)
    assert get_lr(8, config) == 3e-5
    assert get_lr(9, config) == 3e-5
    assert get_lr(10, config) == 3e-5


def test_explicit_constant_and_legacy_decay_lr_keep_their_old_meaning():
    explicit = wsd_config(lr_schedule='constant', decay_lr=True)
    legacy_constant = {**DEFAULTS, 'decay_lr': False, 'learning_rate': 3e-4}

    assert get_lr(0, explicit) == get_lr(100, explicit) == 3e-4
    assert get_lr(0, legacy_constant) == get_lr(100, legacy_constant) == 3e-4


def test_unspecified_schedule_preserves_the_historical_cosine_curve():
    config = {
        **DEFAULTS,
        'learning_rate': 3e-4,
        'min_lr': 3e-5,
        'decay_lr': True,
        'warmup_iters': 2,
        'lr_decay_iters': 6,
    }

    assert get_lr(0, config) == 0.0
    assert get_lr(2, config) == 3e-4
    assert get_lr(3, config) == pytest.approx(
        3e-5 + 0.5 * (1 + 2 ** -0.5) * (3e-4 - 3e-5))
    assert get_lr(6, config) == 3e-5


def test_wsd_requires_a_valid_cooldown_boundary():
    from train import train

    with pytest.raises(ValueError, match='WSD requires'):
        train({'lr_schedule': 'wsd', 'warmup_iters': 2, 'lr_decay_start': None,
               'lr_decay_iters': 6})


def test_wsd_exact_resume_across_warmup_stable_and_decay(prepared_data, tmp_path):
    config = {
        'architecture': 'baseline',
        'dataset': str(prepared_data),
        'n_layer': 2,
        'n_head': 2,
        'n_embd': 16,
        'block_size': 16,
        'batch_size': 1,
        'gradient_accumulation_steps': 1,
        'dropout': 0.2,
        'learning_rate': 3e-4,
        'min_lr': 3e-5,
        'lr_schedule': 'wsd',
        'warmup_iters': 2,
        'lr_decay_start': 4,
        'lr_decay_iters': 6,
        'max_iters': 6,
        'eval_interval': 6,
        'eval_iters': 1,
        'log_interval': 1,
        'compile': False,
        'device': 'cpu',
        'dtype': 'float32',
        'num_threads': 1,
        'seed': 23,
    }
    uninterrupted_path = train({**config, 'out_dir': str(tmp_path / 'uninterrupted')})

    resumed_dir = str(tmp_path / 'resumed')
    for limit in (1, 3, 5, 6):
        train({**config, 'out_dir': resumed_dir, 'max_iters': limit,
               'init_from': 'resume' if limit != 1 else 'scratch'})

    uninterrupted = torch.load(uninterrupted_path, map_location='cpu', weights_only=False)
    resumed = torch.load(tmp_path / 'resumed' / 'ckpt.pt', map_location='cpu', weights_only=False)
    assert uninterrupted['iter_num'] == resumed['iter_num'] == 6
    assert uninterrupted['config']['lr_schedule'] == resumed['config']['lr_schedule'] == 'wsd'
    for name, value in uninterrupted['model'].items():
        torch.testing.assert_close(value, resumed['model'][name], rtol=0, atol=0)
    torch.testing.assert_close(uninterrupted['optimizer'], resumed['optimizer'], rtol=0, atol=0)
    train_events = [json.loads(line) for line in (tmp_path / 'uninterrupted' / 'metrics.jsonl').read_text().splitlines()
                    if json.loads(line)['event'] == 'train']
    assert train_events[-1]['step'] == 6
    assert train_events[-1]['lr'] == 3e-5
