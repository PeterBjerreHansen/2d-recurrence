import random

import pytest
import torch

from evaluation.feedback_diagnostic import require_temporal_feedback
from evaluation.recurrence_grid import evaluate_grid, supported_evaluation_cells
from data_loader import ChessData
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import sample_schedule
from sample import effective_generation_counts
from train import train


def tiny_config(**overrides):
    values = dict(n_layer=4, n_prelude=1, n_buffer=0, n_core=1,
                  n_source=1, n_coda=1, n_head=2, n_embd=8, block_size=8)
    values.update(overrides)
    return RecurrentGPTConfig(**values)


def test_recurrence_modes_and_legacy_default():
    assert RecurrentGPTConfig().recurrence_mode == 'hybrid'
    for mode in ('hybrid', 'temporal', 'depth'):
        config = tiny_config(recurrence_mode=mode)
        assert config.recurrence_mode == mode
    with pytest.raises(ValueError, match='recurrence_mode'):
        tiny_config(recurrence_mode='invalid')
    legacy = RecurrentGPTConfig.from_checkpoint(dict(n_layer=4, n_prelude=1,
                                                     n_core=1, n_coda=1, n_embd=8,
                                                     n_head=2, block_size=8))
    assert legacy.recurrence_mode == 'hybrid'


def test_recurrence_mode_capabilities():
    assert tiny_config(recurrence_mode='hybrid').uses_temporal_recurrence
    assert tiny_config(recurrence_mode='hybrid').uses_depth_recurrence
    assert tiny_config(recurrence_mode='temporal').uses_temporal_recurrence
    assert not tiny_config(recurrence_mode='temporal').uses_depth_recurrence
    assert not tiny_config(recurrence_mode='depth').uses_temporal_recurrence
    assert tiny_config(recurrence_mode='depth').uses_depth_recurrence


def test_default_and_explicit_hybrid_are_identical():
    torch.manual_seed(31)
    implicit = Recurrent2DGPT(tiny_config())
    torch.manual_seed(31)
    explicit = Recurrent2DGPT(tiny_config(recurrence_mode='hybrid'))
    assert list(implicit.state_dict()) == list(explicit.state_dict())
    for name, parameter in implicit.state_dict().items():
        torch.testing.assert_close(parameter, explicit.state_dict()[name], rtol=0, atol=0)
    x = torch.randint(32, (2, 8))
    schedule = sample_schedule(1, 3, random.Random(7))
    implicit_logits, implicit_loss = implicit(x, x, schedule=schedule)
    explicit_logits, explicit_loss = explicit(x, x, schedule=schedule)
    torch.testing.assert_close(implicit_logits, explicit_logits, rtol=0, atol=0)
    torch.testing.assert_close(implicit_loss, explicit_loss, rtol=0, atol=0)


@pytest.mark.parametrize('mode, expected', [
    ('hybrid', [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3), (3, 0), (3, 1), (3, 3)]),
    ('temporal', [(0, 0), (1, 0), (3, 0)]),
    ('depth', [(0, 0), (0, 1), (0, 3)]),
])
def test_mode_aware_evaluation_domains(mode, expected):
    assert supported_evaluation_cells(tiny_config(recurrence_mode=mode)) == expected


@pytest.mark.parametrize('mode, expected', [
    ('hybrid', (3, 3)),
    ('temporal', (3, 0)),
    ('depth', (0, 3)),
])
def test_mode_aware_generation_defaults(mode, expected):
    assert effective_generation_counts(mode) == expected


def test_generation_defaults_preserve_explicit_other_axis():
    assert effective_generation_counts('temporal', u_t=1) == (1, 0)
    assert effective_generation_counts('depth', u_d=1) == (0, 1)
    assert effective_generation_counts('hybrid', u_t=1) == (1, 3)
    assert effective_generation_counts('hybrid', u_d=1) == (3, 1)


def test_feedback_diagnostic_requires_temporal_recurrence():
    require_temporal_feedback(tiny_config(recurrence_mode='hybrid'))
    require_temporal_feedback(tiny_config(recurrence_mode='temporal'))
    with pytest.raises(ValueError, match='requires temporal feedback'):
        require_temporal_feedback(tiny_config(recurrence_mode='depth'))


@pytest.mark.parametrize('mode, schedule_counts, inactive_prefix', [
    ('temporal', (1, 0), 'depth_mixer.'),
    ('depth', (0, 1), 'temporal_mixer.'),
])
def test_specialized_models_omit_inactive_mixer_and_reject_incompatible_schedule(
        mode, schedule_counts, inactive_prefix):
    model = Recurrent2DGPT(tiny_config(recurrence_mode=mode))
    names = list(model.state_dict())
    assert not any(name.startswith(inactive_prefix) for name in names)
    active_prefix = 'temporal_mixer.' if mode == 'temporal' else 'depth_mixer.'
    assert any(name.startswith(active_prefix) for name in names)
    x = torch.randint(32, (1, 8))
    with pytest.raises(ValueError, match='recurrence_mode'):
        model(x, x, schedule=sample_schedule(*(schedule_counts[::-1]), random.Random(2)))


def _copy_common_state(source, target):
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            parameter.copy_(dict(source.named_parameters())[name])


@pytest.mark.parametrize('mode, counts, inactive_name', [
    ('temporal', (1, 0), 'depth_mixer'),
    ('depth', (0, 1), 'temporal_mixer'),
])
@pytest.mark.parametrize('updates', [1, 3])
def test_specialization_matches_hybrid_limiting_case(mode, counts, inactive_name, updates):
    torch.manual_seed(41)
    hybrid = Recurrent2DGPT(tiny_config(recurrence_mode='hybrid'))
    specialized = Recurrent2DGPT(tiny_config(recurrence_mode=mode))
    _copy_common_state(hybrid, specialized)
    x = torch.randint(32, (2, 8))
    # Use exact all-axis schedules so K has a deterministic active-axis write path.
    if mode == 'temporal':
        schedule = sample_schedule(updates, 0, random.Random(updates))
    else:
        schedule = sample_schedule(0, updates, random.Random(updates))
    hybrid_logits, hybrid_loss = hybrid(x, x, schedule=schedule)
    specialized_logits, specialized_loss = specialized(x, x, schedule=schedule)
    torch.testing.assert_close(hybrid_logits, specialized_logits, rtol=0, atol=0)
    torch.testing.assert_close(hybrid_loss, specialized_loss, rtol=0, atol=0)
    hybrid_loss.backward()
    specialized_loss.backward()
    specialized_parameters = dict(specialized.named_parameters())
    for name, parameter in hybrid.named_parameters():
        if name in specialized_parameters:
            torch.testing.assert_close(parameter.grad, specialized_parameters[name].grad,
                                       rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in getattr(hybrid, inactive_name).parameters())


@pytest.mark.parametrize('mode', ['hybrid', 'temporal', 'depth'])
def test_all_modes_zero_update_match_baseline(mode):
    torch.manual_seed(53)
    base = GPT(GPTConfig(n_layer=4, n_head=2, n_embd=8, block_size=8))
    model = Recurrent2DGPT(tiny_config(recurrence_mode=mode))
    model.transformer.load_state_dict(base.transformer.state_dict())
    x = torch.randint(32, (2, 8))
    schedule = sample_schedule(0, 0, random.Random(0))
    base_logits, base_loss = base(x, x)
    model_logits, model_loss = model(x, x, schedule=schedule)
    torch.testing.assert_close(model_logits, base_logits, rtol=0, atol=0)
    torch.testing.assert_close(model_loss, base_loss, rtol=0, atol=0)
    base_loss.backward()
    model_loss.backward()
    model_parameters = dict(model.named_parameters())
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(model_parameters[name].grad, parameter.grad, rtol=0, atol=0)


def test_specialized_parameter_counts_are_structural():
    temporal = Recurrent2DGPT(tiny_config(recurrence_mode='temporal'))
    depth = Recurrent2DGPT(tiny_config(recurrence_mode='depth'))
    hybrid = Recurrent2DGPT(tiny_config(recurrence_mode='hybrid'))
    temporal_mixer_count = sum(parameter.numel() for parameter in temporal.temporal_mixer.parameters())
    depth_mixer_count = sum(parameter.numel() for parameter in depth.depth_mixer.parameters())
    temporal_count = sum(parameter.numel() for parameter in temporal.parameters())
    depth_count = sum(parameter.numel() for parameter in depth.parameters())
    hybrid_count = sum(parameter.numel() for parameter in hybrid.parameters())
    assert temporal_count > temporal_count - temporal_mixer_count
    assert depth_count > depth_count - depth_mixer_count
    assert hybrid_count == temporal_count + depth_mixer_count
    assert hybrid_count == depth_count + temporal_mixer_count


@pytest.mark.parametrize('mode', ['temporal', 'depth', 'hybrid'])
def test_mode_aware_grid_evaluates_only_supported_cells(mode, prepared_data):
    model = Recurrent2DGPT(tiny_config(recurrence_mode=mode))
    data = ChessData(prepared_data, 8)
    report = evaluate_grid(model, data, batches=1, batch_size=1, diagnostics=False)
    assert [(cell['u_t'], cell['u_d']) for cell in report['cells']] == supported_evaluation_cells(model.config)


def mode_training_config(mode, output, **overrides):
    distributions = {
        'temporal': [[.4, 0.], [.6, 0.]],
        'depth': [[.4, .6], [0., 0.]],
        'hybrid': [[.4, 0.], [0., .6]],
    }
    evaluation = {'temporal': (1, 0), 'depth': (0, 1), 'hybrid': (1, 1)}[mode]
    config = dict(architecture='recurrent', recurrence_mode=mode,
                  recurrence_support=[0, 1], recurrence_probabilities=distributions[mode],
                  recurrence_seed=19, n_layer=4, n_prelude=1, n_buffer=0, n_core=1,
                  n_source=1, n_coda=1, n_head=2, n_embd=8, dataset='', block_size=8,
                  batch_size=1, gradient_accumulation_steps=1, max_iters=2,
                  eval_interval=1, eval_iters=1, eval_u_t=evaluation[0], eval_u_d=evaluation[1],
                  log_interval=1, warmup_iters=0, lr_decay_iters=2, compile=False,
                  device='cpu', dtype='float32', dropout=.2, num_threads=1,
                  out_dir=str(output))
    config.update(overrides)
    return config


def test_historical_checkpoint_without_mode_loads_strictly():
    torch.manual_seed(61)
    config = tiny_config(recurrence_mode='hybrid')
    source = Recurrent2DGPT(config)
    model_args = dict(config.__dict__)
    model_args.pop('recurrence_mode')
    loaded = Recurrent2DGPT(RecurrentGPTConfig.from_checkpoint(model_args))
    loaded.load_state_dict(source.state_dict())
    assert loaded.config.recurrence_mode == 'hybrid'
    assert set(loaded.state_dict()) == set(source.state_dict())


@pytest.mark.parametrize('mode', ['temporal', 'depth'])
def test_specialized_training_resume_is_exact(mode, prepared_data, tmp_path):
    config = mode_training_config(mode, tmp_path / 'full', dataset=str(prepared_data))
    full_path = train(config)
    split_dir = tmp_path / 'split'
    train({**config, 'out_dir': str(split_dir), 'max_iters': 1})
    resumed_path = train({**config, 'out_dir': str(split_dir), 'max_iters': 2, 'init_from': 'resume'})
    full = torch.load(full_path, weights_only=False)
    resumed = torch.load(resumed_path, weights_only=False)
    assert full['model_args']['recurrence_mode'] == resumed['model_args']['recurrence_mode'] == mode
    assert not any(name.startswith('depth_mixer.') for name in full['model']) if mode == 'temporal' else not any(name.startswith('temporal_mixer.') for name in full['model'])
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
    torch.testing.assert_close(full['optimizer'], resumed['optimizer'], rtol=0, atol=0)
    assert full['recurrence_sampler'] == resumed['recurrence_sampler']


@pytest.mark.parametrize('source, target', [
    ('temporal', 'depth'), ('temporal', 'hybrid'),
    ('depth', 'temporal'), ('depth', 'hybrid'),
    ('hybrid', 'temporal'), ('hybrid', 'depth'),
])
def test_cross_mode_resume_is_rejected(source, target, prepared_data, tmp_path):
    source_dir = tmp_path / 'source'
    train(mode_training_config(source, source_dir, dataset=str(prepared_data), max_iters=0))
    with pytest.raises(ValueError, match='Resume model configuration differs'):
        train(mode_training_config(target, source_dir, dataset=str(prepared_data),
                                   max_iters=1, init_from='resume'))


@pytest.mark.parametrize('mode, probabilities, eval_counts', [
    ('temporal', [[.5, .5], [0., 0.]], (0, 1)),
    ('depth', [[.5, 0.], [.5, 0.]], (1, 0)),
])
def test_training_rejects_probability_mass_on_inactive_axis(mode, probabilities, eval_counts):
    with pytest.raises(ValueError, match='recurrence_mode'):
        train(dict(architecture='recurrent', recurrence_mode=mode,
                   recurrence_support=[0, 1], recurrence_probabilities=probabilities,
                   eval_u_t=eval_counts[0], eval_u_d=eval_counts[1], compile=False))
