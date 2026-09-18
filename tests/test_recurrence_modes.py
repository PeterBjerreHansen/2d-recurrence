import random

import pytest
import torch

from evaluation.feedback_diagnostic import require_temporal_feedback
from evaluation.recurrence_grid import supported_evaluation_cells
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


@pytest.mark.parametrize('mode, probabilities, eval_counts', [
    ('temporal', [[.5, .5], [0., 0.]], (0, 1)),
    ('depth', [[.5, 0.], [.5, 0.]], (1, 0)),
])
def test_training_rejects_probability_mass_on_inactive_axis(mode, probabilities, eval_counts):
    with pytest.raises(ValueError, match='recurrence_mode'):
        train(dict(architecture='recurrent', recurrence_mode=mode,
                   recurrence_support=[0, 1], recurrence_probabilities=probabilities,
                   eval_u_t=eval_counts[0], eval_u_d=eval_counts[1], compile=False))
