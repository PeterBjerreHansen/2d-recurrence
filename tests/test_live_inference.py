import pytest
import torch
from contextlib import nullcontext
from unittest.mock import patch

from inference.live import (create_live_state, decode_live_step,
                            validate_live_inference_spec)
from inference.reference import create_reference_state, decode_reference_step
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import RecurrenceSchedule


def recurrent_config(mode='hybrid'):
    return RecurrentGPTConfig(n_layer=5, n_prelude=1, n_buffer=1, n_core=1,
                              n_source=1, n_coda=1, n_head=2, n_embd=8,
                              block_size=12, recurrence_mode=mode)


def test_live_spec_validates_modes_without_training_counts():
    baseline = validate_live_inference_spec(GPTConfig(n_layer=2, n_head=2, n_embd=8, block_size=12), None, None)
    assert (baseline.recurrence_mode, baseline.depth_steps, baseline.kv_strategy) == ('baseline', 1, 'ordinary')
    temporal = validate_live_inference_spec(recurrent_config('temporal'), None, None)
    assert (temporal.depth_steps, temporal.kv_strategy) == (1, 'final_depth')
    assert validate_live_inference_spec(recurrent_config('depth'), 4, 'depth_specialized').depth_steps == 4
    assert validate_live_inference_spec(recurrent_config('hybrid'), 1, 'final_depth').depth_steps == 1
    with pytest.raises(ValueError, match='depth_steps'):
        validate_live_inference_spec(recurrent_config('temporal'), 2, None)
    with pytest.raises(ValueError, match='explicit'):
        validate_live_inference_spec(recurrent_config('depth'), None, None)
    with pytest.raises(ValueError, match='kv_strategy'):
        validate_live_inference_spec(recurrent_config('depth'), 2, 'ordinary')


def test_baseline_cached_decoding_matches_full_prefix_logits():
    torch.manual_seed(83)
    model = GPT(GPTConfig(n_layer=3, n_head=2, n_embd=8, block_size=12)).eval()
    tokens = torch.randint(32, (2, 7))
    full_logits, _ = model(tokens, tokens)
    state = create_live_state(model)
    stepped = [decode_live_step(model, tokens[:, position], state) for position in range(tokens.shape[1])]
    stepped_logits = torch.stack(stepped, dim=1)
    torch.testing.assert_close(stepped_logits, full_logits, rtol=1e-5, atol=1e-6)
    assert state.position == tokens.shape[1]
    assert state.cache.cache_length == tokens.shape[1]


def test_live_context_limit_is_enforced():
    model = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=4, block_size=2)).eval()
    state = create_live_state(model)
    tokens = torch.tensor([1, 2, 3])
    decode_live_step(model, tokens[0:1], state)
    decode_live_step(model, tokens[1:2], state)
    with pytest.raises(ValueError, match='context'):
        decode_live_step(model, tokens[2:3], state)


def _live_model(mode, *, source=1, buffer=1):
    n_layer = 1 + buffer + 1 + source + 1
    return Recurrent2DGPT(RecurrentGPTConfig(
        n_layer=n_layer, n_prelude=1, n_buffer=buffer, n_core=1,
        n_source=source, n_coda=1, n_head=2, n_embd=8,
        block_size=12, recurrence_mode=mode)).eval()


@pytest.mark.parametrize('mode', ['temporal', 'depth', 'hybrid'])
@pytest.mark.parametrize('strategy, depth_steps', [
    ('final_depth', 1),
    ('final_depth', 2),
    ('depth_specialized', 2),
])
@pytest.mark.parametrize('source, buffer', [(1, 1), (0, 0)])
def test_cached_live_matches_slow_reference(mode, strategy, depth_steps, source, buffer):
    if mode == 'temporal' and depth_steps != 1:
        pytest.skip('Temporal-only live inference has exactly one depth step')
    torch.manual_seed(89)
    model = _live_model(mode, source=source, buffer=buffer)
    tokens = torch.randint(32, (1, 6))
    cached = create_live_state(model, depth_steps, strategy)
    reference = create_reference_state(model, depth_steps, strategy)
    for position in range(tokens.shape[1]):
        cached_logits = decode_live_step(model, tokens[:, position], cached)
        reference_logits = decode_reference_step(model, tokens[:, position], reference)
        torch.testing.assert_close(cached_logits, reference_logits, rtol=1e-4, atol=1e-5)
    assert cached.position == reference.position == tokens.shape[1]
    assert cached.cache.cache_length == tokens.shape[1]
    assert reference.cache.cache_length == tokens.shape[1]
    assert len(cached.cache.core) == (depth_steps if strategy == 'depth_specialized' else 1)


@pytest.mark.parametrize('strategy', ['final_depth', 'depth_specialized'])
def test_live_cache_commit_policy_is_explicit(strategy):
    torch.manual_seed(91)
    model = _live_model('hybrid')
    state = create_live_state(model, 3, strategy)
    decode_live_step(model, torch.tensor([4]), state)
    assert all(cache.length == 1 for cache in state.cache._streams())
    if strategy == 'final_depth':
        assert len(state.cache.core) == 1
    else:
        assert len(state.cache.core) == 3
    assert all(cache.length == 1 for depth in state.cache.core for cache in depth)


def test_cache_strategies_are_identical_at_one_depth():
    torch.manual_seed(92)
    model = _live_model('hybrid')
    tokens = torch.tensor([3, 4, 5, 6])
    final = create_live_state(model, 1, 'final_depth')
    specialized = create_live_state(model, 1, 'depth_specialized')
    for token in tokens:
        final_logits = decode_live_step(model, token[None], final)
        specialized_logits = decode_live_step(model, token[None], specialized)
        torch.testing.assert_close(final_logits, specialized_logits, rtol=1e-5, atol=1e-6)


def test_live_source_and_coda_are_outside_the_depth_loop():
    torch.manual_seed(93)
    model = _live_model('hybrid')
    with patch.object(model.transformer.h[model.config.source_start], 'forward_step',
                      wraps=model.transformer.h[model.config.source_start].forward_step) as source_step, \
            patch.object(model.transformer.h[model.config.coda_start], 'forward_step',
                         wraps=model.transformer.h[model.config.coda_start].forward_step) as coda_step, \
            patch.object(model.depth_mixer, 'forward', wraps=model.depth_mixer.forward) as depth_step:
        state = create_live_state(model, 4, 'final_depth')
        for token in (2, 3, 5):
            decode_live_step(model, torch.tensor([token]), state)
    assert source_step.call_count == coda_step.call_count == 3
    assert depth_step.call_count == 9


@pytest.mark.parametrize('mode, depth_steps, expected_temporal, expected_depth', [
    ('temporal', 1, 1, 0),
    ('depth', 3, 0, 4),
    ('hybrid', 3, 1, 4),
])
def test_mode_call_count_invariants(mode, depth_steps, expected_temporal, expected_depth):
    torch.manual_seed(94)
    model = _live_model(mode)
    temporal_context = (patch.object(model.temporal_mixer, 'forward_step',
                                     wraps=model.temporal_mixer.forward_step)
                       if model.temporal_mixer is not None else nullcontext())
    depth_context = (patch.object(model.depth_mixer, 'forward', wraps=model.depth_mixer.forward)
                     if model.depth_mixer is not None else nullcontext())
    with temporal_context as temporal_step, depth_context as depth_step:
        state = create_live_state(model, depth_steps,
                                  'final_depth' if mode != 'temporal' else 'ordinary')
        decode_live_step(model, torch.tensor([2]), state)
        decode_live_step(model, torch.tensor([3]), state)
    assert (temporal_step.call_count if temporal_step is not None else 0) == expected_temporal
    assert (depth_step.call_count if depth_step is not None else 0) == expected_depth


def test_live_state_rejects_a_different_cache_semantics():
    model = _live_model('depth')
    state = create_live_state(model, 2, 'final_depth')
    with pytest.raises(ValueError, match='differs'):
        decode_live_step(model, torch.tensor([2]), state,
                         validate_live_inference_spec(model.config, 2, 'depth_specialized'))


def test_live_state_rejects_a_spec_for_a_different_mode():
    model = _live_model('hybrid')
    state = create_live_state(model, 2, 'final_depth')
    with pytest.raises(ValueError, match='model'):
        decode_live_step(model, torch.tensor([2]), state,
                         validate_live_inference_spec(
                             RecurrentGPTConfig(n_layer=5, n_prelude=1, n_buffer=1, n_core=1,
                                                n_source=1, n_coda=1, n_head=2, n_embd=8,
                                                block_size=12, recurrence_mode='depth'),
                             2, 'final_depth'))


def test_live_state_rejects_a_different_model_layout():
    model = _live_model('depth')
    other_model = Recurrent2DGPT(RecurrentGPTConfig(
        n_layer=4, n_prelude=1, n_buffer=0, n_core=1, n_source=1, n_coda=1,
        n_head=2, n_embd=8, block_size=12, recurrence_mode='depth')).eval()
    state = create_live_state(model, 2, 'final_depth')
    with pytest.raises(ValueError, match='model layout'):
        decode_live_step(other_model, torch.tensor([2]), state)


def test_depth_specialized_live_matches_depth_only_training_graph():
    torch.manual_seed(97)
    model = _live_model('depth')
    tokens = torch.randint(32, (1, 7))
    for depth_steps in (1, 2, 4):
        schedule = RecurrenceSchedule((False,) * (depth_steps - 1),
                                      (True,) * (depth_steps - 1))
        training_logits, _ = model(tokens, tokens, schedule=schedule)
        state = create_live_state(model, depth_steps, 'depth_specialized')
        live_logits = [decode_live_step(model, tokens[:, position], state)
                       for position in range(tokens.shape[1])]
        live_logits = torch.stack(live_logits, dim=1)
        torch.testing.assert_close(live_logits, training_logits, rtol=1e-4, atol=1e-5)
        assert state.temporal_memory is None
