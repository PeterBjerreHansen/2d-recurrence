"""Fixed-depth live inference for baseline and recurrence-mode checkpoints."""

from dataclasses import dataclass

import torch

from inference.cache import ModelKVCache, create_model_kv_cache, model_cache_signature
from models.recurrent_2d import Recurrent2DGPT


LIVE_STRATEGIES = frozenset({'ordinary', 'final_depth', 'depth_specialized'})


def _mode(config):
    return getattr(config, 'recurrence_mode', 'baseline')


@dataclass(frozen=True)
class LiveInferenceSpec:
    recurrence_mode: str
    depth_steps: int
    kv_strategy: str


def validate_live_inference_spec(config, depth_steps=None, kv_strategy=None):
    """Validate and normalize fixed-depth live execution semantics."""
    mode = _mode(config)
    if depth_steps is not None and (type(depth_steps) is not int or depth_steps < 1):
        raise ValueError('depth_steps must be a positive integer')
    if mode == 'baseline':
        if depth_steps is None:
            depth_steps = 1
        if depth_steps != 1:
            raise ValueError('Baseline live inference requires depth_steps=1')
        if kv_strategy is None:
            kv_strategy = 'ordinary'
        if kv_strategy != 'ordinary':
            raise ValueError("Baseline live inference requires kv_strategy='ordinary'")
        return LiveInferenceSpec(mode, depth_steps, kv_strategy)
    if mode not in {'temporal', 'depth', 'hybrid'}:
        raise ValueError(f'Unsupported recurrence_mode for live inference: {mode}')
    if depth_steps is None:
        if mode == 'temporal':
            depth_steps = 1
        else:
            raise ValueError(f'{mode} live inference requires explicit depth_steps')
    if mode == 'temporal' and depth_steps != 1:
        raise ValueError('Temporal-only live inference requires depth_steps=1')
    if kv_strategy is None:
        kv_strategy = 'final_depth'
    if kv_strategy not in LIVE_STRATEGIES:
        raise ValueError("kv_strategy must be 'final_depth' or 'depth_specialized'")
    if mode in {'depth', 'hybrid'} and kv_strategy == 'ordinary':
        raise ValueError("Depth-recurrent live inference requires kv_strategy='final_depth' or 'depth_specialized'")
    return LiveInferenceSpec(mode, depth_steps, kv_strategy)


@dataclass
class LiveInferenceState:
    position: int
    cache: ModelKVCache
    temporal_memory: torch.Tensor | None = None


def create_live_state(model, depth_steps=None, kv_strategy=None, *, batch_size=1):
    spec = validate_live_inference_spec(model.config, depth_steps, kv_strategy)
    cache = create_model_kv_cache(model.config, mode=spec.recurrence_mode,
                                  depth_steps=spec.depth_steps, strategy=spec.kv_strategy)
    # batch_size is part of the construction interface even though dynamic
    # caches learn the exact batch shape from the first committed token.
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    return LiveInferenceState(position=0, cache=cache)


def _token_column(token):
    if token.ndim == 1:
        return token[:, None]
    if token.ndim == 2 and token.shape[1] == 1:
        return token
    raise ValueError('decode_live_step expects [batch] or [batch, 1] token IDs')


def _run_blocks_step(blocks, value, caches, *, commit=True):
    for block, cache in zip(blocks, caches):
        value = block.forward_step(value, cache, commit=commit)
    return value


@torch.no_grad()
def decode_live_step(model, token, state, spec=None):
    """Consume one physical token and return logits for the next token."""
    token = _token_column(token)
    if state.position >= model.config.block_size:
        raise ValueError('Live inference context limit reached')
    if state.cache.model_signature != model_cache_signature(model.config):
        raise ValueError('Live cache model layout differs from the model')
    if spec is not None:
        expected = validate_live_inference_spec(model.config, spec.depth_steps, spec.kv_strategy)
        if spec != expected:
            raise ValueError('Live inference spec differs from the model')
        if expected != LiveInferenceSpec(state.cache.mode, state.cache.depth_steps, state.cache.strategy):
            raise ValueError('Live inference spec differs from the existing state')
    if state.cache.mode == 'baseline':
        value = model.embed_step(token, state.position)
        value = _run_blocks_step(model.transformer.h, value, state.cache.baseline)
        logits, _ = model.readout(value)
        state.position += 1
        return logits[:, 0]
    if not isinstance(model, Recurrent2DGPT):
        raise TypeError('Recurrent live state requires a Recurrent2DGPT model')
    config = model.config
    value = model.embed_step(token, state.position)
    value = _run_blocks_step(model.transformer.h[:config.n_prelude], value, state.cache.prelude)
    if state.temporal_memory is None:
        anchor = value
    else:
        if model.temporal_mixer is None:
            raise RuntimeError('Live state contains temporal memory but the model has no temporal mixer')
        anchor = model.temporal_mixer.forward_step(value, state.temporal_memory)
    anchor = _run_blocks_step(
        model.transformer.h[config.n_prelude:config.core_start], anchor, state.cache.buffer)
    h = anchor
    for depth in range(state.cache.depth_steps):
        if depth:
            if model.depth_mixer is None:
                raise RuntimeError('Live depth loop requires a depth mixer')
            h = model.depth_mixer(h, anchor)
        h = _run_blocks_step(
            model.transformer.h[config.core_start:config.core_stop], h,
            state.cache.core_for_depth(depth),
            commit=(state.cache.strategy == 'depth_specialized' or depth == state.cache.depth_steps - 1))
    source = _run_blocks_step(
        model.transformer.h[config.source_start:config.source_stop], h, state.cache.source)
    if config.uses_temporal_recurrence:
        state.temporal_memory = source.detach()
    else:
        state.temporal_memory = None
    output = _run_blocks_step(
        model.transformer.h[config.coda_start:config.n_layer], source, state.cache.coda)
    logits, _ = model.readout(output)
    state.position += 1
    return logits[:, 0]


def consume_prompt_live(model, prompt_tokens, state, spec=None):
    """Consume prompt IDs sequentially and return the next-token logits."""
    if prompt_tokens.ndim != 1:
        raise ValueError('prompt_tokens must be a one-dimensional token sequence')
    logits = None
    for token in prompt_tokens:
        logits = decode_live_step(model, token[None], state, spec)
    return logits
