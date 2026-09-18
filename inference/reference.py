"""Slow full-sequence live oracle independent of incremental KV attention."""

from dataclasses import dataclass
from itertools import chain

import torch

from inference.live import LiveInferenceSpec, validate_live_inference_spec
from models.recurrent_2d import Recurrent2DGPT


@dataclass
class LayerInputHistory:
    inputs: torch.Tensor | None = None

    @property
    def length(self):
        return 0 if self.inputs is None else self.inputs.shape[1]

    def candidate(self, current):
        return current if self.inputs is None else torch.cat((self.inputs, current), dim=1)

    def append(self, current):
        self.inputs = current.detach() if self.inputs is None else torch.cat((self.inputs, current.detach()), dim=1)


@dataclass
class ReferenceCache:
    mode: str
    strategy: str
    depth_steps: int
    baseline: list[LayerInputHistory]
    prelude: list[LayerInputHistory]
    buffer: list[LayerInputHistory]
    core: list[list[LayerInputHistory]]
    source: list[LayerInputHistory]
    coda: list[LayerInputHistory]

    def core_for_depth(self, depth):
        return self.core[depth] if self.strategy == 'depth_specialized' else self.core[0]

    def _streams(self):
        if self.mode == 'baseline':
            return list(self.baseline)
        return list(chain(self.prelude, self.buffer, *(self.core), self.source, self.coda))

    @property
    def cache_length(self):
        lengths = {history.length for history in self._streams()}
        if len(lengths) > 1:
            raise RuntimeError(f'Reference histories have inconsistent lengths: {sorted(lengths)}')
        return lengths.pop() if lengths else 0


def _history_streams(count):
    return [LayerInputHistory() for _ in range(count)]


def create_reference_cache(config, *, mode, depth_steps, strategy):
    if mode == 'baseline':
        return ReferenceCache(mode, strategy, depth_steps, _history_streams(config.n_layer),
                              [], [], [], [], [])
    core_depths = depth_steps if strategy == 'depth_specialized' else 1
    return ReferenceCache(
        mode, strategy, depth_steps, [],
        _history_streams(config.n_prelude),
        _history_streams(config.n_buffer),
        [_history_streams(config.n_core) for _ in range(core_depths)],
        _history_streams(config.n_source),
        _history_streams(config.n_coda),
    )


@dataclass
class ReferenceInferenceState:
    position: int
    cache: ReferenceCache
    temporal_memory: torch.Tensor | None = None


def create_reference_state(model, depth_steps=None, kv_strategy=None):
    spec = validate_live_inference_spec(model.config, depth_steps, kv_strategy)
    return ReferenceInferenceState(
        position=0,
        cache=create_reference_cache(model.config, mode=spec.recurrence_mode,
                                     depth_steps=spec.depth_steps, strategy=spec.kv_strategy),
    )


def _run_block_step(block, value, history, *, commit=True):
    current_input = value
    output = block(history.candidate(current_input))[:, -1:]
    if commit:
        history.append(current_input)
    return output


def _run_blocks_step(blocks, value, histories, *, commit=True):
    for block, history in zip(blocks, histories):
        value = _run_block_step(block, value, history, commit=commit)
    return value


@torch.no_grad()
def decode_reference_step(model, token, state, spec=None):
    """Consume one token through full-sequence blocks without KV attention."""
    if token.ndim == 1:
        token = token[:, None]
    if token.ndim != 2 or token.shape[1] != 1:
        raise ValueError('decode_reference_step expects [batch] or [batch, 1] token IDs')
    if state.position >= model.config.block_size:
        raise ValueError('Reference live inference context limit reached')
    if spec is not None:
        expected = validate_live_inference_spec(model.config, spec.depth_steps, spec.kv_strategy)
        if spec != expected:
            raise ValueError('Live inference spec differs from the model')
        if expected != LiveInferenceSpec(state.cache.mode, state.cache.depth_steps, state.cache.strategy):
            raise ValueError('Live inference spec differs from the existing reference state')
    if state.cache.mode == 'baseline':
        value = model.embed_step(token, state.position)
        value = _run_blocks_step(model.transformer.h, value, state.cache.baseline)
        logits, _ = model.readout(value)
        state.position += 1
        return logits[:, 0]
    if not isinstance(model, Recurrent2DGPT):
        raise TypeError('Recurrent reference state requires a Recurrent2DGPT model')
    config = model.config
    value = model.embed_step(token, state.position)
    value = _run_blocks_step(model.transformer.h[:config.n_prelude], value, state.cache.prelude)
    if state.temporal_memory is None:
        anchor = value
    else:
        if model.temporal_mixer is None:
            raise RuntimeError('Reference state contains temporal memory but the model has no temporal mixer')
        anchor = model.temporal_mixer.forward_step(value, state.temporal_memory)
    anchor = _run_blocks_step(
        model.transformer.h[config.n_prelude:config.core_start], anchor, state.cache.buffer)
    h = anchor
    for depth in range(state.cache.depth_steps):
        if depth:
            if model.depth_mixer is None:
                raise RuntimeError('Reference live depth loop requires a depth mixer')
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


def consume_prompt_reference(model, prompt_tokens, state, spec=None):
    if prompt_tokens.ndim != 1:
        raise ValueError('prompt_tokens must be a one-dimensional token sequence')
    logits = None
    for token in prompt_tokens:
        logits = decode_reference_step(model, token[None], state, spec)
    return logits
