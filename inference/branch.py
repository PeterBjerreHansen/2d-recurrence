"""Batched continuations of one live state without copying its KV history.

A teacher-forced live state (batch one) is the shared prefix. Each branch
attends to that prefix plus its own suffix history, so evaluating dozens of
candidate continuations never replicates the prefix caches. The step mirrors
``inference.live.decode_live_step`` exactly, including temporal memory and the
``final_depth``/``depth_specialized`` commit rules.
"""

from dataclasses import dataclass
import math

import torch

from inference.cache import LayerKVCache
from inference.live import LiveInferenceState


@dataclass
class _Stream:
    prefix: LayerKVCache
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def select(self, indices):
        if self.keys is None:
            return _Stream(self.prefix)
        return _Stream(self.prefix, self.keys.index_select(0, indices),
                       self.values.index_select(0, indices))


@dataclass
class BranchState:
    position: int
    mode: str
    strategy: str
    depth_steps: int
    batch_size: int
    baseline: list
    prelude: list
    buffer: list
    core: list
    source: list
    coda: list
    temporal_memory: torch.Tensor | None = None

    def core_for_depth(self, depth):
        return self.core[depth] if self.strategy == 'depth_specialized' else self.core[0]


def branch_from_live(state: LiveInferenceState, batch_size):
    """Start ``batch_size`` empty branches after a batch-one live prefix."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    cache = state.cache
    for layer in cache._streams():
        if layer.keys is not None and layer.keys.shape[0] != 1:
            raise ValueError('Branching requires a batch-one live prefix')
    wrap = lambda layers: [_Stream(layer) for layer in layers]
    memory = state.temporal_memory
    if memory is not None:
        if memory.shape[0] != 1:
            raise ValueError('Branching requires a batch-one live prefix')
        memory = memory.expand(batch_size, -1, -1)
    return BranchState(state.position, cache.mode, cache.strategy, cache.depth_steps, batch_size,
                       wrap(cache.baseline), wrap(cache.prelude), wrap(cache.buffer),
                       [wrap(layers) for layers in cache.core],
                       wrap(cache.source), wrap(cache.coda), memory)


def select_branches(state: BranchState, indices):
    """Return the branches at ``indices``, e.g. the parents of the next trie level."""
    indices = torch.as_tensor(indices, dtype=torch.long)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError('indices must be a nonempty one-dimensional sequence')
    if int(indices.min()) < 0 or int(indices.max()) >= state.batch_size:
        raise ValueError('Branch index is outside the current batch')
    select = lambda streams: [stream.select(indices.to(_device(stream, indices))) for stream in streams]
    memory = state.temporal_memory
    if memory is not None:
        memory = memory.index_select(0, indices.to(memory.device))
    return BranchState(state.position, state.mode, state.strategy, state.depth_steps, indices.numel(),
                       select(state.baseline), select(state.prelude), select(state.buffer),
                       [select(streams) for streams in state.core],
                       select(state.source), select(state.coda), memory)


def _device(stream, fallback):
    if stream.keys is not None:
        return stream.keys.device
    if stream.prefix.keys is not None:
        return stream.prefix.keys.device
    return fallback.device


def _attend(attention, x, stream, commit):
    batch, _, width = x.shape
    heads = attention.n_head
    head_dim = width // heads
    q, k, v = attention.c_attn(x).split(attention.n_embd, dim=2)
    q = q.view(batch, 1, heads, head_dim).transpose(1, 2)
    k = k.view(batch, 1, heads, head_dim).transpose(1, 2)
    v = v.view(batch, 1, heads, head_dim).transpose(1, 2)
    suffix_keys = k if stream.keys is None else torch.cat((stream.keys, k), dim=2)
    suffix_values = v if stream.values is None else torch.cat((stream.values, v), dim=2)
    scale = 1.0 / math.sqrt(head_dim)
    scores = [q @ suffix_keys.transpose(-1, -2) * scale]
    prefix = stream.prefix
    prefix_length = 0 if prefix.keys is None else prefix.keys.shape[2]
    if prefix_length:
        # Branches become the query axis of one batch-one attention over the prefix.
        prefix_scores = q.transpose(0, 2) @ prefix.keys.transpose(-1, -2) * scale
        scores.insert(0, prefix_scores.transpose(0, 2))
    weights = torch.cat(scores, dim=-1).softmax(dim=-1)
    y = weights[..., prefix_length:] @ suffix_values
    if prefix_length:
        y = y + (weights[..., :prefix_length].transpose(0, 2) @ prefix.values).transpose(0, 2)
    if commit:
        stream.keys, stream.values = suffix_keys, suffix_values
    y = y.transpose(1, 2).contiguous().view(batch, 1, width)
    return attention.resid_dropout(attention.c_proj(y))


def _run_blocks(blocks, value, streams, commit=True):
    for block, stream in zip(blocks, streams):
        value = value + _attend(block.attn, block.ln_1(value), stream, commit)
        value = value + block.mlp(block.ln_2(value))
    return value


@torch.no_grad()
def branch_step(model, tokens, state: BranchState):
    """Feed one token to every branch and return next-token logits [batch, vocab]."""
    tokens = torch.as_tensor(tokens, dtype=torch.long, device=model.lm_head.weight.device)
    if tokens.ndim != 1 or tokens.numel() != state.batch_size:
        raise ValueError('branch_step expects one token per branch')
    if state.position >= model.config.block_size:
        raise ValueError('Live inference context limit reached')
    value = model.embed_step(tokens, state.position)
    if state.mode == 'baseline':
        output = _run_blocks(model.transformer.h, value, state.baseline)
    else:
        config = model.config
        blocks = model.transformer.h
        value = _run_blocks(blocks[:config.n_prelude], value, state.prelude)
        anchor = (value if state.temporal_memory is None else
                  model.temporal_mixer.forward_step(value, state.temporal_memory))
        anchor = _run_blocks(blocks[config.n_prelude:config.core_start], anchor, state.buffer)
        h = anchor
        for depth in range(state.depth_steps):
            if depth:
                h = model.depth_mixer(h, anchor)
            h = _run_blocks(blocks[config.core_start:config.core_stop], h, state.core_for_depth(depth),
                            commit=(state.strategy == 'depth_specialized' or depth == state.depth_steps - 1))
        source = _run_blocks(blocks[config.source_start:config.source_stop], h, state.source)
        state.temporal_memory = source if config.uses_temporal_recurrence else None
        output = _run_blocks(blocks[config.coda_start:config.n_layer], source, state.coda)
    logits, _ = model.readout(output)
    state.position += 1
    return logits[:, 0]
