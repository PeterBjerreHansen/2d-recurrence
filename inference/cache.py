"""State containers for incremental transformer inference."""

from dataclasses import dataclass
from itertools import chain

import torch


def model_cache_signature(config):
    """Return the architecture fields that determine physical cache streams."""
    return (
        config.block_size, config.vocab_size, config.n_layer, config.n_head,
        config.n_embd, config.bias,
        getattr(config, 'n_prelude', None), getattr(config, 'n_buffer', None),
        getattr(config, 'n_core', None), getattr(config, 'n_source', None),
        getattr(config, 'n_coda', None), getattr(config, 'recurrence_mode', 'baseline'),
    )


@dataclass
class LayerKVCache:
    """Dynamic key/value history for one physical transformer block."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    length: int = 0

    def append(self, keys, values):
        if keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape:
            raise ValueError('LayerKVCache entries must be matching [batch, heads, time, head_dim] tensors')
        if keys.shape[2] != 1:
            raise ValueError('LayerKVCache appends exactly one physical token at a time')
        if self.keys is None:
            self.keys = keys.detach()
            self.values = values.detach()
        else:
            if self.keys.shape[:2] != keys.shape[:2] or self.keys.shape[3:] != keys.shape[3:]:
                raise ValueError('LayerKVCache entry shape differs from its existing history')
            if self.keys.device != keys.device or self.keys.dtype != keys.dtype:
                raise ValueError('LayerKVCache entry device or dtype differs from its existing history')
            self.keys = torch.cat((self.keys, keys.detach()), dim=2)
            self.values = torch.cat((self.values, values.detach()), dim=2)
        self.length += 1

    def candidates(self, keys, values):
        """Return historical plus current K/V without changing the cache."""
        if keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape or keys.shape[2] != 1:
            raise ValueError('LayerKVCache candidates must be one matching [batch, heads, 1, head_dim] entry')
        if self.keys is None:
            return keys, values
        if self.keys.shape[:2] != keys.shape[:2] or self.keys.shape[3:] != keys.shape[3:]:
            raise ValueError('LayerKVCache candidate shape differs from its existing history')
        if self.keys.device != keys.device or self.keys.dtype != keys.dtype:
            raise ValueError('LayerKVCache candidate device or dtype differs from its existing history')
        return torch.cat((self.keys, keys), dim=2), torch.cat((self.values, values), dim=2)

    @property
    def bytes(self):
        return sum(value.numel() * value.element_size() for value in (self.keys, self.values)
                   if value is not None)


@dataclass
class ModelKVCache:
    """Physical-layer KV streams for one fixed live-inference session."""

    mode: str
    strategy: str
    depth_steps: int
    baseline: list[LayerKVCache]
    prelude: list[LayerKVCache]
    buffer: list[LayerKVCache]
    core: list[list[LayerKVCache]]
    source: list[LayerKVCache]
    coda: list[LayerKVCache]
    model_signature: tuple = ()

    def core_for_depth(self, depth):
        if not 0 <= depth < self.depth_steps:
            raise ValueError('Depth index is outside the live inference depth budget')
        return self.core[depth] if self.strategy == 'depth_specialized' else self.core[0]

    def _streams(self):
        if self.mode == 'baseline':
            return list(self.baseline)
        return list(chain(self.prelude, self.buffer, *(self.core), self.source, self.coda))

    @property
    def cache_length(self):
        lengths = {cache.length for cache in self._streams()}
        if len(lengths) > 1:
            raise RuntimeError(f'Live KV streams have inconsistent lengths: {sorted(lengths)}')
        return lengths.pop() if lengths else 0

    @property
    def bytes(self):
        return sum(cache.bytes for cache in self._streams())


def create_model_kv_cache(config, *, mode, depth_steps, strategy):
    """Allocate empty physical-layer streams for a validated live spec."""
    signature = model_cache_signature(config)
    if mode == 'baseline':
        return ModelKVCache(mode, strategy, depth_steps,
                            [LayerKVCache() for _ in range(config.n_layer)],
                            [], [], [], [], [], signature)
    core_depths = depth_steps if strategy == 'depth_specialized' else 1
    return ModelKVCache(
        mode, strategy, depth_steps,
        [],
        [LayerKVCache() for _ in range(config.n_prelude)],
        [LayerKVCache() for _ in range(config.n_buffer)],
        [[LayerKVCache() for _ in range(config.n_core)] for _ in range(core_depths)],
        [LayerKVCache() for _ in range(config.n_source)],
        [LayerKVCache() for _ in range(config.n_coda)],
        signature,
    )
