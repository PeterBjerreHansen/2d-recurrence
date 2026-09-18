"""Two-axis training trajectory; no hidden schedule sampling or persistent state.

The ModuleList keeps the baseline block order and checkpoint names. Its ranges
serve as prelude, optional injection buffer, core, optional source, and coda.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from model import GPT, GPTConfig, LayerNorm
from recurrence.schedule import RecurrenceSchedule


@dataclass
class RecurrentGPTConfig(GPTConfig):
    n_prelude: int = 1
    n_core: int = 4
    n_coda: int = 1
    n_buffer: int = 1
    n_source: int = 1

    def __post_init__(self):
        counts = (self.n_prelude, self.n_buffer, self.n_core, self.n_source, self.n_coda)
        if any(type(n) is not int or n < 0 for n in counts) or self.n_core == 0:
            raise ValueError('Block counts must be nonnegative integers with a nonempty core')
        if sum(counts) != self.n_layer:
            raise ValueError('n_layer must equal prelude + buffer + core + source + coda')

    @classmethod
    def from_checkpoint(cls, model_args):
        """Missing layout fields in old checkpoints mean the original layout."""
        return cls(**{'n_prelude': 2, 'n_buffer': 0, 'n_source': 1, **model_args})

    @property
    def core_start(self):
        return self.n_prelude + self.n_buffer

    @property
    def core_end(self):
        return self.core_start + self.n_core - 1

    @property
    def source_index(self):
        return self.core_end + self.n_source

    @property
    def coda_start(self):
        return self.source_index + 1


def shift_right(memory):
    return torch.cat((torch.zeros_like(memory[:, :1]), memory[:, :-1]), dim=1)


class TemporalMixer(nn.Module):
    def __init__(self, width, bias=False):
        super().__init__()
        self.memory_norm = LayerNorm(width, bias)
        self.prelude_norm = LayerNorm(width, bias)
        self.memory_value = nn.Linear(width, width, bias=False)
        self.prelude_value = nn.Linear(width, width, bias=False)
        self.gates = nn.Linear(2 * width, 2 * width, bias=True)
        nn.init.eye_(self.memory_value.weight)
        nn.init.eye_(self.prelude_value.weight)
        nn.init.zeros_(self.gates.weight)
        with torch.no_grad():
            self.gates.bias[:width].fill_(math.log(0.1 / 0.9))
            self.gates.bias[width:].fill_(math.log(0.9 / 0.1))

    def forward(self, prelude, shifted_memory):
        memory = self.memory_norm(shifted_memory)
        anchor = self.prelude_norm(prelude)
        alpha, beta = self.gates(torch.cat((memory, anchor), dim=-1)).sigmoid().chunk(2, dim=-1)
        mixed = alpha * self.memory_value(memory) + beta * self.prelude_value(anchor)
        # Stored rows have no predecessor at position zero. Other zero-valued
        # memories are valid inputs, not a sentinel for absent state.
        return torch.cat((prelude[:, :1], mixed[:, 1:]), dim=1)


class DepthMixer(nn.Module):
    def __init__(self, width, bias=False):
        super().__init__()
        self.state_norm = LayerNorm(width, bias)
        self.anchor_norm = LayerNorm(width, bias)
        self.state_value = nn.Linear(width, width, bias=False)
        self.anchor_value = nn.Linear(width, width, bias=False)
        # Start with balanced sources without adding another learned gate.
        with torch.no_grad():
            nn.init.eye_(self.state_value.weight)
            nn.init.eye_(self.anchor_value.weight)
            self.state_value.weight.mul_(0.5)
            self.anchor_value.weight.mul_(0.5)

    def forward(self, state, anchor):
        return self.state_value(self.state_norm(state)) + self.anchor_value(self.anchor_norm(anchor))


class Recurrent2DGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.temporal_mixer = TemporalMixer(config.n_embd, config.bias)
        self.depth_mixer = DepthMixer(config.n_embd, config.bias)

    def temporal_source(self, h):
        # With zero source blocks, temporal and depth candidates share the core output.
        for index in range(self.config.core_end + 1, self.config.coda_start):
            h = self.transformer.h[index](h)
        return h

    def forward(self, idx, targets=None, *, schedule: RecurrenceSchedule):
        if not isinstance(schedule, RecurrenceSchedule):
            raise TypeError('An explicit RecurrenceSchedule is required')
        p = self.embed(idx)
        core_start = self.config.core_start
        core_stop = self.config.core_end + 1
        blocks = self.transformer.h
        for index in range(self.config.n_prelude):
            p = blocks[index](p)

        temporal_state = None
        depth_state = None
        for b in range(schedule.rounds):
            anchor = p if temporal_state is None else self.temporal_mixer(p, shift_right(temporal_state))
            for index in range(self.config.n_prelude, core_start):
                anchor = blocks[index](anchor)
            h = anchor if depth_state is None else self.depth_mixer(depth_state, anchor)
            for index in range(core_start, core_stop):
                h = blocks[index](h)
            if b < schedule.rounds - 1:
                if schedule.depth_write_mask[b]:
                    depth_state = h
                if schedule.temporal_write_mask[b]:
                    temporal_state = self.temporal_source(h)

        h = self.temporal_source(h)
        for index in range(self.config.coda_start, self.config.n_layer):
            h = blocks[index](h)
        return self.readout(h, targets)
