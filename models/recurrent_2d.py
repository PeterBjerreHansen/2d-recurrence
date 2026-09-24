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


RECURRENCE_MODES = frozenset({'hybrid', 'temporal', 'depth'})


def validate_recurrence_mode(mode):
    if mode not in RECURRENCE_MODES:
        raise ValueError("recurrence_mode must be one of 'hybrid', 'temporal', or 'depth'")


def validate_recurrence_counts(mode, u_t, u_d):
    validate_recurrence_mode(mode)
    if mode == 'temporal' and u_d:
        raise ValueError(f"recurrence_mode='temporal' cannot execute depth recurrence (U_D={u_d})")
    if mode == 'depth' and u_t:
        raise ValueError(f"recurrence_mode='depth' cannot execute temporal recurrence (U_T={u_t})")


def validate_update_probability_distribution(mode, update_support, update_probability_matrix):
    """Reject update probability mass on an unavailable axis."""
    validate_recurrence_mode(mode)
    for u_t, row in zip(update_support, update_probability_matrix):
        for u_d, probability in zip(update_support, row):
            if probability and ((mode == 'temporal' and u_d) or (mode == 'depth' and u_t)):
                validate_recurrence_counts(mode, u_t, u_d)


@dataclass
class RecurrentGPTConfig(GPTConfig):
    n_prelude: int = 1
    n_core: int = 4
    n_coda: int = 1
    n_buffer: int = 1
    n_source: int = 1
    recurrence_mode: str = 'hybrid'
    temporal_memory_gate_init: float = 0.1

    def __post_init__(self):
        validate_recurrence_mode(self.recurrence_mode)
        counts = (self.n_prelude, self.n_buffer, self.n_core, self.n_source, self.n_coda)
        if any(type(n) is not int or n < 0 for n in counts) or self.n_core == 0:
            raise ValueError('Block counts must be nonnegative integers with a nonempty core')
        if sum(counts) != self.n_layer:
            raise ValueError('n_layer must equal prelude + buffer + core + source + coda')
        if (isinstance(self.temporal_memory_gate_init, bool) or
                not isinstance(self.temporal_memory_gate_init, (int, float)) or
                not math.isfinite(self.temporal_memory_gate_init) or
                not 0 < self.temporal_memory_gate_init < 1):
            raise ValueError('temporal_memory_gate_init must be finite and strictly between zero and one')

    @classmethod
    def from_checkpoint(cls, model_args):
        """Missing layout fields in old checkpoints mean the original layout."""
        return cls(**{'n_prelude': 2, 'n_buffer': 0, 'n_source': 1,
                      'recurrence_mode': 'hybrid', **model_args})

    @property
    def uses_temporal_recurrence(self):
        return self.recurrence_mode in ('hybrid', 'temporal')

    @property
    def uses_depth_recurrence(self):
        return self.recurrence_mode in ('hybrid', 'depth')

    @property
    def core_start(self):
        return self.n_prelude + self.n_buffer

    @property
    def core_stop(self):
        """Exclusive stop index for the shared core segment."""
        return self.core_start + self.n_core

    @property
    def core_end(self):
        return self.core_stop - 1

    @property
    def source_start(self):
        """Inclusive start index for the temporal-source segment."""
        return self.core_stop

    @property
    def source_stop(self):
        """Exclusive stop index for the temporal-source segment."""
        return self.source_start + self.n_source

    @property
    def temporal_source_output_index(self):
        """Index of the block producing temporal memory, or the core end."""
        return self.source_stop - 1

    @property
    def source_index(self):
        """Backward-compatible alias for temporal_source_output_index."""
        return self.temporal_source_output_index

    @property
    def coda_start(self):
        return self.source_stop


def shift_right(memory):
    return torch.cat((torch.zeros_like(memory[:, :1]), memory[:, :-1]), dim=1)


class TemporalMixer(nn.Module):
    def __init__(self, width, bias=False, memory_gate_init=0.1):
        super().__init__()
        if (isinstance(memory_gate_init, bool) or
                not isinstance(memory_gate_init, (int, float)) or
                not math.isfinite(memory_gate_init) or not 0 < memory_gate_init < 1):
            raise ValueError('memory_gate_init must be finite and strictly between zero and one')
        self.memory_norm = LayerNorm(width, bias)
        self.prelude_norm = LayerNorm(width, bias)
        self.memory_value = nn.Linear(width, width, bias=False)
        self.prelude_value = nn.Linear(width, width, bias=False)
        self.gates = nn.Linear(2 * width, 2 * width, bias=True)
        nn.init.eye_(self.memory_value.weight)
        nn.init.eye_(self.prelude_value.weight)
        nn.init.zeros_(self.gates.weight)
        with torch.no_grad():
            self.gates.bias[:width].fill_(math.log(memory_gate_init / (1 - memory_gate_init)))
            self.gates.bias[width:].fill_(math.log((1 - memory_gate_init) / memory_gate_init))

    def forward(self, prelude, shifted_memory):
        mixed = self._mix(prelude, shifted_memory)
        # Stored rows have no predecessor at position zero. Other zero-valued
        # memories are valid inputs, not a sentinel for absent state.
        return torch.cat((prelude[:, :1], mixed[:, 1:]), dim=1)

    def _mix(self, prelude, memory):
        memory = self.memory_norm(memory)
        anchor = self.prelude_norm(prelude)
        alpha, beta = self.gates(torch.cat((memory, anchor), dim=-1)).sigmoid().chunk(2, dim=-1)
        return alpha * self.memory_value(memory) + beta * self.prelude_value(anchor)

    def forward_step(self, prelude, previous_memory):
        if prelude.ndim != 3 or previous_memory.ndim != 3 or prelude.shape[1] != 1 or previous_memory.shape[1] != 1:
            raise ValueError('TemporalMixer.forward_step expects [batch, 1, width] tensors')
        return self._mix(prelude, previous_memory)


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
        self.temporal_mixer = (TemporalMixer(config.n_embd, config.bias, config.temporal_memory_gate_init)
                               if config.uses_temporal_recurrence else None)
        self.depth_mixer = (DepthMixer(config.n_embd, config.bias)
                            if config.uses_depth_recurrence else None)

    def temporal_source(self, h):
        # With zero source blocks, temporal and depth candidates share the core output.
        for index in range(self.config.source_start, self.config.source_stop):
            h = self.transformer.h[index](h)
        return h

    def coda(self, h):
        for index in range(self.config.coda_start, self.config.n_layer):
            h = self.transformer.h[index](h)
        return h

    def _prediction(self, h, targets=None):
        return self.readout(self.coda(self.temporal_source(h)), targets)

    def forward(self, idx, targets=None, *, schedule: RecurrenceSchedule,
                deep_supervision=False, deep_supervision_lambda=0.25,
                return_components=False):
        if not isinstance(schedule, RecurrenceSchedule):
            raise TypeError('An explicit RecurrenceSchedule is required')
        validate_recurrence_counts(self.config.recurrence_mode, schedule.u_t, schedule.u_d)
        if (isinstance(deep_supervision_lambda, bool) or
                not isinstance(deep_supervision_lambda, (int, float)) or
                not math.isfinite(deep_supervision_lambda) or
                deep_supervision_lambda < 0):
            raise ValueError('deep_supervision_lambda must be a finite nonnegative number')
        auxiliary_supervision = (deep_supervision and targets is not None and
                                  deep_supervision_lambda > 0)
        p = self.embed(idx)
        core_start = self.config.core_start
        core_stop = self.config.core_stop
        blocks = self.transformer.h
        for index in range(self.config.n_prelude):
            p = blocks[index](p)

        temporal_state = None
        depth_state = None
        intermediate_losses = []
        for b in range(schedule.rounds):
            if temporal_state is None:
                anchor = p
            else:
                assert self.temporal_mixer is not None
                anchor = self.temporal_mixer(p, shift_right(temporal_state))
            for index in range(self.config.n_prelude, core_start):
                anchor = blocks[index](anchor)
            if depth_state is None:
                h = anchor
            else:
                assert self.depth_mixer is not None
                h = self.depth_mixer(depth_state, anchor)
            for index in range(core_start, core_stop):
                h = blocks[index](h)
            if b < schedule.rounds - 1:
                source = None
                if auxiliary_supervision:
                    source = self.temporal_source(h)
                    _, intermediate_loss = self.readout(self.coda(source), targets)
                    intermediate_losses.append(intermediate_loss)
                if schedule.depth_write_mask[b]:
                    depth_state = h
                if schedule.temporal_write_mask[b]:
                    temporal_state = source if source is not None else self.temporal_source(h)

        logits, final_loss = self._prediction(h, targets)
        if final_loss is None or not intermediate_losses:
            loss = final_loss
            intermediate_loss = None
        else:
            intermediate_loss = torch.stack(intermediate_losses).mean()
            loss = (final_loss + deep_supervision_lambda * intermediate_loss) / (1 + deep_supervision_lambda)
        if return_components:
            return logits, loss, dict(final_loss=final_loss, intermediate_loss=intermediate_loss,
                                     intermediate_passes=len(intermediate_losses))
        return logits, loss
