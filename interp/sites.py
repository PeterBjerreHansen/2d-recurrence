"""Residual-stream sites of every model family, read at the training-graph fixed point.

A *site* is the residual stream after one block application or after a mixer.
Sites are listed in execution order together with the number of transformer
blocks applied so far, so a looped core can be compared by physical layer and
by compute.

Recurrent models run as in ``experiments/ablations/live_warm_start/align.py``:
the prelude runs once, temporal memory is settled with gradient-free passes
until it stops changing, and one final pass reads the settled memory. A pass
runs the buffer, ``depth_steps`` core iterations and the T-source. Edits apply
on every pass, so an edited model is read at its own fixed point.

Site names: ``emb``, ``L1``..``L8`` for blocks, ``Tmix`` for the temporal mixer
output, ``Dmix@i`` for the depth mixer before core iteration ``i``. With more
than one core iteration, core blocks are named ``L3@i`` and so on.
"""

from dataclasses import dataclass

import torch

from models.recurrent_2d import Recurrent2DGPT, shift_right


@dataclass(frozen=True)
class Site:
    name: str
    kind: str           # 'embed', 'block' or 'mixer'
    layer: int          # physical layer of the last block applied (0 before any block)
    iteration: int      # 1-based core iteration; 0 outside the core
    applications: int   # transformer blocks applied so far


@dataclass
class Run:
    logits: torch.Tensor              # [batch, time, vocab]
    captures: dict                    # site name -> [points, width] float16 on CPU
    passes: int                       # temporal memory writes before the final pass
    memory_change: float | None       # max relative change of the last settling pass


def _core_name(index, iteration, depth_steps):
    return f'L{index + 1}' if depth_steps == 1 else f'L{index + 1}@{iteration}'


class SiteRunner:
    """Execute one model with named, capturable and editable sites."""

    def __init__(self, model, depth_steps=1, *, max_passes=64, tolerance=1e-3):
        self.model = model
        self.recurrent = isinstance(model, Recurrent2DGPT)
        config = model.config
        if not self.recurrent and depth_steps != 1:
            raise ValueError('A transformer has one core iteration')
        if self.recurrent and not config.uses_depth_recurrence and depth_steps != 1:
            raise ValueError('A temporal-only model has one core iteration')
        if depth_steps < 1 or max_passes < 1:
            raise ValueError('depth_steps and max_passes must be positive')
        self.depth_steps = depth_steps
        self.max_passes = max_passes
        self.tolerance = tolerance
        self.temporal = self.recurrent and config.uses_temporal_recurrence
        self.sites = self._site_list()
        self.site_names = [site.name for site in self.sites]

    def _site_list(self):
        config = self.model.config
        sites = [Site('emb', 'embed', 0, 0, 0)]
        if not self.recurrent:
            return sites + [Site(f'L{i + 1}', 'block', i + 1, 0, i + 1) for i in range(config.n_layer)]
        n = 0
        for i in range(config.n_prelude):
            n += 1
            sites.append(Site(f'L{i + 1}', 'block', i + 1, 0, n))
        if self.temporal:
            sites.append(Site('Tmix', 'mixer', n, 0, n))
        for i in range(config.n_prelude, config.core_start):
            n += 1
            sites.append(Site(f'L{i + 1}', 'block', i + 1, 0, n))
        for iteration in range(1, self.depth_steps + 1):
            if iteration > 1:
                sites.append(Site(f'Dmix@{iteration}', 'mixer', config.core_stop, iteration, n))
            for i in range(config.core_start, config.core_stop):
                n += 1
                sites.append(Site(_core_name(i, iteration, self.depth_steps), 'block', i + 1, iteration, n))
        for i in range(config.source_start, config.n_layer):
            n += 1
            sites.append(Site(f'L{i + 1}', 'block', i + 1, 0, n))
        return sites

    @torch.no_grad()
    def run(self, idx, *, capture=(), index=None, edits=None, passes=None, valid=None):
        """Run ``idx`` [batch, time]; capture ``capture`` sites at ``index`` = (rows, positions).

        ``edits`` maps a site name to a function of the full [batch, time, width]
        state. ``passes`` fixes the number of temporal writes instead of
        settling to ``tolerance``. ``valid`` [batch, time] restricts the
        convergence check to real (unpadded) positions.
        """
        unknown = (set(capture) | set(edits or {})) - set(self.site_names)
        if unknown:
            raise ValueError(f'Unknown sites: {sorted(unknown)}')
        if capture and index is None:
            raise ValueError('Captures need an index of (rows, positions)')
        edits = edits or {}
        captures = {}

        def visit(name, h, final):
            edit = edits.get(name)
            if edit is not None:
                h = edit(h)
            if final and name in capture:
                captures[name] = h[index].to(torch.float16).cpu()
            return h

        model = self.model
        blocks = model.transformer.h
        h = visit('emb', model.embed(idx), True)
        if not self.recurrent:
            for i, block in enumerate(blocks):
                h = visit(f'L{i + 1}', block(h), True)
            return Run(model.lm_head(model.transformer.ln_f(h)), captures, 0, None)

        config = model.config
        for i in range(config.n_prelude):
            h = visit(f'L{i + 1}', blocks[i](h), True)
        prelude = h
        memory, count, change = None, 0, None
        if self.temporal:
            limit = self.max_passes if passes is None else passes
            if limit < 1:
                raise ValueError('A temporal model needs at least one memory write')
            while count < limit:
                written = self._body(prelude, memory, visit, False)
                if memory is not None:
                    change = _relative_change(written, memory, valid)
                memory, count = written, count + 1
                if passes is None and change is not None and change < self.tolerance:
                    break
        h = self._body(prelude, memory, visit, True)
        for i in range(config.coda_start, config.n_layer):
            h = visit(f'L{i + 1}', blocks[i](h), True)
        return Run(model.lm_head(model.transformer.ln_f(h)), captures, count, change)

    def _body(self, prelude, memory, visit, final):
        """Mixer, buffer, core iterations and T-source; returns the T-source output."""
        model, config = self.model, self.model.config
        blocks = model.transformer.h
        anchor = prelude
        if memory is not None:
            anchor = visit('Tmix', model.temporal_mixer(prelude, shift_right(memory)), final)
        for i in range(config.n_prelude, config.core_start):
            anchor = visit(f'L{i + 1}', blocks[i](anchor), final)
        h = anchor
        for iteration in range(1, self.depth_steps + 1):
            if iteration > 1:
                h = visit(f'Dmix@{iteration}', model.depth_mixer(h, anchor), final)
            for i in range(config.core_start, config.core_stop):
                h = visit(_core_name(i, iteration, self.depth_steps), blocks[i](h), final)
        for i in range(config.source_start, config.source_stop):
            h = visit(f'L{i + 1}', blocks[i](h), final)
        return h


def _relative_change(new, old, valid=None):
    """Largest per-position relative L2 change, over valid positions."""
    change = (new - old).norm(dim=-1) / old.norm(dim=-1).clamp_min(1e-6)
    if valid is not None:
        change = change[valid]
    return float(change.max())
