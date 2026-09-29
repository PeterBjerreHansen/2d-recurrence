"""Sparse per-move targets and their losses (see docs/engine_policy_plan.md, Losses).

The human condition uses ordinary next-token targets: a ``[batch, time]`` tensor
of ply ids with -1 where no next ply exists. The legal and engine conditions use
the classes below. Both hold, for each supervised position, the flat index
``batch * time + position`` into the logits, and a list of (owner, move) pairs:
the legal moves of that position. Only supervised positions enter any loss.

Both losses average per position first, then over positions.
"""

# Best-move agreement counts a chosen move within this much Q of the best as agreeing.
NEAR_TIE = 0.01

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F


def _per_position_mean(values, owner, count):
    totals = torch.zeros(count, device=values.device, dtype=values.dtype).index_add_(0, owner, values)
    sizes = torch.bincount(owner, minlength=count).to(values.dtype)
    return totals / sizes


@dataclass
class _SparseTargets:
    positions: torch.Tensor  # [P] flat index batch * time + position
    owner: torch.Tensor      # [M] index into positions
    moves: torch.Tensor      # [M] move id

    def to(self, device, non_blocking=False):
        return type(self)(**{field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
                             for field in fields(self)})

    def pin_memory(self):
        return type(self)(**{field.name: getattr(self, field.name).pin_memory() for field in fields(self)})

    @property
    def count(self):
        return int(self.positions.numel())

    def _logits(self, logits):
        return logits.reshape(-1, logits.size(-1))[self.positions].float()


@dataclass
class LegalTargets(_SparseTargets):
    """Legality: target 1 for every legal move of a position, 0 for every other move."""

    def loss(self, logits):
        z = self._logits(logits)
        # sum_a BCE(z_a, t_a) = sum_a softplus(z_a) - sum_{a legal} z_a
        legal = torch.zeros(self.count, device=z.device).index_add_(0, self.owner, z[self.owner, self.moves])
        return ((F.softplus(z).sum(-1) - legal) / z.size(-1)).mean()

    @torch.no_grad()
    def metrics(self, logits):
        """Sums over positions, so batches can be pooled; divide by ``positions``."""
        z = self._logits(logits)
        target = torch.zeros_like(z, dtype=torch.bool)
        target[self.owner, self.moves] = True
        legal_count = target.sum(-1)
        legal_loss = torch.where(target, F.softplus(-z), 0).sum(-1)
        illegal_loss = torch.where(target, 0, F.softplus(z)).sum(-1)
        predicted = z > 0
        # Exact set: the |L| highest-scoring moves are exactly the legal moves.
        ranks = z.argsort(-1, descending=True).argsort(-1)
        exact = ((ranks < legal_count[:, None]) == target).all(-1)
        size = z.size(-1)
        prevalence = legal_count.float() / size
        constant = -(prevalence * prevalence.log() + (1 - prevalence) * (1 - prevalence).log())
        return dict(positions=float(self.count),
                    legal_bce=float((legal_loss / legal_count).sum()),
                    illegal_bce=float((illegal_loss / (size - legal_count)).sum()),
                    bce=float(((legal_loss + illegal_loss) / size).sum()),
                    constant_bce=float(constant.sum()),
                    true_positives=float((predicted & target).sum()),
                    predicted_positives=float(predicted.sum()),
                    legal_moves=float(target.sum()),
                    exact_set=float(exact.sum()))


@dataclass
class ValueTargets(_SparseTargets):
    """Engine values: binary cross-entropy against Q as a soft label, on legal moves only."""
    values: torch.Tensor     # [M] Q of each (owner, move), from the mover's perspective

    def loss(self, logits):
        z = self._logits(logits)[self.owner, self.moves]
        bce = F.softplus(z) - self.values * z
        return _per_position_mean(bce, self.owner, self.count).mean()

    @torch.no_grad()
    def metrics(self, logits):
        """Sums over positions, so batches can be pooled; divide by ``positions``."""
        z = self._logits(logits)[self.owner, self.moves]
        q = self.values
        bce = F.softplus(z) - q * z
        entropy = -(q * q.clamp_min(1e-12).log() + (1 - q) * (1 - q).clamp_min(1e-12).log())
        best_q = torch.full((self.count,), -1.0, device=q.device).scatter_reduce(
            0, self.owner, q, 'amax')
        top_z = torch.full((self.count,), -torch.inf, device=q.device).scatter_reduce(
            0, self.owner, z, 'amax')
        # The chosen move: highest predicted value; ties go to the lowest move id.
        chosen = torch.full((self.count,), torch.iinfo(torch.long).max, device=q.device).scatter_reduce(
            0, self.owner, torch.where(z == top_z[self.owner], self.moves, torch.iinfo(torch.long).max), 'amin')
        chosen_q = torch.zeros(self.count, device=q.device).index_add_(
            0, self.owner, torch.where(self.moves == chosen[self.owner], q, 0))
        return dict(positions=float(self.count),
                    bce=float(_per_position_mean(bce, self.owner, self.count).sum()),
                    label_entropy=float(_per_position_mean(entropy, self.owner, self.count).sum()),
                    regret=float((best_q - chosen_q).sum()),
                    best_move=float((best_q - chosen_q <= NEAR_TIE).sum()))
