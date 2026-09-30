"""Sparse per-move targets and their losses (see docs/engine_policy_plan.md, Losses).

The human condition uses ordinary next-token targets: a ``[batch, time]`` tensor
of ply ids with -1 where no next ply exists. The other conditions use the
classes below. Each holds, for every supervised position, the flat index
``batch * time + position`` into the logits, and a list of (owner, move) pairs:
the moves listed for that position. Only supervised positions enter any loss.

- ``LegalTargets``: sigmoid per move, 1 for the listed (legal) moves, 0 otherwise.
- ``PolicyTargets``: softmax over the listed (legal) moves against a teacher's
  distribution, such as Leela's search probabilities.

Losses average per position first, then over positions. ``metrics`` returns sums
over positions, so batches can be pooled; ``summary`` turns pooled sums into the
reported metrics.
"""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

_NO_MOVE = torch.iinfo(torch.long).max


def _per_position_sum(values, owner, count):
    return torch.zeros(count, device=values.device, dtype=values.dtype).index_add_(0, owner, values)


def _per_position_max(values, owner, count):
    return torch.full((count,), -torch.inf, device=values.device).scatter_reduce(0, owner, values, 'amax')


def _top_move(scores, moves, owner, count):
    """The highest-scoring move of each position; ties go to the lowest move id."""
    top = _per_position_max(scores, owner, count)
    candidates = torch.where(scores == top[owner], moves, _NO_MOVE)
    return torch.full((count,), _NO_MOVE, device=moves.device).scatter_reduce(0, owner, candidates, 'amin')


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
        legal = _per_position_sum(z[self.owner, self.moves], self.owner, self.count)
        return ((F.softplus(z).sum(-1) - legal) / z.size(-1)).mean()

    @torch.no_grad()
    def metrics(self, logits):
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

    @staticmethod
    def summary(name, sums):
        result = {f'{name}_{key}': sums[key] / sums['positions']
                  for key in ('legal_bce', 'illegal_bce', 'constant_bce', 'exact_set')}
        result[f'{name}_precision'] = sums['true_positives'] / max(sums['predicted_positives'], 1)
        result[f'{name}_recall'] = sums['true_positives'] / sums['legal_moves']
        return result


@dataclass
class PolicyTargets(_SparseTargets):
    """A teacher's distribution over the legal moves: softmax over those moves, cross-entropy."""
    weights: torch.Tensor    # [M] teacher probability of each (owner, move); sums to one per position

    def _log_probabilities(self, logits):
        z = self._logits(logits)[self.owner, self.moves]
        top = _per_position_max(z, self.owner, self.count)
        log_total = _per_position_sum(torch.exp(z - top[self.owner]), self.owner, self.count).log() + top
        return z - log_total[self.owner]

    def loss(self, logits):
        cross_entropy = -_per_position_sum(self.weights * self._log_probabilities(logits), self.owner, self.count)
        return cross_entropy.mean()

    @torch.no_grad()
    def metrics(self, logits):
        log_p = self._log_probabilities(logits)
        w = self.weights
        cross_entropy = -_per_position_sum(w * log_p, self.owner, self.count)
        entropy = -_per_position_sum(torch.where(w > 0, w * w.clamp_min(1e-12).log(), 0), self.owner, self.count)
        agree = _top_move(log_p, self.moves, self.owner, self.count) == _top_move(w, self.moves, self.owner, self.count)
        return dict(positions=float(self.count), cross_entropy=float(cross_entropy.sum()),
                    kl=float((cross_entropy - entropy).sum()), top_move_agreement=float(agree.sum()))

    @staticmethod
    def summary(name, sums):
        return {f'{name}_{key}': sums[key] / sums['positions'] for key in ('kl', 'top_move_agreement')}
