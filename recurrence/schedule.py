"""Write-only schedules. See docs/RECURRENCE_CONTRACT.md."""
from collections import Counter
from dataclasses import dataclass
import math
import random


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError('Update counts must be nonnegative integers')


@dataclass(frozen=True)
class RecurrenceSchedule:
    temporal_write_mask: tuple[bool, ...]
    depth_write_mask: tuple[bool, ...]

    def __post_init__(self):
        masks = (self.temporal_write_mask, self.depth_write_mask)
        if any(type(mask) is not tuple or any(type(bit) is not bool for bit in mask) for mask in masks):
            raise ValueError('Write masks must be immutable tuples of booleans')
        if len(masks[0]) != len(masks[1]):
            raise ValueError('Write masks must have equal lengths')
        if max(self.u_t, self.u_d) != self.rounds - 1:
            raise ValueError('A schedule must have max(U_T, U_D) + 1 passes')

    @property
    def u_t(self):
        return sum(self.temporal_write_mask)

    @property
    def u_d(self):
        return sum(self.depth_write_mask)

    @property
    def rounds(self):
        return len(self.temporal_write_mask) + 1


def sample_schedule(u_t, u_d, rng):
    """Uniform subsets of nonfinal pass outputs, conditional on exact counts."""
    _count(u_t)
    _count(u_d)
    slots = max(u_t, u_d)
    def mask(count):
        selected = set(rng.sample(range(slots), count))
        return tuple(index in selected for index in range(slots))
    return RecurrenceSchedule(mask(u_t), mask(u_d))


class RecurrenceScheduleSampler:
    def __init__(self, support, probabilities, seed):
        self.support = tuple(support)
        if not self.support or len(set(self.support)) != len(self.support):
            raise ValueError('Support must be nonempty with distinct counts')
        for value in self.support:
            _count(value)
        n = len(self.support)
        if len(probabilities) != n or any(len(row) != n for row in probabilities):
            raise ValueError('Probability matrix must match the support')
        self.probabilities = tuple(tuple(float(p) for p in row) for row in probabilities)
        self.weights = tuple(p for row in self.probabilities for p in row)
        if any(not math.isfinite(p) or p < 0 for p in self.weights) or not math.isclose(sum(self.weights), 1.0, abs_tol=1e-9):
            raise ValueError('Probabilities must be finite, nonnegative, and sum to one')
        self.rng = random.Random(seed)
        self.pair_histogram = Counter()
        self.round_histogram = Counter()

    @property
    def draw_count(self):
        return sum(self.pair_histogram.values())

    def sample(self):
        index = self.rng.choices(range(len(self.weights)), weights=self.weights)[0]
        n = len(self.support)
        pair = self.support[index // n], self.support[index % n]
        schedule = sample_schedule(*pair, self.rng)
        self.pair_histogram[pair] += 1
        self.round_histogram[schedule.rounds] += 1
        return schedule

    def state_dict(self):
        return dict(support=self.support, probabilities=self.probabilities,
                    rng_state=self.rng.getstate(), draw_count=self.draw_count,
                    pair_histogram=dict(self.pair_histogram), round_histogram=dict(self.round_histogram))

    def load_state_dict(self, state):
        if state['support'] != self.support or state['probabilities'] != self.probabilities:
            raise ValueError('Checkpoint schedule distribution differs from configuration')
        if state['draw_count'] != sum(state['pair_histogram'].values()) or state['draw_count'] != sum(state['round_histogram'].values()):
            raise ValueError('Checkpoint schedule counts are inconsistent')
        self.rng.setstate(state['rng_state'])
        self.pair_histogram = Counter(state['pair_histogram'])
        self.round_histogram = Counter(state['round_histogram'])
