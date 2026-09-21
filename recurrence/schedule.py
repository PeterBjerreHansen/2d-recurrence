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
    def __init__(self, support, probabilities=None, seed=1729):
        self.support = tuple(support)
        if not self.support or len(set(self.support)) != len(self.support):
            raise ValueError('Support must be nonempty with distinct counts')
        for value in self.support:
            _count(value)
        self.probabilities = validate_probability_matrix(self.support, probabilities)
        self.weights = _flatten(self.probabilities)
        self.rng = random.Random(seed)
        self.pair_histogram = Counter()
        self.round_histogram = Counter()

    @property
    def draw_count(self):
        return sum(self.pair_histogram.values())

    def sample(self, probabilities=None):
        matrix = (self.probabilities if probabilities is None
                  else validate_probability_matrix(self.support, probabilities))
        weights = _flatten(matrix)
        if weights is None:
            raise ValueError('A probability matrix is required for sampling')
        index = self.rng.choices(range(len(weights)), weights=weights)[0]
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
        if tuple(state['support']) != self.support:
            raise ValueError('Checkpoint schedule distribution differs from configuration')
        state_probabilities = state.get('probabilities')
        if state_probabilities is not None:
            state_probabilities = validate_probability_matrix(self.support, state_probabilities)
            if self.probabilities is not None and state_probabilities != self.probabilities:
                raise ValueError('Checkpoint schedule distribution differs from configuration')
            # This accepts old static sampler checkpoints when a caller did
            # not provide a constructor matrix.
            if self.probabilities is None:
                self.probabilities = state_probabilities
                self.weights = _flatten(state_probabilities)
        if state['draw_count'] != sum(state['pair_histogram'].values()) or state['draw_count'] != sum(state['round_histogram'].values()):
            raise ValueError('Checkpoint schedule counts are inconsistent')
        self.rng.setstate(state['rng_state'])
        self.pair_histogram = Counter(state['pair_histogram'])
        self.round_histogram = Counter(state['round_histogram'])


def _flatten(probabilities):
    return (tuple(p for row in probabilities for p in row)
            if probabilities is not None else None)


def validate_probability_matrix(support, probabilities):
    """Return an immutable, validated probability matrix for ``support``."""
    if probabilities is None:
        return None
    support = tuple(support)
    n = len(support)
    try:
        matrix = tuple(tuple(float(p) for p in row) for row in probabilities)
    except (TypeError, ValueError):
        raise ValueError('Probability matrix must contain numeric rows') from None
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError('Probability matrix must match the support')
    weights = _flatten(matrix)
    if any(not math.isfinite(p) or p < 0 for p in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError('Probabilities must be finite, nonnegative, and sum to one')
    return matrix


def _validate_mode_matrix(mode, support, probabilities):
    if mode not in {'hybrid', 'temporal', 'depth'}:
        raise ValueError("recurrence_mode must be one of 'hybrid', 'temporal', or 'depth'")
    if mode == 'hybrid':
        return
    support = tuple(support)
    if 0 not in support:
        raise ValueError('Temporal and depth recurrence modes require support to contain zero')
    zero = support.index(0)
    for i, row in enumerate(probabilities):
        for j, probability in enumerate(row):
            if probability and ((mode == 'temporal' and j != zero) or
                                 (mode == 'depth' and i != zero)):
                raise ValueError(f'Probability matrix is incompatible with recurrence_mode={mode!r}')


def build_probability_matrix(support, recurrence_mode, pass_probabilities,
                            hybrid_diagonal_mass=None):
    """Build the canonical matrix for pass-count bucket probabilities.

    The current recurrence contract uses support ``[0, 1, 3]``.  These are
    write counts, corresponding to one, two, and four executed passes.
    """
    support = tuple(support)
    if support != (0, 1, 3):
        raise ValueError('Probability constructors currently require support [0, 1, 3]')
    if recurrence_mode not in {'hybrid', 'temporal', 'depth'}:
        raise ValueError("recurrence_mode must be one of 'hybrid', 'temporal', or 'depth'")
    if hybrid_diagonal_mass is not None and recurrence_mode != 'hybrid':
        raise ValueError('hybrid_diagonal_mass is only valid for hybrid recurrence')
    try:
        pass_probabilities = tuple(float(p) for p in pass_probabilities)
    except (TypeError, ValueError):
        raise ValueError('pass_probabilities must contain numeric values') from None
    if len(pass_probabilities) != len(support) or any(not math.isfinite(p) or p < 0 for p in pass_probabilities):
        raise ValueError('pass_probabilities must match support and be finite and nonnegative')
    if not math.isclose(sum(pass_probabilities), 1.0, abs_tol=1e-9):
        raise ValueError('pass_probabilities must sum to one')

    matrix = [[0.0] * len(support) for _ in support]
    if recurrence_mode == 'temporal':
        for i, probability in enumerate(pass_probabilities):
            matrix[i][0] = probability
    elif recurrence_mode == 'depth':
        for j, probability in enumerate(pass_probabilities):
            matrix[0][j] = probability
    else:
        if (hybrid_diagonal_mass is None or isinstance(hybrid_diagonal_mass, bool) or
                not isinstance(hybrid_diagonal_mass, (int, float)) or
                not math.isfinite(hybrid_diagonal_mass) or not 0 <= hybrid_diagonal_mass <= 1):
            raise ValueError('hybrid_diagonal_mass must be a finite number between zero and one')
        diagonal_mass = float(hybrid_diagonal_mass)
        # One-pass probability is necessarily on the (0, 0) cell.  For the
        # two- and four-pass buckets, allocate the declared diagonal share to
        # the largest-count diagonal, then fan the remainder out by the
        # number of available placements (the exact 0.05/0.01/0.03 matrix in
        # the current contract follows from this rule).
        matrix[0][0] = pass_probabilities[0]
        for count_index in (1, 2):
            count = support[count_index]
            bucket = pass_probabilities[count_index]
            matrix[count_index][count_index] = diagonal_mass * bucket
            off_diagonal = (1 - diagonal_mass) * bucket
            candidates = [(i, j) for i, temporal in enumerate(support)
                          for j, depth in enumerate(support)
                          if max(temporal, depth) == count and i != j]
            if candidates:
                total_weight = sum(2 * min(support[i], support[j]) + 1
                                   for i, j in candidates)
                for i, j in candidates:
                    weight = 2 * min(support[i], support[j]) + 1
                    matrix[i][j] = off_diagonal * weight / total_weight
            elif off_diagonal:
                raise ValueError('Hybrid pass bucket has no off-diagonal placements')
    # Keep the canonical decimal values stable (for example, emit 0.05
    # rather than the representational 0.04999999999999999 from the fan-out
    # arithmetic) so generated matrices compare cleanly with frozen configs.
    return validate_probability_matrix(support, [[round(value, 12) for value in row]
                                                 for row in matrix])


def probabilities_at_step(config, step):
    """Resolve the probability matrix active at an absolute optimizer step."""
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer')
    support = tuple(config.get('recurrence_support', ()))
    mode = config.get('recurrence_mode', 'hybrid')
    schedule = config.get('recurrence_probability_schedule')
    if schedule is None:
        configured = config.get('recurrence_probabilities')
        matrix = validate_probability_matrix(support, configured)
        if matrix is None:
            raise ValueError('recurrence_probabilities is required when no probability schedule is configured')
        _validate_mode_matrix(mode, support, matrix)
        # Preserve the static configuration's public representation for
        # backwards compatibility; the sampler normalizes it at the boundary.
        return configured
    if not isinstance(schedule, dict) or schedule.get('type') != 'piecewise_constant':
        raise ValueError("recurrence_probability_schedule.type must be 'piecewise_constant'")
    phases = schedule.get('phases')
    if not isinstance(phases, (list, tuple)) or not phases:
        raise ValueError('recurrence_probability_schedule.phases must be nonempty')
    starts = []
    matrices = []
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) != {'start_step', 'probabilities'}:
            raise ValueError("Each probability phase must contain only 'start_step' and 'probabilities'")
        start = phase['start_step']
        if type(start) is not int or start < 0:
            raise ValueError('Probability phase start_step must be a nonnegative integer')
        if starts and start <= starts[-1]:
            raise ValueError('Probability phase start_step values must be strictly increasing')
        matrix = validate_probability_matrix(support, phase['probabilities'])
        _validate_mode_matrix(mode, support, matrix)
        starts.append(start)
        matrices.append(matrix)
    if starts[0] != 0:
        raise ValueError('The first probability phase must start at step zero')
    active = 0
    for index, start in enumerate(starts[1:], 1):
        if step < start:
            break
        active = index
    return matrices[active]


def probability_map_at_step(config, step=0):
    """Return active probabilities keyed by ``(U_T, U_D)`` pair."""
    matrix = probabilities_at_step(config, step)
    support = tuple(config.get('recurrence_support', ()))
    return {(temporal, depth): matrix[i][j]
            for i, temporal in enumerate(support)
            for j, depth in enumerate(support)}
