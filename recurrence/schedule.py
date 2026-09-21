"""Write-only schedules. See docs/RECURRENCE_CONTRACT.md."""
from collections import Counter
from dataclasses import dataclass
import math
import random


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError('Update counts must be nonnegative integers')


def _validate_update_support(update_support):
    update_support = tuple(update_support)
    if not update_support or len(set(update_support)) != len(update_support):
        raise ValueError('update_support must be nonempty with distinct counts')
    for value in update_support:
        _count(value)
    return update_support


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
    def __init__(self, update_support, update_probabilities=None, seed=1729):
        self.update_support = _validate_update_support(update_support)
        self.update_probabilities = validate_update_probability_matrix(
            self.update_support, update_probabilities)
        self.weights = _flatten(self.update_probabilities)
        self.rng = random.Random(seed)
        self.pair_histogram = Counter()
        self.round_histogram = Counter()

    @property
    def draw_count(self):
        return sum(self.pair_histogram.values())

    def sample(self, update_probability_matrix=None):
        matrix = (self.update_probabilities if update_probability_matrix is None
                  else validate_update_probability_matrix(self.update_support,
                                                          update_probability_matrix))
        weights = _flatten(matrix)
        if weights is None:
            raise ValueError('An update probability matrix is required for sampling')
        index = self.rng.choices(range(len(weights)), weights=weights)[0]
        n = len(self.update_support)
        pair = self.update_support[index // n], self.update_support[index % n]
        schedule = sample_schedule(*pair, self.rng)
        self.pair_histogram[pair] += 1
        self.round_histogram[schedule.rounds] += 1
        return schedule

    def state_dict(self):
        return dict(update_support=self.update_support,
                    update_probabilities=self.update_probabilities,
                    rng_state=self.rng.getstate(), draw_count=self.draw_count,
                    pair_histogram=dict(self.pair_histogram),
                    round_histogram=dict(self.round_histogram))

    def load_state_dict(self, state):
        saved_support = state.get('update_support', state.get('support'))
        if saved_support is None or tuple(saved_support) != self.update_support:
            raise ValueError('Checkpoint schedule distribution differs from configuration')
        state_probabilities = state.get('update_probabilities', state.get('probabilities'))
        if state_probabilities is not None:
            state_probabilities = validate_update_probability_matrix(
                self.update_support, state_probabilities)
            if (self.update_probabilities is not None and
                    state_probabilities != self.update_probabilities):
                raise ValueError('Checkpoint schedule distribution differs from configuration')
            # Accept old static sampler checkpoints when the constructor did
            # not provide a static matrix.
            if self.update_probabilities is None:
                self.update_probabilities = state_probabilities
                self.weights = _flatten(state_probabilities)
        if (state['draw_count'] != sum(state['pair_histogram'].values()) or
                state['draw_count'] != sum(state['round_histogram'].values())):
            raise ValueError('Checkpoint schedule counts are inconsistent')
        self.rng.setstate(state['rng_state'])
        self.pair_histogram = Counter(state['pair_histogram'])
        self.round_histogram = Counter(state['round_histogram'])


def _flatten(update_probability_matrix):
    return (tuple(p for row in update_probability_matrix for p in row)
            if update_probability_matrix is not None else None)


def validate_update_probability_matrix(update_support, update_probability_matrix):
    """Return an immutable matrix validated against ``update_support``."""
    update_support = _validate_update_support(update_support)
    if update_probability_matrix is None:
        return None
    n = len(update_support)
    try:
        matrix = tuple(tuple(float(p) for p in row)
                       for row in update_probability_matrix)
    except (TypeError, ValueError):
        raise ValueError('Update probability matrix must contain numeric rows') from None
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError('Update probability matrix must match update_support')
    weights = _flatten(matrix)
    if (any(not math.isfinite(p) or p < 0 for p in weights) or
            not math.isclose(sum(weights), 1.0, abs_tol=1e-9)):
        raise ValueError('Update probabilities must be finite, nonnegative, and sum to one')
    return matrix


def _validate_update_mode_matrix(mode, update_support, update_probability_matrix):
    if mode not in {'hybrid', 'temporal', 'depth'}:
        raise ValueError("recurrence_mode must be one of 'hybrid', 'temporal', or 'depth'")
    if mode == 'hybrid':
        return
    update_support = tuple(update_support)
    if 0 not in update_support:
        raise ValueError('Temporal and depth recurrence modes require update_support to contain zero')
    zero = update_support.index(0)
    for i, row in enumerate(update_probability_matrix):
        for j, probability in enumerate(row):
            if probability and ((mode == 'temporal' and j != zero) or
                                 (mode == 'depth' and i != zero)):
                raise ValueError(f'Update probability matrix is incompatible with recurrence_mode={mode!r}')


def build_update_probability_matrix(update_support, recurrence_mode, update_probabilities,
                                    hybrid_diagonal_mass=None):
    """Build a joint update matrix from max-update-count probabilities.

    The current recurrence contract explicitly supports update_support
    ``[0, 1, 3]``. Physical pass counts remain derived from the selected pair.
    """
    update_support = _validate_update_support(update_support)
    if update_support != (0, 1, 3):
        raise ValueError('Update probability constructors currently require update_support [0, 1, 3]')
    if recurrence_mode not in {'hybrid', 'temporal', 'depth'}:
        raise ValueError("recurrence_mode must be one of 'hybrid', 'temporal', or 'depth'")
    if hybrid_diagonal_mass is not None and recurrence_mode != 'hybrid':
        raise ValueError('hybrid_diagonal_mass is only valid for hybrid recurrence')
    try:
        update_probabilities = tuple(float(p) for p in update_probabilities)
    except (TypeError, ValueError):
        raise ValueError('update_probabilities must contain numeric values') from None
    if (len(update_probabilities) != len(update_support) or
            any(not math.isfinite(p) or p < 0 for p in update_probabilities)):
        raise ValueError('update_probabilities must match update_support and be finite and nonnegative')
    if not math.isclose(sum(update_probabilities), 1.0, abs_tol=1e-9):
        raise ValueError('update_probabilities must sum to one')

    matrix = [[0.0] * len(update_support) for _ in update_support]
    if recurrence_mode == 'temporal':
        for i, probability in enumerate(update_probabilities):
            matrix[i][0] = probability
    elif recurrence_mode == 'depth':
        for j, probability in enumerate(update_probabilities):
            matrix[0][j] = probability
    else:
        if (hybrid_diagonal_mass is None or isinstance(hybrid_diagonal_mass, bool) or
                not isinstance(hybrid_diagonal_mass, (int, float)) or
                not math.isfinite(hybrid_diagonal_mass) or not 0 <= hybrid_diagonal_mass <= 1):
            raise ValueError('hybrid_diagonal_mass must be a finite number between zero and one')
        diagonal_mass = float(hybrid_diagonal_mass)
        matrix[0][0] = update_probabilities[0]
        for count_index in (1, 2):
            count = update_support[count_index]
            bucket = update_probabilities[count_index]
            candidates = [(i, j) for i, temporal in enumerate(update_support)
                          for j, depth in enumerate(update_support)
                          if max(temporal, depth) == count and i != j]
            matrix[count_index][count_index] = diagonal_mass * bucket
            off_diagonal = (1 - diagonal_mass) * bucket
            if candidates:
                total_weight = sum(2 * min(update_support[i], update_support[j]) + 1
                                   for i, j in candidates)
                for i, j in candidates:
                    weight = 2 * min(update_support[i], update_support[j]) + 1
                    matrix[i][j] = off_diagonal * weight / total_weight
            elif off_diagonal:
                raise ValueError('Hybrid update bucket has no off-diagonal placements')
    return validate_update_probability_matrix(
        update_support, [[round(value, 12) for value in row] for row in matrix])


def _is_empty_config_value(value):
    return value is None or (isinstance(value, (list, tuple)) and not value)


def normalize_update_config(config):
    """Normalize historical recurrence keys to the update-based vocabulary."""
    normalized = dict(config)
    aliases = (
        ('recurrence_support', 'update_support'),
        ('recurrence_probabilities', 'update_probabilities'),
        ('recurrence_probability_schedule', 'update_probability_schedule'),
    )
    for old_key, new_key in aliases:
        if old_key not in normalized:
            continue
        old_value = normalized[old_key]
        new_value = normalized.get(new_key)
        if (new_key in normalized and not _is_empty_config_value(new_value) and
                not _is_empty_config_value(old_value) and new_value != old_value):
            raise ValueError(f'Conflicting {old_key} and {new_key} configuration')
        if new_key not in normalized or _is_empty_config_value(new_value):
            normalized[new_key] = old_value
        normalized.pop(old_key, None)

    schedule = normalized.get('update_probability_schedule')
    if isinstance(schedule, dict):
        schedule = dict(schedule)
        phases = []
        for phase in schedule.get('phases', []):
            if not isinstance(phase, dict):
                phases.append(phase)
                continue
            phase = dict(phase)
            if ('probabilities' in phase and 'update_probabilities' in phase and
                    phase['probabilities'] != phase['update_probabilities']):
                raise ValueError('Conflicting phase probability keys')
            if 'update_probabilities' not in phase and 'probabilities' in phase:
                phase['update_probabilities'] = phase['probabilities']
            phase.pop('probabilities', None)
            phases.append(phase)
        schedule['phases'] = phases
        normalized['update_probability_schedule'] = schedule
    return normalized


def update_probabilities_at_step(config, step):
    """Resolve the update probability matrix active at an optimizer step."""
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer')
    config = normalize_update_config(config)
    update_support = tuple(config.get('update_support', ()))
    mode = config.get('recurrence_mode', 'hybrid')
    schedule = config.get('update_probability_schedule')
    if schedule is None:
        configured = config.get('update_probabilities')
        matrix = validate_update_probability_matrix(update_support, configured)
        if matrix is None:
            raise ValueError('update_probabilities is required when no update probability schedule is configured')
        _validate_update_mode_matrix(mode, update_support, matrix)
        return configured
    if not isinstance(schedule, dict) or schedule.get('type') != 'piecewise_constant':
        raise ValueError("update_probability_schedule.type must be 'piecewise_constant'")
    phases = schedule.get('phases')
    if not isinstance(phases, (list, tuple)) or not phases:
        raise ValueError('update_probability_schedule.phases must be nonempty')
    starts = []
    matrices = []
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) != {'start_step', 'update_probabilities'}:
            raise ValueError("Each update probability phase must contain only 'start_step' and 'update_probabilities'")
        start = phase['start_step']
        if type(start) is not int or start < 0:
            raise ValueError('Update probability phase start_step must be a nonnegative integer')
        if starts and start <= starts[-1]:
            raise ValueError('Update probability phase start_step values must be strictly increasing')
        matrix = validate_update_probability_matrix(update_support, phase['update_probabilities'])
        _validate_update_mode_matrix(mode, update_support, matrix)
        starts.append(start)
        matrices.append(matrix)
    if starts[0] != 0:
        raise ValueError('The first update probability phase must start at step zero')
    active = 0
    for index, start in enumerate(starts[1:], 1):
        if step < start:
            break
        active = index
    return matrices[active]


def update_probability_map_at_step(config, step=0):
    """Return active update probabilities keyed by ``(U_T, U_D)``."""
    config = normalize_update_config(config)
    matrix = update_probabilities_at_step(config, step)
    update_support = tuple(config.get('update_support', ()))
    return {(temporal, depth): matrix[i][j]
            for i, temporal in enumerate(update_support)
            for j, depth in enumerate(update_support)}
