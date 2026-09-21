import pytest

from importlib import import_module
from recurrence.schedule import (RecurrenceScheduleSampler, build_probability_matrix,
                                 probability_map_at_step, probabilities_at_step)


_schedule = import_module('experiments.long_runs.5B_axis.configs.common')
PROBABILITIES = _schedule.PROBABILITIES
SUPPORT = _schedule.SUPPORT


def test_hybrid_schedule_is_symmetric_and_normalized():
    probabilities = PROBABILITIES['hybrid_matched']
    assert probabilities == [
        [.10, .05, .01],
        [.05, .40, .03],
        [.01, .03, .32],
    ]
    assert sum(sum(row) for row in probabilities) == pytest.approx(1.0)
    assert all(probabilities[i][j] == probabilities[j][i]
               for i in range(len(SUPPORT)) for j in range(len(SUPPORT)))


def test_hybrid_schedule_preserves_max_pass_distribution():
    probabilities = PROBABILITIES['hybrid_matched']
    by_max = {count: 0.0 for count in SUPPORT}
    for i, temporal in enumerate(SUPPORT):
        for j, depth in enumerate(SUPPORT):
            by_max[max(temporal, depth)] += probabilities[i][j]
    assert by_max == pytest.approx({0: .10, 1: .50, 3: .40})


def test_hybrid_schedule_has_declared_near_diagonal_off_diagonal_mass():
    probabilities = PROBABILITIES['hybrid_matched']
    for count, expected_diagonal, expected_bucket in ((1, .40, .50), (3, .32, .40)):
        index = SUPPORT.index(count)
        diagonal = probabilities[index][index]
        bucket = sum(
            probabilities[i][j]
            for i, temporal in enumerate(SUPPORT)
            for j, depth in enumerate(SUPPORT)
            if max(temporal, depth) == count
        )
        assert diagonal == pytest.approx(expected_diagonal)
        assert bucket - diagonal == pytest.approx(.20 * expected_bucket)


def test_probability_constructor_reproduces_declared_axis_matrices():
    for mode, args, expected in [
        ('temporal', ([.10, .50, .40],), [[.10, .00, .00], [.50, .00, .00], [.40, .00, .00]]),
        ('depth', ([.10, .50, .40],), [[.10, .50, .40], [.00, .00, .00], [.00, .00, .00]]),
        ('hybrid', ([.10, .50, .40], .80), [[.10, .05, .01], [.05, .40, .03], [.01, .03, .32]]),
    ]:
        actual = build_probability_matrix(SUPPORT, mode, *args)
        assert all(actual_row == pytest.approx(expected_row)
                   for actual_row, expected_row in zip(actual, expected))


def test_probability_schedule_switches_on_absolute_step():
    first = [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    second = [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]
    config = dict(recurrence_support=SUPPORT, recurrence_mode='hybrid',
                  recurrence_probabilities=[], recurrence_probability_schedule={
                      'type': 'piecewise_constant',
                      'phases': [{'start_step': 0, 'probabilities': first},
                                 {'start_step': 3, 'probabilities': second}]})
    assert all(row == pytest.approx(expected) for row, expected in zip(probabilities_at_step(config, 0), first))
    assert all(row == pytest.approx(expected) for row, expected in zip(probabilities_at_step(config, 2), first))
    assert all(row == pytest.approx(expected) for row, expected in zip(probabilities_at_step(config, 3), second))
    assert all(row == pytest.approx(expected) for row, expected in zip(probabilities_at_step(config, 300), second))


def test_probability_map_uses_active_schedule_phase():
    config = dict(recurrence_support=SUPPORT, recurrence_mode='hybrid',
                  recurrence_probabilities=[], recurrence_probability_schedule={
                      'type': 'piecewise_constant',
                      'phases': [{'start_step': 0, 'probabilities': [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]},
                                 {'start_step': 3, 'probabilities': [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]}]})
    assert probability_map_at_step(config, 0) == {(0, 0): 1.0, (0, 1): 0.0, (0, 3): 0.0,
                                                   (1, 0): 0.0, (1, 1): 0.0, (1, 3): 0.0,
                                                   (3, 0): 0.0, (3, 1): 0.0, (3, 3): 0.0}
    assert probability_map_at_step(config, 3)[(3, 3)] == 1.0


@pytest.mark.parametrize('schedule', [
    {'type': 'piecewise_constant', 'phases': []},
    {'type': 'piecewise_constant', 'phases': [{'start_step': 1, 'probabilities': [[1, 0, 0]] * 3}]},
    {'type': 'piecewise_constant', 'phases': [
        {'start_step': 0, 'probabilities': [[1, 0, 0]] * 3},
        {'start_step': 0, 'probabilities': [[1, 0, 0]] * 3}]},
    {'type': 'piecewise_constant', 'phases': [
        {'start_step': 0, 'probabilities': [[1, 0, 0]] * 3},
        {'start_step': 2, 'probabilities': [[0, 1, 0], [0, 0, 0], [0, 0, 0]]}]},
])
def test_probability_schedule_rejects_malformed_or_incompatible_phases(schedule):
    config = dict(recurrence_support=SUPPORT, recurrence_mode='depth',
                  recurrence_probabilities=[], recurrence_probability_schedule=schedule)
    with pytest.raises(ValueError):
        probabilities_at_step(config, 0)


def test_sampler_accepts_supplied_matrices_without_changing_static_state():
    first = [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    second = [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]
    sampler = RecurrenceScheduleSampler(SUPPORT, None, 11)
    assert sampler.sample(first).u_t == 0
    assert sampler.sample(second).u_t == 3
    assert sampler.probabilities is None
    assert sampler.state_dict()['probabilities'] is None


@pytest.mark.parametrize('diagonal_mass', [-.01, 1.01, float('nan'), True])
def test_probability_constructor_rejects_invalid_hybrid_diagonal_mass(diagonal_mass):
    with pytest.raises(ValueError):
        build_probability_matrix(SUPPORT, 'hybrid', [.10, .50, .40], diagonal_mass)
