import pytest

from importlib import import_module
from recurrence.schedule import (RecurrenceScheduleSampler, build_update_probability_matrix,
                                 normalize_update_config, update_probability_map_at_step,
                                 update_probabilities_at_step)


_schedule = import_module('experiments.long_runs.5B_axis.configs.common')
UPDATE_PROBABILITIES = _schedule.UPDATE_PROBABILITIES
UPDATE_SUPPORT = _schedule.UPDATE_SUPPORT


def test_hybrid_schedule_is_symmetric_and_normalized():
    probabilities = UPDATE_PROBABILITIES['hybrid_matched']
    assert probabilities == [
        [.10, .05, .01],
        [.05, .40, .03],
        [.01, .03, .32],
    ]
    assert sum(sum(row) for row in probabilities) == pytest.approx(1.0)
    assert all(probabilities[i][j] == probabilities[j][i]
               for i in range(len(UPDATE_SUPPORT)) for j in range(len(UPDATE_SUPPORT)))


def test_hybrid_schedule_preserves_max_update_distribution():
    probabilities = UPDATE_PROBABILITIES['hybrid_matched']
    by_max = {count: 0.0 for count in UPDATE_SUPPORT}
    for i, temporal in enumerate(UPDATE_SUPPORT):
        for j, depth in enumerate(UPDATE_SUPPORT):
            by_max[max(temporal, depth)] += probabilities[i][j]
    assert by_max == pytest.approx({0: .10, 1: .50, 3: .40})


def test_hybrid_schedule_has_declared_near_diagonal_off_diagonal_mass():
    probabilities = UPDATE_PROBABILITIES['hybrid_matched']
    for count, expected_diagonal, expected_bucket in ((1, .40, .50), (3, .32, .40)):
        index = UPDATE_SUPPORT.index(count)
        diagonal = probabilities[index][index]
        bucket = sum(
            probabilities[i][j]
            for i, temporal in enumerate(UPDATE_SUPPORT)
            for j, depth in enumerate(UPDATE_SUPPORT)
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
        actual = build_update_probability_matrix(UPDATE_SUPPORT, mode, *args)
        assert all(actual_row == pytest.approx(expected_row)
                   for actual_row, expected_row in zip(actual, expected))


def test_probability_schedule_switches_on_absolute_step():
    first = [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    second = [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]
    config = dict(update_support=UPDATE_SUPPORT, recurrence_mode='hybrid',
                  update_probabilities=[], update_probability_schedule={
                      'type': 'piecewise_constant',
                      'phases': [{'start_step': 0, 'update_probabilities': first},
                                 {'start_step': 3, 'update_probabilities': second}]})
    assert all(row == pytest.approx(expected) for row, expected in zip(update_probabilities_at_step(config, 0), first))
    assert all(row == pytest.approx(expected) for row, expected in zip(update_probabilities_at_step(config, 2), first))
    assert all(row == pytest.approx(expected) for row, expected in zip(update_probabilities_at_step(config, 3), second))
    assert all(row == pytest.approx(expected) for row, expected in zip(update_probabilities_at_step(config, 300), second))


def test_probability_map_uses_active_schedule_phase():
    config = dict(update_support=UPDATE_SUPPORT, recurrence_mode='hybrid',
                  update_probabilities=[], update_probability_schedule={
                      'type': 'piecewise_constant',
                      'phases': [{'start_step': 0, 'update_probabilities': [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]},
                                 {'start_step': 3, 'update_probabilities': [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]}]})
    assert update_probability_map_at_step(config, 0) == {(0, 0): 1.0, (0, 1): 0.0, (0, 3): 0.0,
                                                   (1, 0): 0.0, (1, 1): 0.0, (1, 3): 0.0,
                                                   (3, 0): 0.0, (3, 1): 0.0, (3, 3): 0.0}
    assert update_probability_map_at_step(config, 3)[(3, 3)] == 1.0


@pytest.mark.parametrize('schedule', [
    {'type': 'piecewise_constant', 'phases': []},
    {'type': 'piecewise_constant', 'phases': [{'start_step': 1, 'update_probabilities': [[1, 0, 0]] * 3}]},
    {'type': 'piecewise_constant', 'phases': [
        {'start_step': 0, 'update_probabilities': [[1, 0, 0]] * 3},
        {'start_step': 0, 'update_probabilities': [[1, 0, 0]] * 3}]},
    {'type': 'piecewise_constant', 'phases': [
        {'start_step': 0, 'update_probabilities': [[1, 0, 0]] * 3},
        {'start_step': 2, 'update_probabilities': [[0, 1, 0], [0, 0, 0], [0, 0, 0]]}]},
])
def test_probability_schedule_rejects_malformed_or_incompatible_phases(schedule):
    config = dict(update_support=UPDATE_SUPPORT, recurrence_mode='depth',
                  update_probabilities=[], update_probability_schedule=schedule)
    with pytest.raises(ValueError):
        update_probabilities_at_step(config, 0)


def test_sampler_accepts_supplied_matrices_without_changing_static_state():
    first = [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    second = [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]
    sampler = RecurrenceScheduleSampler(UPDATE_SUPPORT, None, 11)
    assert sampler.sample(first).u_t == 0
    assert sampler.sample(second).u_t == 3
    assert sampler.update_probabilities is None
    assert sampler.state_dict()['update_probabilities'] is None


def test_legacy_names_and_sampler_state_remain_loadable():
    first = [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    legacy_config = dict(recurrence_support=UPDATE_SUPPORT, recurrence_mode='hybrid',
                         recurrence_probabilities=[], recurrence_probability_schedule={
                             'type': 'piecewise_constant',
                             'phases': [{'start_step': 0, 'probabilities': first}]})
    normalized = normalize_update_config(legacy_config)
    assert set(normalized) >= {'update_support', 'update_probabilities', 'update_probability_schedule'}
    assert 'recurrence_support' not in normalized
    assert update_probabilities_at_step(legacy_config, 0) == tuple(tuple(row) for row in first)

    sampler = RecurrenceScheduleSampler(UPDATE_SUPPORT, None, 11)
    sampler.sample(first)
    current = sampler.state_dict()
    legacy_state = {
        'support': current['update_support'],
        'probabilities': current['update_probabilities'],
        'rng_state': current['rng_state'],
        'draw_count': current['draw_count'],
        'pair_histogram': current['pair_histogram'],
        'round_histogram': current['round_histogram'],
    }
    restored = RecurrenceScheduleSampler(UPDATE_SUPPORT, None, 999)
    restored.load_state_dict(legacy_state)
    assert restored.state_dict() == current
    assert set(restored.state_dict()) == {
        'update_support', 'update_probabilities', 'rng_state', 'draw_count',
        'pair_histogram', 'round_histogram'}


def test_empty_legacy_alias_does_not_conflict_with_new_value():
    config = dict(recurrence_support=[], update_support=UPDATE_SUPPORT,
                  recurrence_probabilities=[], update_probabilities=[
                      [1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    normalized = normalize_update_config(config)
    assert normalized['update_support'] == UPDATE_SUPPORT
    assert normalized['update_probabilities'][0][0] == 1.


@pytest.mark.parametrize('diagonal_mass', [-.01, 1.01, float('nan'), True])
def test_probability_constructor_rejects_invalid_hybrid_diagonal_mass(diagonal_mass):
    with pytest.raises(ValueError):
        build_update_probability_matrix(UPDATE_SUPPORT, 'hybrid', [.10, .50, .40], diagonal_mass)
