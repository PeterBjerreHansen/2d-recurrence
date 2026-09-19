import pytest

from experiments.ablations.recurrence_axes.configs.common import (
    PROBABILITIES,
    SUPPORT,
)


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
