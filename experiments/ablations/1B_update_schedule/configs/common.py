"""Shared matrices and config entry points for ``1B_update_schedule``."""

from importlib import import_module


UPDATE_SUPPORT = [0, 1, 3]
HYBRID_DIAGONAL_MASS = .80

EVALUATION_COUNTS = {'temporal': (3, 0), 'depth': (0, 3), 'hybrid': (3, 3)}


def max_update_probabilities(matrix):
    """Return probabilities for each supported max update count."""
    return tuple(sum(matrix[i][j] for i, temporal in enumerate(UPDATE_SUPPORT)
                     for j, depth in enumerate(UPDATE_SUPPORT)
                     if max(temporal, depth) == count)
                 for count in UPDATE_SUPPORT)


def update_schedule(first, second, crossover):
    return {
        'type': 'piecewise_constant',
        'phases': [
            {'start_step': 0, 'update_probabilities': first},
            {'start_step': crossover, 'update_probabilities': second},
        ],
    }


def update_schedule_config(name):
    """Return the frozen config for one of the six study arms."""
    if name not in {'temporal_fixed', 'temporal_hard_1to3', 'depth_fixed',
                    'depth_hard_1to3', 'hybrid_fixed', 'hybrid_hard_1to3'}:
        raise ValueError(name)
    study = import_module('experiments.ablations.1B_update_schedule.study')
    return study.run_config(name)
