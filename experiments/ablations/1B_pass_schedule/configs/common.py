"""Shared matrices and config entry points for ``1B_pass_schedule``."""

from importlib import import_module


SUPPORT = [0, 1, 3]
HYBRID_DIAGONAL_MASS = .80

EVALUATION_COUNTS = {'temporal': (3, 0), 'depth': (0, 3), 'hybrid': (3, 3)}


def max_pass_probabilities(matrix):
    """Return probabilities for one, two, and four executed passes."""
    return tuple(sum(matrix[i][j] for i, temporal in enumerate(SUPPORT)
                     for j, depth in enumerate(SUPPORT) if max(temporal, depth) == count)
                 for count in SUPPORT)


def schedule(first, second, crossover):
    return {
        'type': 'piecewise_constant',
        'phases': [
            {'start_step': 0, 'probabilities': first},
            {'start_step': crossover, 'probabilities': second},
        ],
    }


def pass_schedule_config(name):
    """Return the frozen config for one of the six study arms."""
    if name not in {'temporal_fixed', 'temporal_hard_2to4', 'depth_fixed',
                    'depth_hard_2to4', 'hybrid_fixed', 'hybrid_hard_2to4'}:
        raise ValueError(name)
    study = import_module('experiments.ablations.1B_pass_schedule.study')
    return study.run_config(name)
