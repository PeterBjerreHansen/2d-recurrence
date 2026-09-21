"""Frozen definitions for the 1B time-dependent update schedule study."""

from copy import deepcopy
import math

from experiments.serious import TOKENS_PER_UPDATE, base
from recurrence.schedule import build_update_probability_matrix
from .configs.common import (EVALUATION_COUNTS, HYBRID_DIAGONAL_MASS, UPDATE_SUPPORT,
                             update_schedule)


STUDY_NAME = '1B_update_schedule'
RESULTS_ROOT = f'experiments/ablations/{STUDY_NAME}/results'
PANEL_PATH = f'{RESULTS_ROOT}/panel.json'
CHARACTERS = 10**9
UPDATES = math.ceil(CHARACTERS / TOKENS_PER_UPDATE)
ACTUAL_CHARACTERS = UPDATES * TOKENS_PER_UPDATE
LEARNING_RATE = 3e-4
CROSSOVER_FRACTION = .85
CROSSOVER_STEP = round(CROSSOVER_FRACTION * UPDATES)
WARMUP_UPDATES = 0
CHECKPOINT_STEPS = [0, 100, 250, 500, 1000, 2500, 5000, 7500,
                    CROSSOVER_STEP, CROSSOVER_STEP + 1, 9000, UPDATES]


FIXED_UPDATE_PROBABILITIES = [0.0, CROSSOVER_STEP / UPDATES,
                              (UPDATES - CROSSOVER_STEP) / UPDATES]
HARD_PHASE1_UPDATE_PROBABILITIES = [0.0, 1.0, 0.0]
HARD_PHASE2_UPDATE_PROBABILITIES = [0.0, 0.0, 1.0]


def _phase_matrix(mode, update_probabilities):
    diagonal_mass = HYBRID_DIAGONAL_MASS if mode == 'hybrid' else None
    return build_update_probability_matrix(UPDATE_SUPPORT, mode, update_probabilities, diagonal_mass)


HARD_PHASES = {
    mode: (_phase_matrix(mode, HARD_PHASE1_UPDATE_PROBABILITIES),
           _phase_matrix(mode, HARD_PHASE2_UPDATE_PROBABILITIES))
    for mode in ('temporal', 'depth', 'hybrid')
}

FIXED_MATRICES = {
    mode: _phase_matrix(mode, FIXED_UPDATE_PROBABILITIES)
    for mode in ('temporal', 'depth', 'hybrid')
}


RUNS = {
    'depth_fixed': dict(
        recurrence_mode='depth', update_probabilities=FIXED_MATRICES['depth'],
        update_probability_schedule=None, eval_u_t=EVALUATION_COUNTS['depth'][0],
        eval_u_d=EVALUATION_COUNTS['depth'][1],
        directory='depth_fixed_1B'),
    'depth_hard_1to3': dict(
        recurrence_mode='depth', update_probabilities=[],
        update_probability_schedule=update_schedule(
            HARD_PHASES['depth'][0], HARD_PHASES['depth'][1], CROSSOVER_STEP),
        eval_u_t=EVALUATION_COUNTS['depth'][0], eval_u_d=EVALUATION_COUNTS['depth'][1],
        directory='depth_hard_1to3_1B'),
    'temporal_fixed': dict(
        recurrence_mode='temporal', update_probabilities=FIXED_MATRICES['temporal'],
        update_probability_schedule=None, eval_u_t=EVALUATION_COUNTS['temporal'][0],
        eval_u_d=EVALUATION_COUNTS['temporal'][1], directory='temporal_fixed_1B'),
    'temporal_hard_1to3': dict(
        recurrence_mode='temporal', update_probabilities=[],
        update_probability_schedule=update_schedule(
            HARD_PHASES['temporal'][0], HARD_PHASES['temporal'][1], CROSSOVER_STEP),
        eval_u_t=EVALUATION_COUNTS['temporal'][0], eval_u_d=EVALUATION_COUNTS['temporal'][1],
        directory='temporal_hard_1to3_1B'),
    'hybrid_fixed': dict(
        recurrence_mode='hybrid', update_probabilities=FIXED_MATRICES['hybrid'],
        update_probability_schedule=None, eval_u_t=EVALUATION_COUNTS['hybrid'][0],
        eval_u_d=EVALUATION_COUNTS['hybrid'][1],
        directory='hybrid_fixed_1B'),
    'hybrid_hard_1to3': dict(
        recurrence_mode='hybrid', update_probabilities=[],
        update_probability_schedule=update_schedule(
            HARD_PHASES['hybrid'][0], HARD_PHASES['hybrid'][1], CROSSOVER_STEP),
        eval_u_t=EVALUATION_COUNTS['hybrid'][0], eval_u_d=EVALUATION_COUNTS['hybrid'][1],
        directory='hybrid_hard_1to3_1B'),
}


def run_config(name):
    if name not in RUNS:
        raise ValueError(f'Unknown {STUDY_NAME} arm: {name}')
    spec = RUNS[name]
    config = base('recurrent')
    config.update(
        out_dir=f'experiments/ablations/{STUDY_NAME}/{spec["directory"]}/results',
        eval_panel_path=PANEL_PATH, init_from='scratch', max_iters=UPDATES,
        learning_rate=LEARNING_RATE, decay_lr=False,
        lr_decay_iters=UPDATES, warmup_iters=WARMUP_UPDATES,
        recurrence_mode=spec['recurrence_mode'], update_support=deepcopy(UPDATE_SUPPORT),
        update_probabilities=deepcopy(spec['update_probabilities']),
        update_probability_schedule=deepcopy(spec['update_probability_schedule']),
        eval_u_t=spec['eval_u_t'], eval_u_d=spec['eval_u_d'],
        deep_supervision=False, eval_interval=500, eval_iters=16,
        keep_checkpoints=True, checkpoint_steps=list(CHECKPOINT_STEPS),
    )
    return config
