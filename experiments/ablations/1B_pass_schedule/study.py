"""Frozen definitions for the 1B time-dependent recurrence study."""

from copy import deepcopy
import math

from experiments.serious import TOKENS_PER_UPDATE, base
from recurrence.schedule import build_probability_matrix
from .configs.common import (EVALUATION_COUNTS, HYBRID_DIAGONAL_MASS, SUPPORT,
                             schedule)


STUDY_NAME = '1B_pass_schedule'
RESULTS_ROOT = f'experiments/ablations/{STUDY_NAME}/results'
PANEL_PATH = f'{RESULTS_ROOT}/panel.json'
CHARACTERS = 10**9
UPDATES = math.ceil(CHARACTERS / TOKENS_PER_UPDATE)
ACTUAL_CHARACTERS = UPDATES * TOKENS_PER_UPDATE
CROSSOVER_FRACTION = .85
CROSSOVER_STEP = round(CROSSOVER_FRACTION * UPDATES)
WARMUP_UPDATES = round(.02 * UPDATES)
CHECKPOINT_STEPS = [0, 100, 250, 500, 1000, 2500, 5000, 7500,
                    CROSSOVER_STEP, CROSSOVER_STEP + 1, 9000, UPDATES]


FIXED_PASS_PROBABILITIES = [0.0, CROSSOVER_STEP / UPDATES,
                             (UPDATES - CROSSOVER_STEP) / UPDATES]
HARD_PHASE1_PASS_PROBABILITIES = [0.0, 1.0, 0.0]
HARD_PHASE2_PASS_PROBABILITIES = [0.0, 0.0, 1.0]


def _phase_matrix(mode, pass_probabilities):
    diagonal_mass = HYBRID_DIAGONAL_MASS if mode == 'hybrid' else None
    return build_probability_matrix(SUPPORT, mode, pass_probabilities, diagonal_mass)


HARD_PHASES = {
    mode: (_phase_matrix(mode, HARD_PHASE1_PASS_PROBABILITIES),
           _phase_matrix(mode, HARD_PHASE2_PASS_PROBABILITIES))
    for mode in ('temporal', 'depth', 'hybrid')
}

FIXED_MATRICES = {
    mode: _phase_matrix(mode, FIXED_PASS_PROBABILITIES)
    for mode in ('temporal', 'depth', 'hybrid')
}


RUNS = {
    'depth_fixed': dict(
        recurrence_mode='depth', recurrence_probabilities=FIXED_MATRICES['depth'],
        recurrence_probability_schedule=None, eval_u_t=EVALUATION_COUNTS['depth'][0],
        eval_u_d=EVALUATION_COUNTS['depth'][1],
        directory='depth_fixed_1B'),
    'depth_hard_2to4': dict(
        recurrence_mode='depth', recurrence_probabilities=[],
        recurrence_probability_schedule=schedule(
            HARD_PHASES['depth'][0], HARD_PHASES['depth'][1], CROSSOVER_STEP),
        eval_u_t=EVALUATION_COUNTS['depth'][0], eval_u_d=EVALUATION_COUNTS['depth'][1],
        directory='depth_hard_2to4_1B'),
    'temporal_fixed': dict(
        recurrence_mode='temporal', recurrence_probabilities=FIXED_MATRICES['temporal'],
        recurrence_probability_schedule=None, eval_u_t=EVALUATION_COUNTS['temporal'][0],
        eval_u_d=EVALUATION_COUNTS['temporal'][1], directory='temporal_fixed_1B'),
    'temporal_hard_2to4': dict(
        recurrence_mode='temporal', recurrence_probabilities=[],
        recurrence_probability_schedule=schedule(
            HARD_PHASES['temporal'][0], HARD_PHASES['temporal'][1], CROSSOVER_STEP),
        eval_u_t=EVALUATION_COUNTS['temporal'][0], eval_u_d=EVALUATION_COUNTS['temporal'][1],
        directory='temporal_hard_2to4_1B'),
    'hybrid_fixed': dict(
        recurrence_mode='hybrid', recurrence_probabilities=FIXED_MATRICES['hybrid'],
        recurrence_probability_schedule=None, eval_u_t=EVALUATION_COUNTS['hybrid'][0],
        eval_u_d=EVALUATION_COUNTS['hybrid'][1],
        directory='hybrid_fixed_1B'),
    'hybrid_hard_2to4': dict(
        recurrence_mode='hybrid', recurrence_probabilities=[],
        recurrence_probability_schedule=schedule(
            HARD_PHASES['hybrid'][0], HARD_PHASES['hybrid'][1], CROSSOVER_STEP),
        eval_u_t=EVALUATION_COUNTS['hybrid'][0], eval_u_d=EVALUATION_COUNTS['hybrid'][1],
        directory='hybrid_hard_2to4_1B'),
}


def run_config(name):
    if name not in RUNS:
        raise ValueError(f'Unknown {STUDY_NAME} arm: {name}')
    spec = RUNS[name]
    config = base('recurrent')
    config.update(
        out_dir=f'experiments/ablations/{STUDY_NAME}/{spec["directory"]}/results',
        eval_panel_path=PANEL_PATH, init_from='scratch', max_iters=UPDATES,
        lr_decay_iters=UPDATES, warmup_iters=WARMUP_UPDATES,
        recurrence_mode=spec['recurrence_mode'], recurrence_support=deepcopy(SUPPORT),
        recurrence_probabilities=deepcopy(spec['recurrence_probabilities']),
        recurrence_probability_schedule=deepcopy(spec['recurrence_probability_schedule']),
        eval_u_t=spec['eval_u_t'], eval_u_d=spec['eval_u_d'],
        deep_supervision=False, eval_interval=500, eval_iters=16,
        keep_checkpoints=True, checkpoint_steps=list(CHECKPOINT_STEPS),
    )
    return config
