"""Frozen definitions for the 20B recurrence-axis discovery study."""
from copy import deepcopy
import math

from experiments.serious import TOKENS_PER_UPDATE, base
from recurrence.schedule import build_update_probability_matrix


STUDY_NAME = '20B_recurrence'
RESULTS_ROOT = f'experiments/long_runs/{STUDY_NAME}/results'
PANEL_PATH = f'{RESULTS_ROOT}/panel.json'
CHARACTERS = 20_000_000_000
UPDATES = math.ceil(CHARACTERS / TOKENS_PER_UPDATE)
ACTUAL_CHARACTERS = UPDATES * TOKENS_PER_UPDATE
WARMUP_UPDATES = 2_000
WSD_DECAY_START = 175_954
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
TEMPORAL_GATE_INIT = 0.10
UPDATE_SUPPORT = [0, 1, 3]
UPDATE_COUNT_PROBABILITIES = [
    (0.10, 0.80, 0.10),
    (0.05, 0.60, 0.35),
    (0.00, 0.40, 0.60),
    (0.00, 0.20, 0.80),
]
CURRICULUM_STARTS = (0, 9_776, 39_101, 97_752)
CURRICULUM_PHASES = tuple(zip(CURRICULUM_STARTS, UPDATE_COUNT_PROBABILITIES))
MAJOR_CHECKPOINT_STEPS = (9_776, 39_101, 97_752, WSD_DECAY_START, UPDATES)
CHECKPOINT_STEPS = [
    0, 9_776, 19_551, 39_101, 78_202, 97_752,
    136_853, 156_403, WSD_DECAY_START, UPDATES,
]
EVALUATION_CHECKPOINT_STEPS = tuple(step for step in CHECKPOINT_STEPS if step > 0)
PRIMARY_EVALUATION_CELLS = {
    'transformer': [(0, 0)],
    'temporal': [(0, 0), (3, 0)],
    'depth': [(0, 0), (0, 3)],
    'hybrid': [(0, 0), (3, 3)],
}
EXTENDED_EVALUATION_CELLS = {
    'transformer': [(0, 0)],
    'temporal': [(0, 0), (1, 0), (3, 0), (7, 0), (15, 0)],
    'depth': [(0, 0), (0, 1), (0, 3), (0, 7), (0, 15)],
    'hybrid': [(0, 0), (1, 1), (3, 3), (7, 7), (15, 15)],
}
MASK_SEEDS = [11, 23, 37]
# Sequential live-feedback, teacher-forced NLL on the selection panel at major
# checkpoints. Temporal-only live inference has exactly one core pass per
# token. Depth and hybrid use per-depth KV caches, matching the training
# graph's per-pass attention; depth live at J=4 must equal its (0,3) cell.
# The transformer's live execution is its ordinary forward pass.
LIVE_EVALUATION_SETTINGS = {
    'transformer': [],
    'temporal': [(1, 'final_depth')],
    'depth': [(4, 'depth_specialized')],
    'hybrid': [(1, 'depth_specialized'), (4, 'depth_specialized')],
}

ARM_ORDER = ('transformer', 'hybrid', 'temporal', 'depth')
RUNS = {
    'transformer': dict(
        architecture='baseline', recurrence_mode='hybrid', update_support=[],
        update_probabilities=[], update_probability_schedule=None,
        eval_u_t=0, eval_u_d=0, directory='transformer_20B'),
    'hybrid': dict(
        architecture='recurrent', recurrence_mode='hybrid',
        update_support=UPDATE_SUPPORT, update_probabilities=[],
        eval_u_t=3, eval_u_d=3, directory='hybrid_20B'),
    'temporal': dict(
        architecture='recurrent', recurrence_mode='temporal',
        update_support=UPDATE_SUPPORT, update_probabilities=[],
        eval_u_t=3, eval_u_d=0, directory='temporal_20B'),
    'depth': dict(
        architecture='recurrent', recurrence_mode='depth',
        update_support=UPDATE_SUPPORT, update_probabilities=[],
        eval_u_t=0, eval_u_d=3, directory='depth_20B'),
}


def _phase_matrix(mode, probabilities):
    diagonal_mass = 0.80 if mode == 'hybrid' else None
    return build_update_probability_matrix(
        UPDATE_SUPPORT, mode, probabilities, hybrid_diagonal_mass=diagonal_mass)


def update_probability_schedule(mode):
    if mode not in ('temporal', 'depth', 'hybrid'):
        raise ValueError(f'Unsupported recurrent mode: {mode}')
    return {
        'type': 'piecewise_constant',
        'phases': [
            {'start_step': start, 'update_probabilities': _phase_matrix(mode, probabilities)}
            for start, probabilities in CURRICULUM_PHASES
        ],
    }


def run_config(name):
    """Resolve one arm of the fixed 20B discovery protocol."""
    if name not in RUNS:
        raise ValueError(f'Unknown {STUDY_NAME} arm: {name}')

    spec = RUNS[name]
    config = base(spec['architecture'])
    schedule = (update_probability_schedule(spec['recurrence_mode'])
                if spec['architecture'] == 'recurrent' else None)
    config.update(
        out_dir=f"experiments/long_runs/{STUDY_NAME}/{spec['directory']}/results",
        eval_panel_path=PANEL_PATH,
        init_from='scratch',
        max_iters=UPDATES,
        learning_rate=LEARNING_RATE,
        min_lr=MIN_LEARNING_RATE,
        lr_schedule='wsd',
        warmup_iters=WARMUP_UPDATES,
        lr_decay_start=WSD_DECAY_START,
        lr_decay_iters=UPDATES,
        update_support=deepcopy(spec['update_support']),
        update_probabilities=deepcopy(spec['update_probabilities']),
        update_probability_schedule=deepcopy(schedule),
        recurrence_mode=spec['recurrence_mode'],
        eval_u_t=spec['eval_u_t'],
        eval_u_d=spec['eval_u_d'],
        deep_supervision=False,
        eval_interval=10_000,
        checkpoint_interval=1_000,
        eval_iters=16,
        keep_checkpoints=True,
        checkpoint_steps=list(CHECKPOINT_STEPS),
    )
    if spec['architecture'] == 'recurrent':
        config['temporal_memory_gate_init'] = TEMPORAL_GATE_INIT
    return config


def evaluation_cells(name, step):
    if name not in RUNS:
        raise ValueError(f'Unknown {STUDY_NAME} arm: {name}')
    if type(step) is not int or step < 0:
        raise ValueError('Evaluation step must be a nonnegative integer')
    cells = (EXTENDED_EVALUATION_CELLS if step in MAJOR_CHECKPOINT_STEPS
             else PRIMARY_EVALUATION_CELLS)
    return list(cells[name])


def live_evaluation_settings(name, step):
    """Return ``(depth_steps, kv_strategy)`` pairs; only major checkpoints run live."""
    if name not in RUNS:
        raise ValueError(f'Unknown {STUDY_NAME} arm: {name}')
    if type(step) is not int or step < 0:
        raise ValueError('Evaluation step must be a nonnegative integer')
    if step not in MAJOR_CHECKPOINT_STEPS:
        return []
    return list(LIVE_EVALUATION_SETTINGS[name])
