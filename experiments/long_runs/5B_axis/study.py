"""Frozen definitions for the 5B recurrence-axis training study.

This module only constructs configurations.  It does not launch training or
create result files.  The four arms receive the same data exposure and
optimizer schedule; the recurrent arms intentionally differ in executed
computation according to the declared axis schedule.
"""
from copy import deepcopy

from .configs.common import (
    EVALUATION_COUNTS,
    PROBABILITIES,
    SUPPORT,
)
from experiments.serious import TOKENS_PER_UPDATE, base


STUDY_NAME = '5B_axis'
RESULTS_ROOT = f'experiments/long_runs/{STUDY_NAME}/results'
PANEL_PATH = f'{RESULTS_ROOT}/panel.json'
CHARACTERS = 5 * 10**9
UPDATES = (CHARACTERS + TOKENS_PER_UPDATE - 1) // TOKENS_PER_UPDATE
ACTUAL_CHARACTERS = UPDATES * TOKENS_PER_UPDATE
WARMUP_UPDATES = round(0.02 * UPDATES)

# These checkpoints support early-learning inspection while retaining a few
# late points for plotting.  Step zero is an explicit initial checkpoint.
CHECKPOINT_STEPS = [
    0, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 40_000, UPDATES,
]

RUNS = {
    'transformer': dict(
        architecture='baseline',
        recurrence_mode='hybrid',
        recurrence_support=[],
        recurrence_probabilities=[],
        eval_u_t=0,
        eval_u_d=0,
        directory='transformer_5B',
    ),
    'temporal': dict(
        architecture='recurrent',
        recurrence_mode='temporal',
        recurrence_support=SUPPORT,
        recurrence_probabilities=PROBABILITIES['temporal'],
        eval_u_t=EVALUATION_COUNTS['temporal'][0],
        eval_u_d=EVALUATION_COUNTS['temporal'][1],
        directory='temporal_5B',
    ),
    'depth': dict(
        architecture='recurrent',
        recurrence_mode='depth',
        recurrence_support=SUPPORT,
        recurrence_probabilities=PROBABILITIES['depth'],
        eval_u_t=EVALUATION_COUNTS['depth'][0],
        eval_u_d=EVALUATION_COUNTS['depth'][1],
        directory='depth_5B',
    ),
    'hybrid': dict(
        architecture='recurrent',
        recurrence_mode='hybrid',
        recurrence_support=SUPPORT,
        recurrence_probabilities=PROBABILITIES['hybrid_matched'],
        eval_u_t=EVALUATION_COUNTS['hybrid_matched'][0],
        eval_u_d=EVALUATION_COUNTS['hybrid_matched'][1],
        directory='hybrid_5B',
    ),
}


def run_config(name):
    """Return the complete immutable-by-convention config for one study arm."""
    if name not in RUNS:
        raise ValueError(f'Unknown {STUDY_NAME} arm: {name}')

    spec = RUNS[name]
    config = base(spec['architecture'])
    config.update(
        out_dir=f'experiments/long_runs/{STUDY_NAME}/{spec["directory"]}/results',
        eval_panel_path=PANEL_PATH,
        init_from='scratch',
        max_iters=UPDATES,
        lr_decay_iters=UPDATES,
        warmup_iters=WARMUP_UPDATES,
        recurrence_mode=spec['recurrence_mode'],
        recurrence_support=deepcopy(spec['recurrence_support']),
        recurrence_probabilities=deepcopy(spec['recurrence_probabilities']),
        eval_u_t=spec['eval_u_t'],
        eval_u_d=spec['eval_u_d'],
        deep_supervision=False,
        eval_interval=500,
        eval_iters=16,
        keep_checkpoints=True,
        checkpoint_steps=list(CHECKPOINT_STEPS),
    )
    return config
