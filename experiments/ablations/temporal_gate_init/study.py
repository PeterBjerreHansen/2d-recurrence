"""Frozen definitions for the 250M temporal-gate initialization preflight."""

from copy import deepcopy
import math

from experiments.serious import TOKENS_PER_UPDATE, base
from .configs.common import GATE_INITIALIZATIONS


STUDY_NAME = 'temporal_gate_init'
RESULTS_ROOT = f'experiments/ablations/{STUDY_NAME}/results'
PANEL_PATH = f'{RESULTS_ROOT}/panel.json'
CHARACTERS = 250_000_000
UPDATES = math.ceil(CHARACTERS / TOKENS_PER_UPDATE)
ACTUAL_CHARACTERS = UPDATES * TOKENS_PER_UPDATE
WARMUP_UPDATES = round(.02 * UPDATES)
CHECKPOINT_STEPS = [0, 100, 500, 1000, 1500, 2000, UPDATES]


def run_config(name):
    if name not in GATE_INITIALIZATIONS:
        raise ValueError(f'Unknown {STUDY_NAME} arm: {name}')
    config = base('recurrent')
    config.update(
        out_dir=f'{RESULTS_ROOT}/{name}/results',
        eval_panel_path=PANEL_PATH,
        init_from='scratch',
        max_iters=UPDATES,
        lr_decay_iters=UPDATES,
        warmup_iters=WARMUP_UPDATES,
        temporal_memory_gate_init=GATE_INITIALIZATIONS[name],
        eval_interval=500,
        eval_iters=16,
        keep_checkpoints=True,
        checkpoint_steps=list(CHECKPOINT_STEPS),
    )
    return deepcopy(config)
