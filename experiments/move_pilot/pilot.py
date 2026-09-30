"""Pilot runs for the move-target study (docs/engine_policy_plan.md, Pilot).

    uv run python -m experiments.move_pilot.pilot list
    uv run python -m experiments.move_pilot.pilot run lr_legal_3e-4
    uv run python -m experiments.move_pilot.pilot run legal_hybrid
    uv run python -m experiments.move_pilot.pilot run engine_hybrid_from_legal

Runs:
- ``{human,legal}_{arm}``: stage 1 from scratch, one pass over the stage-1 dataset;
  ``legal_hybrid_seed2`` repeats one arm with another initialisation.
- ``engine_{arm}_from_{human,legal}``: stage 2, continued from that stage-1 run on
  ``STAGE2_POSITIONS`` Leela positions, with a fresh output layer.
- ``lr_{objective}_{rate}``: the learning-rate check on the transformer, a fifth of
  a run's length (the engine check continues from ``legal_transformer``).

Every run reads the same rows in the same order; seeds change only the initial
weights. A run resumes from its ``ckpt.pt`` when one exists.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import torch

from recurrence.schedule import build_update_probability_matrix
from train import DEFAULTS, train

ROOT = Path('experiments/move_pilot/results')
# Dataset names, under data/ (as the trainer resolves them).
STAGE1 = 'stage1_v1'
LEELA = 'leela_v1'
STAGE2_POSITIONS = 50_000_000
ARMS = ('transformer', 'temporal', 'depth', 'hybrid')
SEEDS = (1337, 2024)

# 400 rows of 256 tokens per update (~100k tokens, as the 20B study's 100 x 1,023 characters),
# in micro-batches of 20 rows (~5k tokens, the 20B micro-batch).
BATCH_SIZE = 20
ACCUMULATION = 20
ROWS_PER_UPDATE = BATCH_SIZE * ACCUMULATION

MODEL = dict(n_layer=8, n_head=8, n_embd=512, bias=False, dropout=0.0,
             n_prelude=1, n_buffer=1, n_core=4, n_source=1, n_coda=1)
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_FRACTION = 0.03
DECAY_FRACTION = 0.10      # the last tenth of updates decays linearly, as in the 20B study
LR_CHECK_RATES = ('1e-4', '3e-4', '1e-3')
LR_CHECK_FRACTION = 0.2

# Gap-free and broad throughout; deepen by moving the centre, never by concentrating the mass
# (20B report). Probabilities of max(U_T, U_D) = 0, 1, 2, 3 from each fraction of the run on;
# the last change is well before the decay.
UPDATE_SUPPORT = [0, 1, 2, 3]
CURRICULUM = ((0.00, (0.25, 0.35, 0.25, 0.15)),
              (0.25, (0.15, 0.30, 0.30, 0.25)),
              (0.50, (0.10, 0.25, 0.35, 0.30)))
HYBRID_DIAGONAL_MASS = 0.80
TEMPORAL_GATE_INIT = 0.10
# Temporal only, in the decay phase: a quarter of each update's micro-batches start from settled
# temporal memory, the memory live execution uses (20B report, alignment during the decay).
WARM_START = dict(warm_start_fraction=0.25, warm_start_max_passes=64, warm_start_tolerance=0.01)

ARCHITECTURES = {
    'transformer': dict(architecture='baseline', update_support=[], update_probabilities=[],
                        eval_u_t=0, eval_u_d=0),
    'temporal': dict(architecture='recurrent', recurrence_mode='temporal', eval_u_t=3, eval_u_d=0),
    'depth': dict(architecture='recurrent', recurrence_mode='depth', eval_u_t=0, eval_u_d=3),
    'hybrid': dict(architecture='recurrent', recurrence_mode='hybrid', eval_u_t=3, eval_u_d=3),
}


def run_names():
    names = [f'{objective}_{arm}' for objective in ('human', 'legal') for arm in ARMS]
    names.append('legal_hybrid_seed2')
    names += [f'engine_{arm}_from_{objective}' for objective in ('legal', 'human') for arm in ARMS]
    names += [f'lr_{objective}_{rate}' for objective in ('human', 'legal', 'engine') for rate in LR_CHECK_RATES]
    return names


def updates_for(dataset, positions=None):
    """Updates for one pass over the training rows, or for about ``positions`` supervised positions."""
    train_split = json.loads((Path('data') / dataset / 'dataset.json').read_text())['splits']['train']
    rows = train_split['rows'] if positions is None else train_split['rows'] * positions / train_split['positions']
    return math.ceil(rows / ROWS_PER_UPDATE)


def _matrix(mode, probabilities):
    return build_update_probability_matrix(UPDATE_SUPPORT, mode, probabilities,
                                           HYBRID_DIAGONAL_MASS if mode == 'hybrid' else None)


def _config(arm, objective, dataset, updates, out_dir, *, seed=SEEDS[0], curriculum=True, learning_rate=LEARNING_RATE):
    config = deepcopy(DEFAULTS)
    decay_start = updates - round(DECAY_FRACTION * updates)
    config.update(**MODEL, **ARCHITECTURES[arm],
                  data_format='moves', objective=objective, dataset=str(dataset), block_size=256,
                  out_dir=str(out_dir), init_from='scratch', seed=seed,
                  batch_size=BATCH_SIZE, gradient_accumulation_steps=ACCUMULATION,
                  max_iters=updates, learning_rate=learning_rate, min_lr=MIN_LEARNING_RATE * learning_rate / LEARNING_RATE,
                  lr_schedule='wsd', warmup_iters=max(1, round(WARMUP_FRACTION * updates)),
                  lr_decay_start=decay_start, lr_decay_iters=updates,
                  device='cuda', dtype='bfloat16', compile=False,
                  eval_interval=500, eval_iters=50, log_interval=10, live_eval_iters=5,
                  checkpoint_interval=500, keep_checkpoints=True, checkpoint_steps=[decay_start, updates])
    mode = config.get('recurrence_mode')
    if config['architecture'] == 'recurrent':
        config.update(update_support=list(UPDATE_SUPPORT), temporal_memory_gate_init=TEMPORAL_GATE_INIT)
        if curriculum:
            config.update(update_probabilities=[], update_probability_schedule=dict(
                type='piecewise_constant',
                phases=[dict(start_step=round(fraction * updates), update_probabilities=_matrix(mode, probabilities))
                        for fraction, probabilities in CURRICULUM]))
        else:
            # A continuation keeps the final mixture of stage 1 throughout.
            config.update(update_probabilities=_matrix(mode, CURRICULUM[-1][1]), update_probability_schedule=None)
    return config


def run_config(name):
    """The resolved training configuration of one pilot run."""
    if name not in run_names():
        raise ValueError(f'Unknown pilot run: {name}')
    out_dir = ROOT / name
    parts = name.split('_')
    if parts[0] == 'lr':
        objective, rate = parts[1], parts[2]
        if objective == 'engine':
            config = _config('transformer', 'engine', LEELA,
                             round(LR_CHECK_FRACTION * updates_for(LEELA, STAGE2_POSITIONS)), out_dir,
                             curriculum=False, learning_rate=float(rate))
            config.update(init_from='continue', continue_from=str(ROOT / 'legal_transformer' / 'ckpt.pt'))
            return config
        return _config('transformer', objective, STAGE1, round(LR_CHECK_FRACTION * updates_for(STAGE1)), out_dir,
                       learning_rate=float(rate))
    if parts[0] == 'engine':
        arm, source = parts[1], parts[3]
        config = _config(arm, 'engine', LEELA, updates_for(LEELA, STAGE2_POSITIONS), out_dir, curriculum=False)
        config.update(init_from='continue', continue_from=str(ROOT / f'{source}_{arm}' / 'ckpt.pt'))
        return config
    objective, arm = parts[0], parts[1]
    seed = SEEDS[1] if name.endswith('_seed2') else SEEDS[0]
    return _config(arm, objective, STAGE1, updates_for(STAGE1), out_dir, seed=seed)


def run(name, **overrides):
    """Train one pilot run, resuming it if it has a checkpoint."""
    config = {**run_config(name), **overrides}
    checkpoint = Path(config['out_dir']) / 'ckpt.pt'
    step = 0
    if checkpoint.exists():
        step = torch.load(checkpoint, map_location='cpu', weights_only=False)['iter_num']
        config['init_from'] = 'resume'
    if config.get('recurrence_mode') == 'temporal' and config['architecture'] == 'recurrent':
        # Warm-start batches in the decay phase only: train to its start, then resume with them.
        if step < config['lr_decay_start']:
            train({**config, 'max_iters': config['lr_decay_start']})
            config['init_from'] = 'resume'
        config.update(WARM_START)
    return train(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    runner = commands.add_parser('run')
    runner.add_argument('name', choices=run_names())
    arguments = parser.parse_args()
    if arguments.command == 'list':
        print('\n'.join(run_names()))
    else:
        print(run(arguments.name))


if __name__ == '__main__':
    main()
