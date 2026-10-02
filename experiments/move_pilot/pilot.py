"""Pilot runs for the move-target study (docs/engine_policy_plan.md, Pilot).

    uv run python -m experiments.move_pilot.pilot list
    uv run python -m experiments.move_pilot.pilot run lr_legal_3e-4
    uv run python -m experiments.move_pilot.pilot queue legal_transformer legal_temporal legal_depth \
        legal_hybrid legal_hybrid_seed2 --slots 4 --threads 3
    uv run python -m experiments.move_pilot.pilot bench legal_transformer legal_temporal legal_depth \
        legal_hybrid --micro-batches 20 50 100 --updates 40 --threads 3

``queue`` runs several pilot runs side by side on one GPU, starting the next as a
slot frees up; each run logs to ``results/<name>/console.log``. ``bench`` runs the
given runs side by side for a few updates at each micro-batch size, with stage 1's
final (deepest) update mixture and no live evaluation, and prints seconds per update
and peak memory. The micro-batch size only trades speed for memory: every update is
400 rows either way.

Runs:
- ``legal_{arm}``: stage 1 from scratch, one pass over the stage-1 dataset;
  ``legal_hybrid_seed2`` repeats one arm with another initialisation.
- ``engine_{arm}``: stage 2, continued from ``legal_{arm}`` on ``STAGE2_POSITIONS``
  Leela positions, with a fresh output layer; ``engine_long_{arm}`` runs
  ``ENGINE_LONG_PASSES`` passes over the whole Leela dataset, evaluating every 400 updates.
- ``full_legal_{arm}``, ``full_engine_{arm}``: the full stage-1 run on ``STAGE1_FULL``, then stage 2,
  one pass over ``LEELA_FULL`` continued from its final model.
- ``lr_{objective}_{rate}``: the learning-rate check on the transformer, a fifth of
  a run's length (the engine check continues from ``legal_transformer``);
  ``lr_legal_{arm}_{rate}`` checks a recurrent arm, whose shared core weights may
  want a different rate.

Every run reads the same rows in the same order; seeds change only the initial
weights. A run resumes from its ``ckpt.pt`` when one exists.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from recurrence.schedule import build_update_probability_matrix
from train import DEFAULTS, train

ROOT = Path('experiments/move_pilot/results')
# Dataset names, under data/ (as the trainer resolves them).
STAGE1 = 'stage1_v1'
STAGE1_FULL = 'stage1_full_v1'
LEELA = 'leela_v1'
LEELA_FULL = 'leela_full_v1'
STAGE2_POSITIONS = 50_000_000
ENGINE_LONG_PASSES = 2
ARMS = ('transformer', 'temporal', 'depth', 'hybrid')
SEEDS = (1337, 2024)

# 400 rows of 256 tokens per update (~100k tokens, as the 20B study's 100 x 1,023 characters),
# accumulated over micro-batches whose size only affects speed and memory.
ROWS_PER_UPDATE = 400
MICRO_BATCH = 50   # measured on an A100: 20 is 10-25% slower, 100 and above barely faster
# Evaluation populations, in dev games, whatever the micro-batch size.
EVAL_GAMES = 1000
LIVE_GAMES = 100

MODEL = dict(n_layer=8, n_head=8, n_embd=512, bias=False, dropout=0.0,
             n_prelude=1, n_buffer=1, n_core=4, n_source=1, n_coda=1)
# Every arm trains at 1e-3 in both stages: best for the transformer in the short checks, and better than
# 3e-4 over full stage-1 runs of temporal and depth, although the short checks favoured 3e-4 for them
# (docs/engine_policy_plan.md, Decisions). The rate decays to a tenth of the peak, as in the 20B study.
LEARNING_RATES = dict.fromkeys(ARMS, 1e-3)
MIN_RATE_FRACTION = 0.1
WARMUP_FRACTION = 0.03
DECAY_FRACTION = 0.10      # the last tenth of updates decays linearly, as in the 20B study
LR_CHECK_RATES = ('1e-4', '3e-4', '1e-3', '3e-3')
LR_CHECK_FRACTION = 0.2

# Gap-free and broad throughout; deepen by moving the centre, never by concentrating the mass
# (20B report). Probabilities of max(U_T, U_D) = 0, 1, 2, 3 from each fraction of the run on;
# the last change is well before the decay.
UPDATE_SUPPORT = [0, 1, 2, 3]
CURRICULUM = ((0.00, (0.25, 0.35, 0.25, 0.15)),
              (0.25, (0.15, 0.30, 0.30, 0.25)),
              (0.50, (0.10, 0.25, 0.35, 0.30)))
PILOT_SCHEDULE = (UPDATE_SUPPORT, CURRICULUM)
# Full stage-1 runs start shallower and end deeper than the pilot, up to update count 4 (five passes),
# and stop sampling U=0 after the first quarter: these are multi-pass models. The pilot's recurrent arms
# left the early plateau late, possibly from unrolling 3-4 passes from the first update. Phases are
# fractions of the run, so the longer run also deepens later in updates.
FULL_SCHEDULE = ([0, 1, 2, 3, 4], ((0.00, (0.50, 0.35, 0.15, 0.00, 0.00)),
                                   (0.25, (0.00, 0.55, 0.30, 0.15, 0.00)),
                                   (0.50, (0.00, 0.30, 0.30, 0.25, 0.15)),
                                   (0.75, (0.00, 0.15, 0.20, 0.30, 0.35))))
HYBRID_DIAGONAL_MASS = 0.80
TEMPORAL_GATE_INIT = 0.10
# Temporal only, in the decay phase: a quarter of each update's micro-batches start from settled
# temporal memory, the memory live execution uses (20B report, alignment during the decay).
WARM_START = dict(warm_start_fraction=0.25, warm_start_max_passes=64, warm_start_tolerance=0.01)

ARCHITECTURES = {
    'transformer': dict(architecture='baseline', update_support=[], update_probabilities=[],
                        eval_u_t=0, eval_u_d=0),
    # Recurrent arms are evaluated in the training graph at the schedule's largest update count.
    'temporal': dict(architecture='recurrent', recurrence_mode='temporal'),
    'depth': dict(architecture='recurrent', recurrence_mode='depth'),
    'hybrid': dict(architecture='recurrent', recurrence_mode='hybrid'),
}


def run_names():
    names = [f'legal_{arm}' for arm in ARMS] + ['legal_hybrid_seed2'] + [f'engine_{arm}' for arm in ARMS]
    names += [f'engine_long_{arm}' for arm in ARMS] + [f'full_legal_{arm}' for arm in ARMS]
    names += [f'full_engine_{arm}' for arm in ARMS]
    names += [f'lr_{objective}_{rate}' for objective in ('legal', 'engine') for rate in LR_CHECK_RATES]
    names += [f'lr_legal_{arm}_{rate}' for arm in ARMS[1:] for rate in LR_CHECK_RATES]
    return names


def updates_for(dataset, positions=None):
    """Updates for one pass over the training rows, or for about ``positions`` supervised positions."""
    train_split = json.loads((Path('data') / dataset / 'dataset.json').read_text())['splits']['train']
    rows = train_split['rows'] if positions is None else train_split['rows'] * positions / train_split['positions']
    return math.ceil(rows / ROWS_PER_UPDATE)


def _matrix(mode, probabilities, support=UPDATE_SUPPORT):
    return build_update_probability_matrix(support, mode, probabilities,
                                           HYBRID_DIAGONAL_MASS if mode == 'hybrid' else None)


def _config(arm, objective, dataset, updates, out_dir, *, seed=SEEDS[0], curriculum=True, learning_rate=None,
            micro_batch=MICRO_BATCH, schedule=PILOT_SCHEDULE):
    if ROWS_PER_UPDATE % micro_batch:
        raise ValueError(f'The micro-batch size must divide {ROWS_PER_UPDATE}')
    accumulation = ROWS_PER_UPDATE // micro_batch
    if arm == 'temporal' and (WARM_START['warm_start_fraction'] * accumulation) % 1:
        raise ValueError(f'Temporal warm starts use a quarter of the micro-batches: {accumulation} is not divisible by 4')
    learning_rate = LEARNING_RATES[arm] if learning_rate is None else learning_rate
    config = deepcopy(DEFAULTS)
    decay_start = updates - round(DECAY_FRACTION * updates)
    config.update(**MODEL, **ARCHITECTURES[arm],
                  data_format='moves', objective=objective, dataset=str(dataset), block_size=256,
                  out_dir=str(out_dir), init_from='scratch', seed=seed,
                  batch_size=micro_batch, gradient_accumulation_steps=accumulation,
                  max_iters=updates, learning_rate=learning_rate, min_lr=MIN_RATE_FRACTION * learning_rate,
                  lr_schedule='wsd', warmup_iters=max(1, round(WARMUP_FRACTION * updates)),
                  lr_decay_start=decay_start, lr_decay_iters=updates,
                  device='cuda', dtype='bfloat16', compile=False,
                  # Measured on MPS: evaluation every 200 updates takes 1-3% of training time; the
                  # evaluation events log their seconds so this can be checked on the GPU. Every
                  # evaluation also saves the recovery checkpoint.
                  eval_interval=200, eval_iters=math.ceil(EVAL_GAMES / micro_batch), log_interval=10,
                  live_eval_batches=math.ceil(LIVE_GAMES / micro_batch),
                  keep_checkpoints=True, checkpoint_steps=[decay_start, updates])
    mode = config.get('recurrence_mode')
    if config['architecture'] == 'recurrent':
        support, phases = schedule
        config.update(update_support=list(support), temporal_memory_gate_init=TEMPORAL_GATE_INIT,
                      eval_u_t=max(support) if mode != 'depth' else 0, eval_u_d=max(support) if mode != 'temporal' else 0)
        if curriculum:
            config.update(update_probabilities=[], update_probability_schedule=dict(
                type='piecewise_constant',
                phases=[dict(start_step=round(fraction * updates), update_probabilities=_matrix(mode, probabilities, support))
                        for fraction, probabilities in phases]))
        else:
            # A continuation keeps the final mixture of stage 1 throughout.
            config.update(update_probabilities=_matrix(mode, phases[-1][1], support), update_probability_schedule=None)
    return config


def run_config(name, micro_batch=MICRO_BATCH):
    """The resolved training configuration of one pilot run."""
    if name not in run_names():
        raise ValueError(f'Unknown pilot run: {name}')
    out_dir = ROOT / name
    parts = name.split('_')
    if parts[0] == 'lr' and len(parts) == 4:
        arm, rate = parts[2], parts[3]
        return _config(arm, 'legal', STAGE1, round(LR_CHECK_FRACTION * updates_for(STAGE1)), out_dir,
                       learning_rate=float(rate), micro_batch=micro_batch)
    if parts[0] == 'lr':
        objective, rate = parts[1], parts[2]
        if objective == 'engine':
            config = _config('transformer', 'engine', LEELA,
                             round(LR_CHECK_FRACTION * updates_for(LEELA, STAGE2_POSITIONS)), out_dir,
                             curriculum=False, learning_rate=float(rate), micro_batch=micro_batch)
            config.update(init_from='continue', continue_from=str(ROOT / 'legal_transformer' / 'ckpt.pt'))
            return config
        return _config('transformer', objective, STAGE1, round(LR_CHECK_FRACTION * updates_for(STAGE1)), out_dir,
                       learning_rate=float(rate), micro_batch=micro_batch)
    if parts[0] == 'full' and parts[1] == 'engine':
        # Stage 2: one pass over the 400M-position Leela dataset, continued from the final full stage-1
        # model with a fresh output layer, at stage 1's final update mixture throughout.
        arm = parts[2]
        config = _config(arm, 'engine', LEELA_FULL, updates_for(LEELA_FULL), out_dir, curriculum=False,
                         micro_batch=micro_batch, schedule=FULL_SCHEDULE)
        config.update(init_from='continue', continue_from=str(ROOT / f'full_legal_{arm}' / 'ckpt.pt'),
                      eval_interval=500, checkpoint_interval=500, checkpoint_steps=[config['lr_decay_start']])
        if arm in ('depth', 'hybrid'):
            config['live_eval_depths'] = [1, 2, 4, 5]
        return config
    if parts[0] == 'full':
        arm = parts[2]
        updates = updates_for(STAGE1_FULL)
        config = _config(arm, 'legal', STAGE1_FULL, updates, out_dir, micro_batch=micro_batch, schedule=FULL_SCHEDULE)
        # Evaluation every 1,000 updates; a recovery checkpoint every 500, and the pre-decay checkpoint kept
        # (a stage-2 run may continue from it). Live decoding covers the schedule's deepest count (J=5).
        config.update(eval_interval=1000, checkpoint_interval=500, checkpoint_steps=[config['lr_decay_start']])
        if arm in ('depth', 'hybrid'):
            config['live_eval_depths'] = [1, 2, 4, 5]
        return config
    if parts[0] == 'engine' and parts[1] == 'long':
        arm = parts[2]
        config = _config(arm, 'engine', LEELA, ENGINE_LONG_PASSES * updates_for(LEELA), out_dir, curriculum=False,
                         micro_batch=micro_batch)
        config.update(init_from='continue', continue_from=str(ROOT / f'legal_{arm}' / 'ckpt.pt'), eval_interval=400)
        return config
    if parts[0] == 'engine':
        arm = parts[1]
        config = _config(arm, 'engine', LEELA, updates_for(LEELA, STAGE2_POSITIONS), out_dir, curriculum=False,
                         micro_batch=micro_batch)
        config.update(init_from='continue', continue_from=str(ROOT / f'legal_{arm}' / 'ckpt.pt'))
        return config
    objective, arm = parts[0], parts[1]
    seed = SEEDS[1] if name.endswith('_seed2') else SEEDS[0]
    return _config(arm, objective, STAGE1, updates_for(STAGE1), out_dir, seed=seed, micro_batch=micro_batch)


def run(name, micro_batch=MICRO_BATCH, **overrides):
    """Train one pilot run, resuming it if it has a checkpoint."""
    config = {**run_config(name, micro_batch), **overrides}
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


def bench_config(name, updates, micro_batch, out_dir):
    """A short copy of a run for timing: the deepest stage-1 mixture throughout, no live evaluation."""
    config = run_config(name, micro_batch)
    config.update(out_dir=str(out_dir), max_iters=updates, warmup_iters=1, lr_decay_start=updates - 2,
                  lr_decay_iters=updates, eval_interval=10**9, eval_iters=1, live_eval_batches=0, log_interval=1,
                  keep_checkpoints=False)
    if config['architecture'] == 'recurrent' and config['update_probability_schedule']:
        phases = config['update_probability_schedule']['phases']
        config.update(update_probabilities=phases[-1]['update_probabilities'], update_probability_schedule=None)
    return config


def bench_one(name, updates, micro_batch, out_dir, **overrides):
    train({**bench_config(name, updates, micro_batch, out_dir), **overrides})
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    (Path(out_dir) / 'peak_memory.json').write_text(json.dumps(dict(peak_memory_gb=peak)))


def bench_summary(out_dir, skip=3):
    """Median seconds per update after the first ``skip`` updates, and peak memory, of one bench run."""
    records = [json.loads(line) for line in (Path(out_dir) / 'metrics.jsonl').read_text().splitlines()]
    seconds = sorted(r['seconds'] for r in records if r['event'] == 'train' and r['step'] > skip)
    memory = json.loads((Path(out_dir) / 'peak_memory.json').read_text())['peak_memory_gb']
    return dict(seconds_per_update=seconds[len(seconds) // 2], peak_memory_gb=memory)


def _command(*arguments):
    return [sys.executable, '-m', 'experiments.move_pilot.pilot', *map(str, arguments)]


def _options(threads, device):
    return (['--threads', threads] if threads else []) + (['--device', device] if device else [])


def queue(names, slots, micro_batch=MICRO_BATCH, threads=None, device=None, poll_seconds=10):
    """Run ``names`` side by side, at most ``slots`` at a time; return each run's exit code."""
    pending, running, codes = list(names), {}, {}
    while pending or running:
        while pending and len(running) < slots:
            name = pending.pop(0)
            (ROOT / name).mkdir(parents=True, exist_ok=True)
            log = open(ROOT / name / 'console.log', 'a')
            command = _command('run', name, '--micro-batch', micro_batch, *_options(threads, device))
            running[name] = (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log)
            print(f'{time.strftime("%H:%M:%S")} started {name}', flush=True)
        for name, (process, log) in list(running.items()):
            if process.poll() is not None:
                log.close()
                codes[name] = process.returncode
                del running[name]
                print(f'{time.strftime("%H:%M:%S")} {name} finished with exit code {process.returncode}', flush=True)
        if running:
            time.sleep(poll_seconds)
    return codes


def bench(names, micro_batches, updates, threads=None, device=None):
    """Time ``names`` running side by side at each micro-batch size; print and return the results."""
    results = []
    for micro_batch in micro_batches:
        directories = {name: ROOT / 'bench' / f'micro{micro_batch}' / name for name in names}
        processes = {}
        for name, directory in directories.items():
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True)
            log = open(directory / 'console.log', 'w')
            command = _command('bench-one', name, '--updates', updates, '--micro-batch', micro_batch,
                               '--out-dir', directory, *_options(threads, device))
            processes[name] = (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log)
        for name, (process, log) in processes.items():
            process.wait()
            log.close()
            if process.returncode:
                results.append(dict(name=name, micro_batch=micro_batch, error=f'exit code {process.returncode}'))
                continue
            results.append(dict(name=name, micro_batch=micro_batch, **bench_summary(directories[name])))
    print(f'{len(names)} runs side by side, at the deepest update mixture')
    print(f"{'run':<22}{'micro':>6}{'s/update':>10}{'h/1k upd':>10}{'peak GB':>9}")
    for result in results:
        if 'error' in result:
            print(f"{result['name']:<22}{result['micro_batch']:>6}  {result['error']}")
            continue
        hours = result['seconds_per_update'] * 1000 / 3600
        memory = f"{result['peak_memory_gb']:.1f}" if result['peak_memory_gb'] is not None else '-'
        print(f"{result['name']:<22}{result['micro_batch']:>6}{result['seconds_per_update']:>10.2f}{hours:>10.2f}{memory:>9}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')

    def common(command):
        command.add_argument('--threads', type=int, help='CPU threads per run (default: the trainer default, 4)')
        command.add_argument('--device', help='Default: cuda')

    runner = commands.add_parser('run')
    runner.add_argument('name', choices=run_names())
    runner.add_argument('--micro-batch', type=int, default=MICRO_BATCH)
    common(runner)
    queuer = commands.add_parser('queue')
    queuer.add_argument('names', nargs='+', choices=run_names())
    queuer.add_argument('--slots', type=int, required=True, help='Runs at once')
    queuer.add_argument('--micro-batch', type=int, default=MICRO_BATCH)
    common(queuer)
    bencher = commands.add_parser('bench')
    bencher.add_argument('names', nargs='+', choices=run_names())
    bencher.add_argument('--micro-batches', type=int, nargs='+', default=[20, 50, 100])
    bencher.add_argument('--updates', type=int, default=40)
    common(bencher)
    single = commands.add_parser('bench-one')
    single.add_argument('name', choices=run_names())
    single.add_argument('--updates', type=int, required=True)
    single.add_argument('--micro-batch', type=int, required=True)
    single.add_argument('--out-dir', required=True)
    common(single)
    arguments = parser.parse_args()
    overrides = {}
    if getattr(arguments, 'threads', None):
        overrides['num_threads'] = arguments.threads
    if getattr(arguments, 'device', None):
        overrides['device'] = arguments.device
        overrides['dtype'] = 'bfloat16' if arguments.device == 'cuda' else 'float32'
    if arguments.command == 'list':
        print('\n'.join(run_names()))
    elif arguments.command == 'run':
        print(run(arguments.name, arguments.micro_batch, **overrides))
    elif arguments.command == 'queue':
        codes = queue(arguments.names, arguments.slots, arguments.micro_batch, arguments.threads, arguments.device)
        raise SystemExit(max(codes.values(), default=0))
    elif arguments.command == 'bench':
        bench(arguments.names, arguments.micro_batches, arguments.updates, arguments.threads, arguments.device)
    else:
        bench_one(arguments.name, arguments.updates, arguments.micro_batch, arguments.out_dir, **overrides)


if __name__ == '__main__':
    main()
