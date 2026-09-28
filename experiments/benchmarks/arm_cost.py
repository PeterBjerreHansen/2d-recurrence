"""Same-host training and live-inference cost of the four 20B arms.

All arms run in one process on one device, so host CPU, GPU and software are
held fixed. For every ``(U_T, U_D)`` pair an arm can train on, one training
microbatch (forward and backward, synthetic rows, BF16 autocast on CUDA as in
training) is timed; arms and pairs are interleaved round-robin so drift hits all
arms alike. Median pair times are combined with each arm's frozen update-count
distribution into expected seconds per optimizer update, for the late phase and
for the whole-run curriculum (phases weighted by their update counts). Live
decoding is timed per token for the deployed settings.

Timing covers model compute only: no data loading, logging, evaluation or
checkpointing. It measures relative cost between arms, not end-to-end runtime.

Run from the repository root:
    python -m experiments.benchmarks.arm_cost --device cuda --output .../arm_cost.json
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib import import_module
import json
from pathlib import Path
import pickle
import random
import statistics
import time

import torch

from evaluation.recurrence_grid import compute_estimate
from inference.live import create_live_state, decode_live_step
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import RecurrenceSchedule, sample_schedule, update_probabilities_at_step
from training_utils import provenance

study = import_module('experiments.long_runs.20B_recurrence.study')

LIVE_SETTINGS = (('transformer', 1, 'ordinary'), ('temporal', 1, 'final_depth'),
                 ('hybrid', 1, 'depth_specialized'), ('hybrid', 4, 'depth_specialized'),
                 ('depth', 4, 'depth_specialized'))


def build_model(config, vocab_size, device):
    args = {key: config[key] for key in ('n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout')}
    args['vocab_size'] = vocab_size
    if config['architecture'] == 'recurrent':
        args.update({key: config[key] for key in ('n_prelude', 'n_buffer', 'n_core', 'n_source', 'n_coda',
                                                  'recurrence_mode', 'temporal_memory_gate_init')})
        return Recurrent2DGPT(RecurrentGPTConfig(**args)).to(device)
    return GPT(GPTConfig(**args)).to(device)


def phase_distributions(config):
    """(update weight, {pair: probability}) for each curriculum phase of an arm."""
    if config['architecture'] != 'recurrent':
        return [(study.UPDATES, {(0, 0): 1.0})]
    support = config['update_support']
    starts = list(study.CURRICULUM_STARTS) + [study.UPDATES]
    phases = []
    for start, stop in zip(starts, starts[1:]):
        matrix = update_probabilities_at_step(config, start)
        phases.append((stop - start, {(u_t, u_d): p for u_t, row in zip(support, matrix)
                                      for u_d, p in zip(support, row) if p}))
    return phases


def synchronize(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elif device.startswith('mps'):
        torch.mps.synchronize()


def time_microbatch(model, x, y, pair, rng, device, context):
    schedule = sample_schedule(*pair, rng) if isinstance(model, Recurrent2DGPT) else None
    synchronize(device)
    started = time.perf_counter()
    with context():
        _, loss = model(x, y, schedule=schedule) if schedule is not None else model(x, y)
    loss.backward()
    synchronize(device)
    return time.perf_counter() - started, schedule


def time_optimizer_step(model, optimizer, grad_clip, device):
    synchronize(device)
    started = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    return time.perf_counter() - started


@torch.no_grad()
def time_live(model, depth_steps, strategy, tokens, device):
    state = create_live_state(model, depth_steps if isinstance(model, Recurrent2DGPT) else None,
                              strategy if isinstance(model, Recurrent2DGPT) else None)
    synchronize(device)
    started = time.perf_counter()
    for token in tokens:
        decode_live_step(model, token[None], state)
    synchronize(device)
    return (time.perf_counter() - started) / len(tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rounds', type=int, default=5, help='Timed round-robin rounds per pair')
    parser.add_argument('--warmup-rounds', type=int, default=1)
    parser.add_argument('--batch-size', type=int, help='Microbatch rows; defaults to the frozen 5')
    parser.add_argument('--live-tokens', type=int, default=256)
    parser.add_argument('--live-rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output exists; choose a new path')
    torch.manual_seed(args.seed)
    device = args.device
    vocab_size = pickle.loads(Path('data/chess_v1/meta.pkl').read_bytes())['vocab_size']
    configs = {arm: study.run_config(arm) for arm in study.ARM_ORDER}
    first = configs['hybrid']
    batch_size = args.batch_size or first['batch_size']
    accumulation = first['gradient_accumulation_steps']
    length = first['block_size']
    bf16 = device.startswith('cuda') and first['dtype'] == 'bfloat16'
    context = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if bf16 else nullcontext

    models, optimizers, pairs, phases = {}, {}, {}, {}
    for arm, config in configs.items():
        models[arm] = build_model(config, vocab_size, device).train()
        optimizers[arm] = models[arm].configure_optimizers(
            config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']),
            'cuda' if device.startswith('cuda') else device)
        phases[arm] = phase_distributions(config)
        pairs[arm] = sorted({pair for _, distribution in phases[arm] for pair in distribution})
    x = torch.randint(vocab_size, (batch_size, length), device=device)
    y = torch.randint(vocab_size, (batch_size, length), device=device)
    rng = random.Random(args.seed)
    times = {arm: {pair: [] for pair in pairs[arm]} for arm in configs}
    optimizer_times = {arm: [] for arm in configs}
    flops = {arm: {} for arm in configs}
    peak = {arm: {} for arm in configs}
    for round_index in range(args.warmup_rounds + args.rounds):
        timed = round_index >= args.warmup_rounds
        for arm in configs:
            for pair in pairs[arm]:
                if device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats()
                seconds, schedule = time_microbatch(models[arm], x, y, pair, rng, device, context)
                if timed:
                    times[arm][pair].append(seconds)
                    if device.startswith('cuda'):
                        peak[arm][pair] = max(peak[arm].get(pair, 0), torch.cuda.max_memory_allocated())
                    if schedule is not None:
                        flops[arm].setdefault(pair, []).append(compute_estimate(
                            models[arm].config, schedule, length)['estimated_forward_matmul_flops_per_sequence'])
            seconds = time_optimizer_step(models[arm], optimizers[arm], configs[arm]['grad_clip'], device)
            if timed:
                optimizer_times[arm].append(seconds)
        print(f'round {round_index + 1}/{args.warmup_rounds + args.rounds} done', flush=True)

    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), device=device,
                  device_name=torch.cuda.get_device_name(0) if device.startswith('cuda') else device,
                  bf16_autocast=bf16, microbatch_rows=batch_size, accumulation=accumulation,
                  context_length=length, rounds=args.rounds, warmup_rounds=args.warmup_rounds,
                  provenance=provenance(), arms={})
    for arm in configs:
        median = {pair: statistics.median(values) for pair, values in times[arm].items()}
        step = statistics.median(optimizer_times[arm])

        def expected(distribution):
            return accumulation * sum(p * median[pair] for pair, p in distribution.items()) + step

        total_weight = sum(weight for weight, _ in phases[arm])
        report['arms'][arm] = dict(
            pair_microbatch_seconds={f'{u_t},{u_d}': value for (u_t, u_d), value in median.items()},
            pair_peak_allocated_bytes={f'{u_t},{u_d}': value for (u_t, u_d), value in peak[arm].items()},
            pair_forward_matmul_flops_per_sequence={f'{u_t},{u_d}': statistics.mean(value)
                                                    for (u_t, u_d), value in flops[arm].items()},
            optimizer_step_seconds=step,
            late_phase_seconds_per_update=expected(phases[arm][-1][1]),
            curriculum_seconds_per_update=sum(weight * expected(distribution)
                                              for weight, distribution in phases[arm]) / total_weight)

    tokens = torch.randint(vocab_size, (args.live_tokens,), device=device)
    report['live_seconds_per_token'] = {}
    for model in models.values():
        model.eval()
    samples = {setting: [] for setting in LIVE_SETTINGS}
    for _ in range(args.live_rounds + 1):
        for setting in LIVE_SETTINGS:
            arm, depth_steps, strategy = setting
            samples[setting].append(time_live(models[arm], depth_steps, strategy, tokens, device))
    for (arm, depth_steps, strategy), values in samples.items():
        report['live_seconds_per_token'][f'{arm} J={depth_steps}'] = statistics.median(values[1:])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    base = report['arms']['temporal']['late_phase_seconds_per_update']
    print(f"{'arm':<12}{'late s/update':>15}{'curriculum s/update':>22}{'late vs temporal':>18}")
    for arm, item in report['arms'].items():
        print(f"{arm:<12}{item['late_phase_seconds_per_update']:>15.3f}"
              f"{item['curriculum_seconds_per_update']:>22.3f}"
              f"{item['late_phase_seconds_per_update'] / base:>18.3f}")
    for name, value in report['live_seconds_per_token'].items():
        print(f'live {name}: {1000 * value:.2f} ms/token')


if __name__ == '__main__':
    main()
