"""Explicit prepare/benchmark/train/evaluate/choose commands; never provisions a VM."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

import torch

from data_loader import ChessData, file_hash
from evaluation.all_rows import load_model, evaluate_baseline
from evaluation.panels import load_panel, fixed_panel_batches
from evaluation.recurrence_grid import evaluate_grid
from recurrence.schedule import probability_map_at_step, probabilities_at_step
from experiments.serious import ROOT, REVISION, COMMON, TOKENS_PER_UPDATE, ablation, base, long_run
from train import train

RESULTS = ROOT / 'results'


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'{path} exists with different contents; use a new experiment')
        return
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_hashes():
    files = subprocess.check_output(
        ['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z'], text=True).split('\0')
    return {name: file_hash(name) for name in sorted(set(files))
            if name and Path(name).is_file() and
            (name.endswith('.py') or name in ('pyproject.toml', 'uv.lock'))}


def environment():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Serious runs require one CUDA GPU')
    if 'RTX A6000' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This protocol is frozen for an RTX A6000')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is required by the serious profile')
    return dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                python=sys.version, source_hashes=source_hashes(), common=COMMON)


def preflight():
    receipt = environment()
    data = ChessData('data/chess_8M_v1', 1023)
    source = data.manifest['source']
    if (source.get('dataset') != 'adamkarvonen/chess_games' or
            source.get('file') != 'lichess_6gb_blocks.zip' or source.get('revision') != REVISION or
            source.get('max_rows') is not None or data.manifest['split_seed'] != 2357 or
            data.manifest['validation_fraction'] != .01):
        raise ValueError('Expected the complete pinned 6GB Lichess dataset and reference split')
    receipt['manifest_hash'] = data.manifest_hash
    count = len(data.rows['val'])
    if count <= 768:
        raise ValueError('Full dataset must have enough validation rows for independent panels')
    selection = sorted(random.Random(2027).sample(range(count), 256))
    selected = set(selection)
    write_once(RESULTS / 'panel.json', dict(dataset_manifest_hash=data.manifest_hash,
        validation_row_count=count, selection_seed=2027, selection_indices=selection,
        confirmation_indices=[i for i in range(count) if i not in selected]))
    receipt['panel_hash'] = file_hash(RESULTS / 'panel.json')
    write_once(RESULTS / 'environment.json', receipt)
    return receipt


def verify_environment():
    saved = json.loads((RESULTS / 'environment.json').read_text())
    current = environment()
    if any(saved[k] != v for k, v in current.items()):
        raise ValueError('Frozen source, runtime or hardware changed')
    if file_hash(RESULTS / 'panel.json') != saved['panel_hash']:
        raise ValueError('Frozen panel changed')
    if file_hash(Path('data/chess_8M_v1/manifest.json')) != saved['manifest_hash']:
        raise ValueError('Dataset manifest changed')
    return saved


def benchmark(variant, steps):
    verify_environment()
    if steps < 30:
        raise ValueError('Benchmark requires at least 30 updates (first 10 excluded)')
    config = base('baseline') if variant == 'transformer' else ablation(variant)
    config.update(out_dir=str(RESULTS / 'benchmarks' / variant), max_iters=steps,
                  training_budget_seconds=0.0, warmup_iters=10, lr_decay_iters=10000,
                  keep_checkpoints=False, checkpoint_steps=None, eval_interval=steps,
                  log_interval=1)
    if Path(config['out_dir']).exists():
        raise FileExistsError('Benchmark output exists; do not mix repeated measurements')
    torch.cuda.reset_peak_memory_stats()
    train(config)
    rows = [json.loads(line) for line in (Path(config['out_dir']) / 'metrics.jsonl').read_text().splitlines()]
    seconds = [r['seconds'] for r in rows if r['event'] == 'train' and r['step'] > 10]
    write_once(RESULTS / f'benchmark-{variant}.json', dict(variant=variant, config=config,
        measured_updates=len(seconds), mean_seconds=statistics.mean(seconds),
        median_seconds=statistics.median(seconds), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        characters_per_second=TOKENS_PER_UPDATE / statistics.mean(seconds),
        environment_hash=file_hash(RESULTS / 'environment.json')))


def freeze(reference_characters):
    verify_environment()
    if reference_characters < 100_000_000:
        raise ValueError('Use at least 100M reference characters for the supervision ablation')
    benchmarks = {}
    for variant in ('final', 'deep'):
        benchmarks[variant] = json.loads((RESULTS / f'benchmark-{variant}.json').read_text())
        if benchmarks[variant]['environment_hash'] != file_hash(RESULTS / 'environment.json'):
            raise ValueError('Benchmark environment differs')
    steps = math.ceil(reference_characters / TOKENS_PER_UPDATE)
    budget = steps * benchmarks['final']['mean_seconds']
    configs = {v: {**ablation(v), 'training_budget_seconds': budget} for v in ('final', 'deep')}
    write_once(RESULTS / 'protocol.json', dict(reference_characters=reference_characters,
        reference_updates=steps, budget_seconds=budget, configs=configs,
        benchmark_hashes={v: file_hash(RESULTS / f'benchmark-{v}.json') for v in benchmarks},
        environment_hash=file_hash(RESULTS / 'environment.json')))
    print(f'Frozen: {budget / 3600:.2f} training hours per arm; deep overhead '
          f"{benchmarks['deep']['mean_seconds'] / benchmarks['final']['mean_seconds']:.3f}x")


def execute(config):
    verify_environment()
    out = Path(config['out_dir'])
    write_once(out / 'plan.json', dict(config=config,
        environment_hash=file_hash(RESULTS / 'environment.json')))
    latest = out / 'ckpt.pt'
    if latest.exists():
        config = {**config, 'init_from': 'resume'}
    return train(config)


def ablation_config(variant):
    protocol = json.loads((RESULTS / 'protocol.json').read_text())
    if protocol['environment_hash'] != file_hash(RESULTS / 'environment.json'):
        raise ValueError('Ablation protocol environment differs')
    return protocol['configs'][variant]


def evaluate(path, split):
    verify_environment()
    output = path.parent / f'evaluation-{split}.json'
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved['checkpoint_hash'] != file_hash(path) or
                saved['panel_hash'] != file_hash(RESULTS / 'panel.json')):
            raise ValueError(f'{output} describes a different checkpoint/panel; preserve it under another name')
        print(f'Reusing verified {output}')
        return
    checkpoint, digest, data, model, recurrent = load_model(path, 'cuda')
    panel = load_panel(RESULTS / 'panel.json', data, split)
    if split == 'confirmation':
        # A fixed, disjoint 512-row sample, not the entire 6GB-corpus validation split.
        panel['row_indices'] = sorted(random.Random(2028).sample(panel['row_indices'], 512))
    batches, metadata = fixed_panel_batches(data, panel, 5)
    if recurrent:
        active_matrix = probabilities_at_step(checkpoint['config'], checkpoint['iter_num'])
        metrics = evaluate_grid(model, data, fixed_batches=batches, data_seed=None, batch_size=5,
            mask_seeds=[11, 23, 37], training_probabilities=probability_map_at_step(
                checkpoint['config'], checkpoint['iter_num']),
            diagnostics=False, sampling=metadata['sampling'])
        metrics['training_probability_step'] = checkpoint['iter_num']
        metrics['training_probability_matrix'] = [list(row) for row in active_matrix]
    else:
        metrics = evaluate_baseline(model, batches, 'cuda')
    write_once(output, dict(checkpoint_hash=digest, step=checkpoint['iter_num'],
        training_seconds=checkpoint['training_seconds'],
        characters=checkpoint['iter_num'] * TOKENS_PER_UPDATE,
        manifest_hash=data.manifest_hash, panel_hash=file_hash(RESULTS / 'panel.json'),
        split=split, fixed_batches=metadata, device='cuda', dtype='float32',
        metrics=metrics))
    print(output)


def choose(mode, reason):
    verify_environment()
    if not reason.strip():
        raise ValueError('Record why this supervision mode was selected')
    hashes = {}
    budget = json.loads((RESULTS / 'protocol.json').read_text())['budget_seconds']
    for variant in ('final', 'deep'):
        path = RESULTS / variant / 'ckpt.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['training_seconds'] < budget:
            raise ValueError(f'{variant} has not finished its time budget')
        report_path = path.parent / 'evaluation-selection.json'
        report = json.loads(report_path.read_text())
        if report['checkpoint_hash'] != file_hash(path):
            raise ValueError('Evaluate the final checkpoint before choosing')
        hashes[variant] = file_hash(report_path)
    write_once(RESULTS / 'decision.json', dict(mode=mode, reason=reason, reports=hashes,
        protocol_hash=file_hash(RESULTS / 'protocol.json')))
    print('Supervision locked. The 1B pair is ready; nothing was launched automatically.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight')
    p = sub.add_parser('benchmark')
    p.add_argument('variant', choices=['transformer', 'final', 'deep', 'deep_more'])
    p.add_argument('--steps', type=int, default=60)
    p = sub.add_parser('freeze')
    p.add_argument('--reference-characters', type=int, default=250_000_000)
    p = sub.add_parser('ablation')
    p.add_argument('variant', choices=['final', 'deep'])
    p = sub.add_parser('evaluate')
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--split', choices=['selection', 'confirmation'], default='selection')
    p = sub.add_parser('choose')
    p.add_argument('mode', choices=['final', 'deep'])
    p.add_argument('--reason', required=True)
    p = sub.add_parser('pair')
    p.add_argument('--billions', type=int, choices=[1, 64], default=1)
    args = parser.parse_args()
    if args.command == 'preflight':
        preflight()
    elif args.command == 'benchmark':
        benchmark(args.variant, args.steps)
    elif args.command == 'freeze':
        freeze(args.reference_characters)
    elif args.command == 'ablation':
        execute(ablation_config(args.variant))
    elif args.command == 'evaluate':
        evaluate(args.checkpoint, args.split)
    elif args.command == 'choose':
        choose(args.mode, args.reason)
    elif args.command == 'pair':
        # Explicit foreground queue. Re-running resumes the interrupted member.
        for model in ('transformer', 'recurrent_a'):
            checkpoint = execute(long_run(model, args.billions * 10**9))
            evaluate(checkpoint, 'selection')


if __name__ == '__main__':
    main()
