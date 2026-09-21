"""Explicit, restartable commands for the two-architecture CUDA experiment.

No provisioning, background scheduling, or automatic extra training runs.
Run from the repository root: python -m experiments.ablations.architecture_sites.run --help
"""
import argparse
import csv
import json
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time

import torch

from data_loader import ChessData, file_hash
from evaluation.panels import load_panel
from train import DEFAULTS, train

ROOT = Path('experiments/ablations/architecture_sites/results')
VARIANTS = ('separated', 'coincident')
STEPS = (1000, 2000, 5000, 8000, 10000)


def configuration(variant):
    namespace = runpy.run_path(f'experiments/ablations/architecture_sites/configs/{variant}.py')
    config = {key: namespace.get(key, value) for key, value in DEFAULTS.items()}
    config['out_dir'] = str(ROOT / variant)
    return config


def source_hashes():
    # Includes untracked implementation files; excludes datasets and output artifacts.
    files = list(Path('.').glob('*.py')) + [Path('pyproject.toml'), Path('uv.lock')]
    for directory in ('models', 'recurrence', 'evaluation', 'configs', 'experiments'):
        files.extend(p for p in Path(directory).rglob('*.py') if 'results' not in p.parts)
    return {str(path): file_hash(path) for path in sorted(files) if path.is_file()}


def freeze():
    """Validate the actual data and freeze implementation before either run."""
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires CUDA; CPU tests are separate')
    if torch.cuda.device_count() != 1 or 'RTX A6000' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected one RTX A6000; do not silently substitute hardware')
    data = ChessData('data/chess_143K_v1', 1023)
    panel = load_panel('experiments/ablations/architecture_sites/panel.json', data, split='selection')
    receipt = dict(files=source_hashes(), dataset_manifest_hash=data.manifest_hash,
                   panel_sha256=panel['sha256'], torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0), configs={v: configuration(v) for v in VARIANTS},
                   precision='float32 parameters/activations, trainer enables TF32; eager SDPA')
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise ValueError('Frozen protocol/environment changed; investigate before resuming')
    else:
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def train_until(variant, until):
    freeze()
    config = configuration(variant)
    checkpoint = Path(config['out_dir']) / 'ckpt.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        step = saved['iter_num']
        if step >= until:
            print(f'{variant}: already at step {step}; nothing to train')
            return
        config['init_from'] = 'resume'
        del saved
    config['max_iters'] = until
    started = time.monotonic()
    train(config)
    with (ROOT / 'segments.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(variant=variant, until=until,
                                     elapsed_seconds=time.monotonic() - started)) + '\n')


def report_path(variant, step, split='selection'):
    return ROOT / variant / f'grid-{split}-step{step:06d}.json'


def validate_report(report, checkpoint, step, split, receipt):
    if (report['checkpoint_sha256'] != file_hash(checkpoint) or
            report['checkpoint_step'] != step or report['panel_split'] != split or
            report['panel_file_sha256'] != receipt['panel_sha256'] or
            report['manifest_hash'] != receipt['dataset_manifest_hash'] or
            report['mask_seeds'] != [11, 23, 37] or report['device'] != 'cuda'):
        raise ValueError('Existing evaluation does not match this experiment')
    if {(c['u_t'], c['u_d']) for c in report['cells']} != {(t, d) for t in (0, 1, 3) for d in (0, 1, 3)}:
        raise ValueError('Incomplete recurrence grid')


def recorded_protocol():
    """Read-only analysis does not require the original GPU or source checkout."""
    return json.loads((ROOT / 'protocol.json').read_text())


def evaluate(variant, step, split):
    receipt = recorded_protocol()
    if split == 'confirmation':
        decision_path = ROOT / 'decision.json'
        if not decision_path.exists():
            decision_path = ROOT / 'analysis/decision.json'
        decision = json.loads(decision_path.read_text())
        if decision['selected'] != variant or step != 10000:
            raise ValueError('Confirmation is reserved for the selected final checkpoint')
    checkpoint = ROOT / variant / f'ckpt-step{step:06d}.pt'
    output = report_path(variant, step, split)
    if output.exists():
        validate_report(json.loads(output.read_text()), checkpoint, step, split, receipt)
        print(f'Reusing verified {output}')
        return
    freeze()  # New evaluation must still match the frozen training environment.
    subprocess.run([sys.executable, '-m', 'evaluation.recurrence_grid',
                    '--checkpoint', str(checkpoint), '--device', 'cuda',
                    '--panel-file', 'experiments/ablations/architecture_sites/panel.json', '--panel-split', split,
                    '--batch-size', '2', '--mask-seeds', '11', '23', '37',
                    '--output', str(output)], check=True)
    validate_report(json.loads(output.read_text()), checkpoint, step, split, receipt)


def training_rows(path):
    # On recovery the log can contain updates newer than the last durable checkpoint.
    # Latest completed occurrence wins, so replayed updates are not double-counted.
    rows = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # An interrupted append can leave an incomplete line.
        if row.get('event') == 'train':
            rows[row['step']] = row
    return rows


def benchmark():
    recorded_protocol()
    result = {}
    for variant in VARIANTS:
        rows = training_rows(ROOT / variant / 'metrics.jsonl')
        measured = [rows[i]['seconds'] for i in range(11, 101)]
        mean = statistics.mean(measured)
        result[variant] = dict(mean_seconds_per_update=mean,
                               median_seconds_per_update=statistics.median(measured),
                               estimated_remaining_training_hours=9900 * mean / 3600)
    result['note'] = 'Updates 11–100; excludes setup, evaluation, checkpoint I/O, and spot replay. Budget these separately.'
    analysis_path('benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def analysis_path(name):
    directory = ROOT / 'analysis'
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def summarize():
    receipt = recorded_protocol()
    rows, scores, fingerprints = [], {}, {}
    for variant in VARIANTS:
        timing = training_rows(ROOT / variant / 'metrics.jsonl')
        late = []
        for step in STEPS:
            checkpoint = ROOT / variant / f'ckpt-step{step:06d}.pt'
            report = json.loads(report_path(variant, step).read_text())
            validate_report(report, checkpoint, step, 'selection', receipt)
            fingerprint = report['batch_fingerprint']
            if fingerprints and fingerprint not in fingerprints:
                raise ValueError('Evaluation batches differ')
            fingerprints[fingerprint] = True
            cells = {(c['u_t'], c['u_d']): c for c in report['cells']}
            for (t, d), cell in cells.items():
                rows.append(dict(variant=variant, step=step, u_t=t, u_d=d,
                                 nll=cell['nll_mean'], accuracy=cell['accuracy_mean'],
                                 characters=step * 8184,
                                 training_seconds=sum(timing[i]['seconds'] for i in range(1, step + 1)),
                                 forward_matmul_flops=cell['estimated_forward_matmul_flops_per_sequence_mean']))
            if step in (8000, 10000):
                late.append(dict(step=step, nll33=cells[3, 3]['nll_mean'],
                                 weighted_nll=sum(c['nll_mean'] * c['training_update_probability'] for c in cells.values()),
                                 depth_gain_at_t3=cells[3, 0]['nll_mean'] - cells[3, 3]['nll_mean']))
        scores[variant] = dict(late=late, S=statistics.mean(c['nll33'] for c in late),
                              W=statistics.mean(c['weighted_nll'] for c in late))
    a, b = scores['separated'], scores['coincident']
    choose_b = (b['S'] <= a['S'] - .005 and b['W'] <= a['W'] + .005 and
                all(y['nll33'] <= x['nll33'] for x, y in zip(a['late'], b['late'])))
    decision = dict(selected='coincident' if choose_b else 'separated', scores=scores,
                    rule='Choose B only with >=0.005 mean late (3,3) NLL gain, no worse at either late checkpoint, and weighted NLL within 0.005; otherwise prefer A.',
                    limitation='One training seed; practical selection tolerance, not a confidence interval.',
                    panel_sha256=receipt['panel_sha256'],
                    report_hashes={str(report_path(v, s)): file_hash(report_path(v, s)) for v in VARIANTS for s in STEPS})
    analysis_path('decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    with analysis_path('curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(decision, indent=2))


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT, help='Use a fresh experiment-local directory for a rerun')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight')
    for command in ('train', 'evaluate'):
        p = sub.add_parser(command)
        p.add_argument('variant', choices=VARIANTS)
        if command == 'train':
            p.add_argument('--until', type=int, choices=(100, *STEPS), required=True)
        else:
            p.add_argument('--step', type=int, choices=STEPS, required=True)
            p.add_argument('--split', choices=('selection', 'confirmation'), default='selection')
    sub.add_parser('benchmark')
    sub.add_parser('summarize')
    args = parser.parse_args()
    ROOT = args.results_dir
    if args.command == 'preflight':
        print(json.dumps(freeze(), indent=2))
    elif args.command == 'train':
        train_until(args.variant, args.until)
    elif args.command == 'evaluate':
        evaluate(args.variant, args.step, args.split)
    elif args.command == 'benchmark':
        benchmark()
    else:
        summarize()


if __name__ == '__main__':
    main()
