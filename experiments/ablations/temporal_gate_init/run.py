"""Freeze, train, evaluate, and summarize the temporal-gate preflight."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import torch

from data_loader import ChessData, file_hash
from evaluation.panels import load_panel
from evaluation.recurrence_grid import evaluate_checkpoint
from train import train
from .study import (ACTUAL_CHARACTERS, CHARACTERS, CHECKPOINT_STEPS, GATE_INITIALIZATIONS,
                    PANEL_PATH, RESULTS_ROOT, STUDY_NAME, UPDATES, WARMUP_UPDATES, run_config)


ROOT = Path(RESULTS_ROOT)
DATASET = Path('data/chess_8M_v1')
DATASET_NAME = 'chess_8M_v1'
BLOCK_SIZE = 1023
PANEL = Path(PANEL_PATH)
PROTOCOL = ROOT / 'protocol.json'
ENVIRONMENT = ROOT / 'environment.json'
MASK_SEEDS = [11, 23, 37]
ARM_ORDER = tuple(GATE_INITIALIZATIONS)
SOURCE_EXCLUDED_PARTS = {'results', '__pycache__'}
SOURCE_FILES = (
    Path('configurator.py'), Path('data_loader.py'), Path('model.py'), Path('train.py'),
    Path('training_utils.py'), Path('pyproject.toml'), Path('uv.lock'),
    Path('docs/RECURRENCE_CONTRACT.md'), Path('experiments/serious.py'),
    Path('experiments/ablations/temporal_gate_init/README.md'),
)
SOURCE_DIRECTORIES = (
    Path('models'), Path('recurrence'), Path('evaluation'),
    Path('experiments/ablations/temporal_gate_init'),
)
SOURCE_ROOTS = SOURCE_FILES + SOURCE_DIRECTORIES
PACKAGE_NAMES = ('torch', 'numpy', 'chess', 'datasets', 'huggingface-hub')


def _git(*args, binary=False):
    output = subprocess.check_output(['git', *args])
    return output if binary else output.decode().strip()


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(_json_ready(value), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _source_paths():
    paths = set(SOURCE_FILES)
    for root in SOURCE_DIRECTORIES:
        paths.update(path for path in root.rglob('*')
                     if path.is_file() and path.suffix == '.py' and
                     not any(part in SOURCE_EXCLUDED_PARTS for part in path.parts))
    return sorted(paths)


def _scoped_git(*args, binary=False):
    return _git(*args, '--', *(str(path) for path in SOURCE_ROOTS), binary=binary)


def source_snapshot():
    patch = _scoped_git('diff', '--binary', 'HEAD', binary=True)
    return dict(branch=_git('branch', '--show-current'), head_commit=_git('rev-parse', 'HEAD'),
                working_tree_status=_scoped_git('status', '--porcelain=v1', '--untracked-files=all'),
                working_tree_patch_sha256=hashlib.sha256(patch).hexdigest(),
                files={str(path): file_hash(path) for path in _source_paths() if path.is_file()})


def runtime_environment():
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                cuda_device_count=torch.cuda.device_count(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                bf16_supported=(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
                packages=packages, uv_lock_sha256=file_hash('uv.lock'))


def _manifest():
    manifest_path, meta_path = DATASET / 'manifest.json', DATASET / 'meta.pkl'
    if not manifest_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f'{DATASET} must contain manifest.json and meta.pkl before freezing')
    manifest = json.loads(manifest_path.read_text())
    if file_hash(meta_path) != manifest['meta_sha256']:
        raise ValueError(f'{meta_path} does not match the pinned dataset manifest')
    return file_hash(manifest_path), manifest


def _materialized_data():
    manifest_hash, _ = _manifest()
    if not all((DATASET / name).is_file() for name in ('train.bin', 'val.bin')):
        return None
    data = ChessData(DATASET, BLOCK_SIZE)
    if data.manifest_hash != manifest_hash:
        raise ValueError('Materialized dataset differs from its manifest')
    return data


def _expected_panel(validation_count, manifest_hash):
    selection = (sorted(random.Random(2027).sample(range(validation_count), 128))
                 if validation_count >= 129 else list(range(max(1, validation_count // 2))))
    selected = set(selection)
    return dict(dataset_manifest_hash=manifest_hash, validation_row_count=validation_count,
                selection_seed=2027, selection_indices=selection,
                confirmation_indices=[i for i in range(validation_count) if i not in selected])


def freeze_panel(data):
    manifest_hash, manifest = _manifest()
    expected = _expected_panel(manifest['splits']['val']['rows'], manifest_hash)
    if PANEL.exists() and json.loads(PANEL.read_text()) != expected:
        raise ValueError(f'{PANEL} exists with different panel contents')
    if not PANEL.exists():
        PANEL.parent.mkdir(parents=True, exist_ok=True)
        PANEL.write_text(json.dumps(expected, indent=2) + '\n')
    if data is None:
        return dict(path=str(PANEL), sha256=file_hash(PANEL), split='selection',
                    selection_indices=expected['selection_indices'],
                    confirmation_indices=expected['confirmation_indices'],
                    row_indices=expected['selection_indices'],
                    row_count=len(expected['selection_indices']))
    return load_panel(PANEL, data, split='selection')


def _dataset_identity(data=None):
    manifest_hash, manifest = _manifest()
    if data is not None and data.manifest_hash != manifest_hash:
        raise ValueError('Materialized dataset differs from its manifest')
    return dict(name=DATASET_NAME, path=str(DATASET), manifest_sha256=manifest_hash,
                manifest_hash=manifest_hash, row_size=manifest['row_size'], context_length=BLOCK_SIZE,
                training_rows=manifest['splits']['train']['rows'],
                validation_rows=manifest['splits']['val']['rows'])


def _protocol_payload(data, panel):
    configs = {name: run_config(name) for name in ARM_ORDER}
    return dict(
        schema_version=1, study=STUDY_NAME,
        purpose='250M initialization preflight for temporal feedback gates',
        source=source_snapshot(), dataset=_dataset_identity(data),
        panel=dict(path=str(PANEL), sha256=panel['sha256'], split='selection',
                   selection_rows=len(panel['selection_indices']),
                   confirmation_rows=len(panel['confirmation_indices']), selection_seed=2027),
        configurations=configs,
        configuration_sha256={name: _canonical_hash(config) for name, config in configs.items()},
        training=dict(target_characters=CHARACTERS, actual_characters=ACTUAL_CHARACTERS,
                      optimizer_updates=UPDATES, effective_batch_size=100,
                      characters_per_update=100 * BLOCK_SIZE, warmup_updates=WARMUP_UPDATES,
                      lr_decay_updates=UPDATES, checkpoint_steps=CHECKPOINT_STEPS,
                      changed_parameter='temporal_memory_gate_init'),
        evaluation=dict(mask_seeds=MASK_SEEDS, dtype='float32', execution='training_graph',
                        diagnostics='gate contributions, state RMS/change, and clipping are inspected'),
        requested_hardware=dict(provider='runpod', gpu='RTX 3090', count=1,
                                pricing='on-demand', provisioning='external to this runner'),
        freeze_environment=runtime_environment())


def _compare_frozen(existing, current):
    if existing.get('source') != current.get('source'):
        return False
    keys = ('schema_version', 'study', 'dataset', 'panel', 'configurations',
            'configuration_sha256', 'training', 'evaluation', 'requested_hardware')
    return all(existing.get(key) == current.get(key) for key in keys)


def freeze(refresh=False):
    data = _materialized_data()
    panel = freeze_panel(data)
    receipt = _protocol_payload(data, panel)
    ROOT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        existing = json.loads(PROTOCOL.read_text())
        if not _compare_frozen(existing, receipt):
            if not refresh:
                raise ValueError(f'{PROTOCOL} differs from the current study; use a new study')
            if any((_arm_output(name) / 'ckpt.pt').exists() for name in ARM_ORDER):
                raise ValueError('Cannot refresh after an arm has produced a checkpoint')
        else:
            return existing
    PROTOCOL.write_text(json.dumps(_json_ready(receipt), indent=2, allow_nan=False) + '\n')
    return receipt


def recorded_protocol():
    if not PROTOCOL.is_file():
        raise FileNotFoundError(f'{PROTOCOL} is missing; run freeze first')
    return json.loads(PROTOCOL.read_text())


def verify_protocol():
    recorded = recorded_protocol()
    data = _materialized_data()
    panel = freeze_panel(data)
    if not _compare_frozen(recorded, _protocol_payload(data, panel)):
        raise ValueError('Frozen protocol/source/data/panel/configuration changed')
    return recorded, data, panel


def preflight():
    data = _materialized_data()
    if data is None:
        raise RuntimeError(f'{DATASET} is incomplete; materialize train.bin and val.bin first')
    recorded, _, _ = verify_protocol()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('The experiment requires exactly one CUDA GPU')
    if 'RTX 3090' not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported():
        raise RuntimeError('The experiment requires one BF16-capable RTX 3090')
    receipt = dict(protocol_sha256=file_hash(PROTOCOL), protocol_branch=recorded['source']['branch'],
                   protocol_commit=recorded['source']['head_commit'], runtime=runtime_environment())
    if ENVIRONMENT.exists() and json.loads(ENVIRONMENT.read_text()) != receipt:
        raise ValueError(f'{ENVIRONMENT} does not match the frozen runtime')
    if not ENVIRONMENT.exists():
        ENVIRONMENT.write_text(json.dumps(_json_ready(receipt), indent=2, allow_nan=False) + '\n')
    return receipt


def _arm_output(name):
    return Path(run_config(name)['out_dir'])


def _write_arm_plan(name):
    config = run_config(name)
    expected = dict(study=STUDY_NAME, arm=name, protocol_sha256=file_hash(PROTOCOL),
                    configuration=config, configuration_sha256=_canonical_hash(config),
                    expected_optimizer_updates=UPDATES, expected_characters=ACTUAL_CHARACTERS)
    path = _arm_output(name) / 'plan.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != _json_ready(expected):
        raise ValueError(f'{path} differs from the frozen study')
    if not path.exists():
        path.write_text(json.dumps(_json_ready(expected), indent=2, allow_nan=False) + '\n')
    return expected


def train_arm(name):
    if name not in ARM_ORDER:
        raise ValueError(name)
    preflight()
    plan = _write_arm_plan(name)
    config = run_config(name)
    checkpoint = _arm_output(name) / 'ckpt.pt'
    step = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        step = saved['iter_num']
        if step > UPDATES:
            raise ValueError(f'{checkpoint} is beyond the frozen horizon')
        if step == UPDATES:
            return checkpoint
        config['init_from'] = 'resume'
    started = time.monotonic()
    result = train(config)
    final = torch.load(result, map_location='cpu', weights_only=False)
    if final['iter_num'] != UPDATES:
        raise RuntimeError(f'{name} stopped at {final["iter_num"]}, expected {UPDATES}')
    with (ROOT / 'run_history.jsonl').open('a') as stream:
        stream.write(json.dumps(_json_ready(dict(
            event='training', arm=name, gate_init=GATE_INITIALIZATIONS[name],
            started_utc=datetime.now(timezone.utc).isoformat(), resumed_from_step=step,
            final_step=final['iter_num'], elapsed_seconds=time.monotonic() - started, plan=plan)),
            allow_nan=False) + '\n')
    return result


def _checkpoint_for(name, step):
    return _arm_output(name) / f'ckpt-step{step:06d}.pt'


def _evaluation_path(name, step, split):
    return _arm_output(name) / f'grid-{split}-step{step:06d}.json'


def _validate_report(report, checkpoint, step, split, panel):
    if (report.get('checkpoint_sha256') != file_hash(checkpoint) or
            report.get('checkpoint_step') != step or report.get('panel_split') != split or
            report.get('panel_file_sha256') != panel['sha256'] or
            report.get('mask_seeds') != MASK_SEEDS or
            report.get('study') != STUDY_NAME):
        raise ValueError('Existing evaluation does not match this experiment')


def evaluate_arm(name, step, split='selection', device='cuda'):
    if name not in ARM_ORDER:
        raise ValueError(name)
    _, data, _ = verify_protocol()
    if data is None:
        raise RuntimeError(f'{DATASET} is incomplete')
    if step not in CHECKPOINT_STEPS:
        raise ValueError(f'step must be one of {CHECKPOINT_STEPS}')
    checkpoint = _checkpoint_for(name, step)
    output = _evaluation_path(name, step, split)
    panel = load_panel(PANEL, data, split=split)
    if output.exists():
        report = json.loads(output.read_text())
        _validate_report(report, checkpoint, step, split, panel)
        print(f'Reusing verified {output}')
        return report
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch.set_num_threads(4)
    report = evaluate_checkpoint(
        checkpoint, device=device, panel_file=str(PANEL), panel_split=split,
        batch_size=5, mask_seeds=MASK_SEEDS, diagnostics=True)
    report.update(study=STUDY_NAME, arm=name, gate_init=GATE_INITIALIZATIONS[name])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(report), indent=2, allow_nan=False) + '\n')
    rows = [{key: value for key, value in cell.items()
             if key not in {'placements', 'diagnostics'}} for cell in report['cells']]
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _validate_report(report, checkpoint, step, split, panel)
    return report


def summarize():
    recorded_protocol()
    rows = []
    for name in ARM_ORDER:
        for step in CHECKPOINT_STEPS:
            report = json.loads(_evaluation_path(name, step, 'selection').read_text())
            cells = {(cell['u_t'], cell['u_d']): cell for cell in report['cells']}
            cell = cells[(3, 3)]
            diagnostics = cell.get('diagnostics', {})
            gate = diagnostics.get('gate_value_contributions', {}).get('temporal', [])
            rows.append(dict(
                arm=name, step=step, gate_init=GATE_INITIALIZATIONS[name],
                nll_00=cells[(0, 0)]['nll_mean'], nll_11=cells[(1, 1)]['nll_mean'],
                nll_33=cell['nll_mean'], core_rms=diagnostics.get('core_output_rms'),
                core_relative_change=diagnostics.get('successive_pass_relative_change', {}).get('core'),
                alpha_mean=(sum(item['alpha_mean'] for item in gate) / len(gate) if gate else None),
                beta_mean=(sum(item['beta_mean'] for item in gate) / len(gate) if gate else None),
                memory_value_rms=(sum(item['memory_value_rms'] for item in gate) / len(gate)
                                  if gate else None),
                anchor_value_rms=(sum(item['prelude_value_rms'] for item in gate) / len(gate)
                                 if gate else None),
            ))
    path = ROOT / 'analysis'
    path.mkdir(parents=True, exist_ok=True)
    (path / 'summary.json').write_text(json.dumps(_json_ready(rows), indent=2, allow_nan=False) + '\n')
    flat_rows = [{key: value for key, value in row.items()
                  if key not in {'core_rms', 'core_relative_change'}} for row in rows]
    with (path / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--refresh', action='store_true')
    sub.add_parser('preflight')
    p = sub.add_parser('train')
    p.add_argument('arm', choices=ARM_ORDER)
    p = sub.add_parser('evaluate')
    p.add_argument('arm', choices=ARM_ORDER)
    p.add_argument('--step', type=int, choices=CHECKPOINT_STEPS, required=True)
    p.add_argument('--split', choices=('selection', 'confirmation'), default='selection')
    p.add_argument('--device', default='cuda')
    sub.add_parser('summarize')
    args = parser.parse_args()
    if args.command == 'freeze':
        print(json.dumps(_json_ready(freeze(args.refresh)), indent=2))
    elif args.command == 'preflight':
        print(json.dumps(_json_ready(preflight()), indent=2))
    elif args.command == 'train':
        train_arm(args.arm)
    elif args.command == 'evaluate':
        evaluate_arm(args.arm, args.step, args.split, args.device)
    else:
        print(json.dumps(_json_ready(summarize()), indent=2))


if __name__ == '__main__':
    main()
