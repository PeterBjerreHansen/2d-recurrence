"""Freeze and run the four-arm 5B recurrence-axis study.

This runner does not provision Verda. Run ``freeze`` locally after inspecting
the working tree, transfer the frozen source/data snapshot to the VM, then run
``preflight`` and the arm commands from the repository root on the VM. Every
training and evaluation command verifies the frozen protocol and resumes only
the matching arm checkpoint.
"""
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
from evaluation.all_rows import evaluate_baseline, load_model, training_probabilities
from evaluation.panels import fixed_panel_batches, load_panel
from evaluation.recurrence_grid import evaluate_grid
from .study import (
    ACTUAL_CHARACTERS,
    CHARACTERS,
    CHECKPOINT_STEPS,
    PANEL_PATH,
    RESULTS_ROOT,
    RUNS,
    STUDY_NAME,
    UPDATES,
    WARMUP_UPDATES,
    run_config,
)
from train import train


ROOT = Path(RESULTS_ROOT)
DATASET = Path('data/chess_8M_v1')
DATASET_NAME = 'chess_8M_v1'
BLOCK_SIZE = 1023
PANEL = Path(PANEL_PATH)
PROTOCOL = ROOT / 'protocol.json'
ENVIRONMENT = ROOT / 'environment.json'
TRANSFER_MANIFEST = Path('TRANSFER_MANIFEST.json')
ARM_ORDER = tuple(RUNS)
MASK_SEEDS = [11, 23, 37]
SOURCE_SUFFIXES = {'.py', '.toml', '.lock', '.json', '.md', '.txt', '.yaml', '.yml'}
SOURCE_EXCLUDED_PARTS = {'data', 'results', '__pycache__'}
PACKAGE_NAMES = ('torch', 'numpy', 'chess', 'datasets', 'huggingface-hub')


def _git(*args, binary=False):
    output = subprocess.check_output(['git', *args])
    return output if binary else output.decode().strip()


def _canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_paths():
    names = set()
    for command in (['ls-files', '-z'], ['ls-files', '--others', '--exclude-standard', '-z']):
        names.update(name.decode() for name in _git(*command, binary=True).split(b'\0') if name)
    paths = []
    for name in names:
        path = Path(name)
        if (path.suffix in SOURCE_SUFFIXES and
                not any(part in SOURCE_EXCLUDED_PARTS for part in path.parts)):
            paths.append(path)
    return sorted(paths)


def source_hashes():
    """Hash source/configuration files, including untracked implementation files."""
    hashes = {}
    for path in _source_paths():
        hashes[str(path)] = file_hash(path) if path.is_file() else None
    return hashes


def source_snapshot():
    try:
        patch = _git('diff', '--binary', 'HEAD', binary=True)
        return dict(
            branch=_git('branch', '--show-current'),
            head_commit=_git('rev-parse', 'HEAD'),
            working_tree_status=_git('status', '--porcelain=v1', '--untracked-files=all'),
            working_tree_patch_sha256=hashlib.sha256(patch).hexdigest(),
            files=source_hashes(),
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        # A transferred working-tree bundle intentionally excludes .git. The
        # transfer manifest supplies the original branch/commit and all source
        # hashes; verify those files directly instead of inventing Git state.
        if not TRANSFER_MANIFEST.is_file() or not PROTOCOL.is_file():
            raise RuntimeError('Source provenance requires Git or a verified transfer bundle') from error
        transfer = json.loads(TRANSFER_MANIFEST.read_text())
        recorded = json.loads(PROTOCOL.read_text())
        expected = recorded['source']['files']
        transferred = transfer.get('files', {})
        if any(transferred.get(path) != digest for path, digest in expected.items()):
            raise ValueError('Transfer manifest does not match the frozen source hashes')
        return dict(
            branch=transfer['branch'],
            head_commit=transfer['base_commit'],
            working_tree_status='verified transfer bundle without Git metadata',
            working_tree_patch_sha256=transfer.get(
                'working_tree_patch_sha256', recorded['source']['working_tree_patch_sha256']),
            files={path: transferred[path] for path in expected},
        )


def runtime_environment():
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return dict(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
        gpu=gpu,
        bf16_supported=(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
        packages=packages,
        uv_lock_sha256=file_hash('uv.lock'),
    )


def _manifest():
    manifest_path = DATASET / 'manifest.json'
    meta_path = DATASET / 'meta.pkl'
    if not manifest_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f'{DATASET} must contain manifest.json and meta.pkl before freezing the study')
    manifest_hash = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if file_hash(meta_path) != manifest['meta_sha256']:
        raise ValueError(f'{meta_path} does not match the pinned dataset manifest')
    return manifest_hash, manifest


def _materialized_data():
    _manifest()
    required = [DATASET / name for name in ('train.bin', 'val.bin')]
    if not all(path.is_file() for path in required):
        return None
    return ChessData(DATASET, BLOCK_SIZE)


def _expected_panel(validation_count, manifest_hash):
    count = validation_count
    if count == 1:
        raise ValueError('The validation split has only one row; no confirmation panel is possible')
    selection = (sorted(random.Random(2027).sample(range(count), 128))
                 if count >= 129 else list(range(max(1, count // 2))))
    selected = set(selection)
    return dict(
        dataset_manifest_hash=manifest_hash,
        validation_row_count=count,
        selection_seed=2027,
        selection_indices=selection,
        confirmation_indices=[index for index in range(count) if index not in selected],
    )


def freeze_panel(data):
    manifest_hash, manifest = _manifest()
    expected = _expected_panel(manifest['splits']['val']['rows'], manifest_hash)
    if data is not None and data.manifest_hash != manifest_hash:
        raise ValueError('Materialized dataset differs from its manifest')
    if PANEL.exists():
        actual = json.loads(PANEL.read_text())
        if actual != expected:
            raise ValueError(f'{PANEL} exists with different panel contents')
    else:
        PANEL.parent.mkdir(parents=True, exist_ok=True)
        PANEL.write_text(json.dumps(expected, indent=2) + '\n')
    if data is None:
        return dict(path=str(PANEL), sha256=file_hash(PANEL), split='selection',
                    dataset_manifest_hash=manifest_hash,
                    validation_row_count=expected['validation_row_count'],
                    selection_indices=expected['selection_indices'],
                    confirmation_indices=expected['confirmation_indices'],
                    row_indices=expected['selection_indices'],
                    row_count=len(expected['selection_indices']))
    return load_panel(PANEL, data, split='selection')


def configurations():
    return {name: run_config(name) for name in ARM_ORDER}


def _dataset_identity(data=None):
    manifest_hash, manifest = _manifest()
    if data is not None and data.manifest_hash != manifest_hash:
        raise ValueError('Materialized dataset differs from its manifest')
    return dict(
        name=DATASET_NAME,
        path=str(DATASET),
        manifest_sha256=manifest_hash,
        manifest_hash=manifest_hash,
        row_size=manifest['row_size'],
        context_length=BLOCK_SIZE,
        training_rows=manifest['splits']['train']['rows'],
        validation_rows=manifest['splits']['val']['rows'],
    )


def _config_hashes(configs):
    return {name: _canonical_hash(config) for name, config in configs.items()}


def _protocol_payload(data, panel):
    configs = configurations()
    return dict(
        schema_version=1,
        study=STUDY_NAME,
        purpose='Five-billion-character data-matched recurrence-axis ablation',
        source=source_snapshot(),
        dataset=_dataset_identity(data),
        panel=dict(path=str(PANEL), sha256=panel['sha256'], split='selection',
                   selection_rows=len(panel['selection_indices']),
                   confirmation_rows=len(panel['confirmation_indices']),
                   selection_seed=2027),
        configurations=configs,
        configuration_sha256=_config_hashes(configs),
        training=dict(
            target_characters=CHARACTERS,
            actual_characters=ACTUAL_CHARACTERS,
            optimizer_updates=UPDATES,
            effective_batch_size=100,
            characters_per_update=100 * BLOCK_SIZE,
            warmup_updates=WARMUP_UPDATES,
            lr_decay_updates=UPDATES,
            learning_rate=3e-4,
            min_learning_rate=3e-5,
            objective='final_only',
            dtype='bfloat16',
            device='cuda',
            checkpoint_steps=CHECKPOINT_STEPS,
        ),
        evaluation=dict(
            mask_seeds=MASK_SEEDS,
            dtype='float32',
            execution='training_graph',
            placement_variation='distinct declared mask placements',
            training_seed_variation='not included; all arms use the pinned training seed',
        ),
        requested_hardware=dict(provider='verda', gpu='RTX A6000', count=1,
                                pricing='spot', provisioning='external to this runner'),
        freeze_environment=runtime_environment(),
    )


def _source_matches(existing, current):
    left, right = existing['source'], current['source']
    return all(left.get(key) == right.get(key)
               for key in ('branch', 'head_commit', 'working_tree_patch_sha256', 'files'))


def _compare_frozen(existing, current):
    # The local freeze environment may differ from the VM runtime environment;
    # source, data, panel, and resolved training configurations may not differ.
    if not _source_matches(existing, current):
        return False
    keys = ('schema_version', 'study', 'dataset', 'panel', 'configurations',
            'configuration_sha256', 'training', 'evaluation', 'requested_hardware')
    return all(existing.get(key) == current.get(key) for key in keys)


def freeze(refresh=False):
    """Freeze branch/source, dataset, panel, and all four resolved configs."""
    data = _materialized_data()
    panel = freeze_panel(data)
    receipt = _protocol_payload(data, panel)
    ROOT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        existing = json.loads(PROTOCOL.read_text())
        if not _compare_frozen(existing, receipt):
            if not refresh:
                raise ValueError(f'{PROTOCOL} does not match the current source/configuration; use a new study')
            if any((_arm_output(name) / 'ckpt.pt').exists() for name in ARM_ORDER):
                raise ValueError('Cannot refresh a protocol after any arm has produced a checkpoint')
            PROTOCOL.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            return receipt
        return existing
    PROTOCOL.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def recorded_protocol():
    if not PROTOCOL.is_file():
        raise FileNotFoundError(f'{PROTOCOL} is missing; run freeze first')
    return json.loads(PROTOCOL.read_text())


def verify_protocol():
    recorded = recorded_protocol()
    data = _materialized_data()
    panel = freeze_panel(data)
    current = _protocol_payload(data, panel)
    if not _compare_frozen(recorded, current):
        raise ValueError('Frozen protocol/source/data/panel/configuration changed; do not continue this study')
    return recorded, data, panel


def preflight():
    """Verify the frozen study and the required single-GPU CUDA environment."""
    data = _materialized_data()
    if data is None:
        raise RuntimeError(
            f'{DATASET} is incomplete: train.bin and val.bin are required before CUDA preflight')
    recorded, _, _ = verify_protocol()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('The study requires exactly one CUDA GPU')
    if 'RTX A6000' not in torch.cuda.get_device_name(0):
        raise RuntimeError('The frozen study requires one RTX A6000')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The frozen study requires CUDA BF16 support')
    receipt = dict(protocol_sha256=file_hash(PROTOCOL), protocol_branch=recorded['source']['branch'],
                   protocol_commit=recorded['source']['head_commit'], runtime=runtime_environment())
    if ENVIRONMENT.exists():
        existing = json.loads(ENVIRONMENT.read_text())
        if existing != receipt:
            raise ValueError(f'{ENVIRONMENT} does not match the frozen runtime')
    else:
        ENVIRONMENT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def _arm_output(name):
    return Path(run_config(name)['out_dir'])


def _arm_plan(name):
    config = run_config(name)
    return dict(study=STUDY_NAME, arm=name, protocol_sha256=file_hash(PROTOCOL),
                configuration_sha256=_canonical_hash(config), configuration=config,
                expected_optimizer_updates=UPDATES, expected_characters=ACTUAL_CHARACTERS)


def _write_arm_plan(name):
    output = _arm_output(name)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'plan.json'
    expected = _arm_plan(name)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f'{path} differs from the frozen study; preserve it and use a new arm')
    else:
        path.write_text(json.dumps(expected, indent=2, allow_nan=False) + '\n')
    return expected


def _append_history(record):
    with (ROOT / 'run_history.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def train_arm(name):
    if name not in ARM_ORDER:
        raise ValueError(name)
    preflight()
    plan = _write_arm_plan(name)
    config = run_config(name)
    checkpoint = Path(config['out_dir']) / 'ckpt.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        step = saved['iter_num']
        if step > UPDATES:
            raise ValueError(f'{checkpoint} is beyond the frozen study horizon')
        if step == UPDATES:
            print(f'{name}: already complete at step {step}')
            return checkpoint
        config['init_from'] = 'resume'
    else:
        step = 0
    started = time.monotonic()
    result = train(config)
    elapsed = time.monotonic() - started
    final = torch.load(result, map_location='cpu', weights_only=False)
    if final['iter_num'] < UPDATES:
        raise RuntimeError(f'{name} stopped at {final["iter_num"]}, expected {UPDATES}')
    _append_history(dict(event='training', arm=name, started_utc=datetime.now(timezone.utc).isoformat(),
                         resumed_from_step=step, final_step=final['iter_num'],
                         elapsed_seconds=elapsed, checkpoint_sha256=file_hash(result),
                         plan=plan))
    return result


def _checkpoint_for(name, step):
    output = _arm_output(name)
    return output / ('ckpt.pt' if step is None else f'ckpt-step{step:06d}.pt')


def _evaluation_path(name, checkpoint, split):
    return _arm_output(name) / f'evaluation-{split}-{checkpoint.stem}.json'


def _validate_existing_evaluation(report, checkpoint, split, panel, data):
    return (report.get('study') == STUDY_NAME and report.get('split') == split and
            report.get('checkpoint_sha256') == file_hash(checkpoint) and
            report.get('panel_sha256') == panel['sha256'] and
            report.get('manifest_hash') == data.manifest_hash and
            report.get('mask_seeds') == MASK_SEEDS)


def evaluate_arm(name, step=None, split='selection', device='cuda'):
    if name not in ARM_ORDER:
        raise ValueError(name)
    _, data, _ = verify_protocol()
    if data is None:
        raise RuntimeError(f'{DATASET} is incomplete; materialize and validate it before evaluation')
    checkpoint = _checkpoint_for(name, step)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = _evaluation_path(name, checkpoint, split)
    panel = load_panel(PANEL, data, split=split)
    if output.exists():
        report = json.loads(output.read_text())
        if not _validate_existing_evaluation(report, checkpoint, split, panel, data):
            raise ValueError(f'{output} does not match the frozen checkpoint/panel')
        print(f'Reusing verified {output}')
        return report

    checkpoint_data, checkpoint_hash, checkpoint_data_source, model, recurrent = load_model(checkpoint, device)
    if checkpoint_data_source.manifest_hash != data.manifest_hash:
        raise ValueError('Checkpoint dataset differs from the frozen study dataset')
    batches, metadata = fixed_panel_batches(data, panel, run_config(name)['batch_size'])
    if recurrent:
        result = evaluate_grid(
            model, data, fixed_batches=batches, data_seed=None,
            batch_size=metadata['batch_size'], mask_seeds=MASK_SEEDS,
            training_probabilities=training_probabilities(checkpoint_data['config']),
            diagnostics=False, sampling=metadata['sampling'])
    else:
        result = evaluate_baseline(model, batches, device)
    report = dict(
        study=STUDY_NAME,
        arm=name,
        split=split,
        execution='training_graph' if recurrent else 'baseline',
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash,
        checkpoint_step=checkpoint_data['iter_num'],
        training_seed=checkpoint_data['config']['seed'],
        manifest_hash=data.manifest_hash,
        panel_sha256=panel['sha256'],
        panel_row_count=panel['row_count'],
        fixed_batches=metadata,
        mask_seeds=MASK_SEEDS,
        device=device,
        dtype='float32',
        protocol_sha256=file_hash(PROTOCOL),
        metrics=result,
    )
    if file_hash(checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed during evaluation')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if recurrent:
        rows = [dict(u_t=cell['u_t'], u_d=cell['u_d'], nll=cell['nll_mean'],
                     accuracy=cell['accuracy_mean'], placement_count=cell['placement_count'])
                for cell in result['cells']]
    else:
        rows = [dict(u_t=0, u_d=0, nll=result['nll'], accuracy=result['accuracy'], placement_count=1)]
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return report


def status():
    recorded_protocol()
    for name in ARM_ORDER:
        checkpoint = _checkpoint_for(name, None)
        if not checkpoint.is_file():
            print(f'{name}: not started')
            continue
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        print(f'{name}: step {saved["iter_num"]}/{UPDATES}, '
              f'characters {saved["iter_num"] * 100 * BLOCK_SIZE}/{ACTUAL_CHARACTERS}, '
              f'checkpoint {file_hash(checkpoint)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--refresh', action='store_true',
                   help='Refresh a pre-training receipt after a source/runner fix; refuses existing checkpoints')
    sub.add_parser('preflight')
    p = sub.add_parser('train')
    p.add_argument('arm', choices=ARM_ORDER)
    p = sub.add_parser('evaluate')
    p.add_argument('arm', choices=ARM_ORDER)
    p.add_argument('--step', type=int, choices=CHECKPOINT_STEPS)
    p.add_argument('--split', choices=('selection', 'confirmation'), default='selection')
    p.add_argument('--device', default='cuda')
    p = sub.add_parser('study')
    p.add_argument('--evaluate', action='store_true', help='Evaluate each completed arm on selection after training')
    sub.add_parser('status')
    args = parser.parse_args()
    if args.command == 'freeze':
        print(json.dumps(freeze(args.refresh), indent=2))
    elif args.command == 'preflight':
        print(json.dumps(preflight(), indent=2))
    elif args.command == 'train':
        train_arm(args.arm)
    elif args.command == 'evaluate':
        evaluate_arm(args.arm, args.step, args.split, args.device)
    elif args.command == 'study':
        for name in ARM_ORDER:
            train_arm(name)
            if args.evaluate:
                evaluate_arm(name)
    else:
        status()


if __name__ == '__main__':
    main()
