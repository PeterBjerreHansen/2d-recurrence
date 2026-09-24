"""Freeze, validate, train, and evaluate the four-arm 20B recurrence study.

``dry-run`` checks the resolved protocol without writing files or starting
training. ``freeze`` records the settled protocol; training and evaluation
then verify that immutable receipt before continuing.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import torch

from data_loader import ChessData, file_hash
from evaluation.all_rows import evaluate_baseline, load_model
from evaluation.live_inference import evaluate_teacher_forced
from evaluation.panels import fixed_panel_batches, load_panel
from evaluation.recurrence_grid import evaluate_grid
from evaluation.stress_checks import check as run_stress_checks
from recurrence.schedule import update_probability_map_at_step, update_probabilities_at_step
from train import get_lr, train
from . import study


ROOT = Path(study.RESULTS_ROOT)
DATASET = Path('data/chess_8M_v1')
DATASET_NAME = 'chess_8M_v1'
BLOCK_SIZE = 1023
PANEL = Path(study.PANEL_PATH)
PROTOCOL = ROOT / 'protocol.json'
TRANSFER_MANIFEST = Path('TRANSFER_MANIFEST.json')
ARM_ORDER = study.ARM_ORDER
MASK_SEEDS = study.MASK_SEEDS
SOURCE_SUFFIXES = {'.py', '.toml', '.lock', '.json', '.md', '.txt', '.yaml', '.yml'}
SOURCE_EXCLUDED_PARTS = {'data', 'results', '__pycache__'}
# Pod scheduling and monitoring runs locally and may be fixed mid-run; it is
# not part of the experiment's scientific source.
SOURCE_EXCLUDED_DIRS = (Path('experiments/long_runs/20B_recurrence/ops'),)
SOURCE_REQUIRED_FILES = {
    Path('data/chess_v1/prepare.py'),
    Path('data/chess_v1/meta.pkl'),
    Path('docs/upstream.json'),
}
PACKAGE_NAMES = ('torch', 'numpy', 'chess', 'datasets', 'huggingface-hub')
ENVIRONMENT_LOG = 'environment.jsonl'
# An arm may move to a replacement host. These runtime fields must stay fixed;
# the OS/kernel string, Python patch level, and host identity may change.
PINNED_RUNTIME_KEYS = ('torch', 'cuda', 'gpu', 'bf16_supported', 'packages', 'uv_lock_sha256')
# Set to True only after the deterministic >=1,000-update interrupted WSD and
# curriculum comparison matches exactly, and before freezing.
EXACT_RESUME_GATE_PASSED = True


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
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_paths():
    names = set()
    for command in (['ls-files', '-z'], ['ls-files', '--others', '--exclude-standard', '-z']):
        names.update(name.decode() for name in _git(*command, binary=True).split(b'\0') if name)
    paths = []
    for name in names:
        path = Path(name)
        if path.suffix not in SOURCE_SUFFIXES and path not in SOURCE_REQUIRED_FILES:
            continue
        if (path not in SOURCE_REQUIRED_FILES and
                any(part in SOURCE_EXCLUDED_PARTS for part in path.parts)):
            continue
        if any(path.is_relative_to(directory) for directory in SOURCE_EXCLUDED_DIRS):
            continue
        paths.append(path)
    return sorted(paths)


def source_hashes():
    """Hash source/configuration files, including untracked implementation files."""
    return {str(path): file_hash(path) if path.is_file() else None for path in _source_paths()}


def _verify_transferred_source(expected):
    for name, digest in expected.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'Transferred source differs from the frozen protocol: {name}')


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
        if not TRANSFER_MANIFEST.is_file() or not PROTOCOL.is_file():
            raise RuntimeError('Source provenance requires Git or a verified transfer bundle') from error
        transfer = json.loads(TRANSFER_MANIFEST.read_text())
        recorded = json.loads(PROTOCOL.read_text())
        expected = recorded['source']['files']
        transferred = transfer.get('files', {})
        if any(transferred.get(path) != digest for path, digest in expected.items()):
            raise ValueError('Transfer manifest does not match frozen source hashes')
        _verify_transferred_source(expected)
        return dict(
            branch=transfer['branch'], head_commit=transfer['base_commit'],
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
    cuda = torch.cuda.is_available()
    return dict(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cuda_available=cuda,
        cuda_device_count=torch.cuda.device_count(),
        gpu=torch.cuda.get_device_name(0) if cuda else None,
        bf16_supported=torch.cuda.is_bf16_supported() if cuda else False,
        packages=packages,
        uv_lock_sha256=file_hash('uv.lock'),
    )


def _manifest():
    manifest_path, meta_path = DATASET / 'manifest.json', DATASET / 'meta.pkl'
    if not manifest_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f'{DATASET} must contain manifest.json and meta.pkl before freezing')
    manifest_hash = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if file_hash(meta_path) != manifest['meta_sha256']:
        raise ValueError(f'{meta_path} does not match the pinned dataset manifest')
    return manifest_hash, manifest


def _materialized_data():
    manifest_hash, _ = _manifest()
    if not all((DATASET / name).is_file() for name in ('train.bin', 'val.bin')):
        return None
    data = ChessData(DATASET, BLOCK_SIZE)
    if data.manifest_hash != manifest_hash:
        raise ValueError('Materialized dataset differs from its manifest')
    return data


def _expected_panel(validation_count, manifest_hash):
    if validation_count < 2:
        raise ValueError('At least two validation rows are required for a frozen panel')
    selection = (sorted(random.Random(2027).sample(range(validation_count), 128))
                 if validation_count >= 129 else list(range(max(1, validation_count // 2))))
    selected = set(selection)
    return dict(
        dataset_manifest_hash=manifest_hash,
        validation_row_count=validation_count,
        selection_seed=2027,
        selection_indices=selection,
        confirmation_indices=[index for index in range(validation_count) if index not in selected],
    )


def freeze_panel(data):
    manifest_hash, manifest = _manifest()
    expected = _expected_panel(manifest['splits']['val']['rows'], manifest_hash)
    if data is not None and data.manifest_hash != manifest_hash:
        raise ValueError('Materialized dataset differs from its manifest')
    if PANEL.exists():
        if json.loads(PANEL.read_text()) != expected:
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
                    row_indices=expected['selection_indices'], row_count=len(expected['selection_indices']))
    return load_panel(PANEL, data, split='selection')


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


def configurations():
    return {name: _json_ready(study.run_config(name))
            for name in ARM_ORDER}


def _protocol_payload(data, panel):
    configs = configurations()
    return dict(
        schema_version=1,
        study=study.STUDY_NAME,
        purpose='Twenty-billion-character, data-matched recurrence-axis discovery study',
        source=source_snapshot(),
        dataset=_dataset_identity(data),
        panel=dict(
            path=str(PANEL), sha256=panel['sha256'], split='selection',
            selection_rows=len(panel['selection_indices']),
            confirmation_rows=len(panel['confirmation_indices']), selection_seed=2027),
        gate_initialization=dict(
            selected_value=study.TEMPORAL_GATE_INIT,
            selection_source='20B plan: retain .10 after inconclusive sensitivity preflight'),
        configurations=configs,
        configuration_sha256={name: _canonical_hash(config) for name, config in configs.items()},
        training=dict(
            target_characters=study.CHARACTERS,
            actual_characters=study.ACTUAL_CHARACTERS,
            optimizer_updates=study.UPDATES,
            effective_batch_size=100,
            characters_per_update=100 * BLOCK_SIZE,
            curriculum_starts=list(study.CURRICULUM_STARTS),
            curriculum_max_update_probabilities=[list(p) for p in study.UPDATE_COUNT_PROBABILITIES],
            warmup_updates=study.WARMUP_UPDATES,
            lr_schedule='wsd',
            lr_decay_start=study.WSD_DECAY_START,
            lr_decay_endpoint=study.UPDATES,
            lr_decay_endpoint_semantics=(
                'lr_decay_iters counts updates; the final update at index UPDATES-1 uses min LR.'),
            learning_rate=study.LEARNING_RATE,
            min_learning_rate=study.MIN_LEARNING_RATE,
            objective='final_only', dtype='bfloat16', device='cuda',
            checkpoint_steps=study.CHECKPOINT_STEPS,
            arm_order=list(ARM_ORDER)),
        evaluation=dict(
            major_checkpoint_steps=list(study.MAJOR_CHECKPOINT_STEPS),
            primary_cells={name: [list(cell) for cell in cells]
                           for name, cells in study.PRIMARY_EVALUATION_CELLS.items()},
            extended_cells={name: [list(cell) for cell in cells]
                            for name, cells in study.EXTENDED_EVALUATION_CELLS.items()},
            curve_checkpoints='primary recurrence cells only',
            mask_seeds=MASK_SEEDS, dtype='float32', execution='training_graph',
            placement_variation='distinct declared write-mask placements',
            live=dict(
                checkpoints='major checkpoints only',
                settings={name: [dict(depth_steps=depth, kv_strategy=kv) for depth, kv in settings]
                          for name, settings in study.LIVE_EVALUATION_SETTINGS.items()},
                panel_split='selection', prefill='sequential_live', dtype='float32',
                semantics='teacher-forced next-character NLL; state reset per stored row')),
        requested_hardware=dict(cuda_devices=1, bf16=True,
                                provider='user-provisioned; this runner does not provision hardware'),
        freeze_environment=runtime_environment(),
    )


def _compare_frozen(existing, current):
    # Git metadata describes provenance; identical source bytes define the experiment.
    if existing['source']['files'] != current['source']['files']:
        return False
    keys = ('schema_version', 'study', 'dataset', 'panel', 'gate_initialization',
            'configurations', 'configuration_sha256', 'training', 'evaluation', 'requested_hardware')
    return all(existing.get(key) == current.get(key) for key in keys)


def freeze(refresh=False):
    """Write the receipt for the fixed study protocol."""
    data = _materialized_data()
    panel = freeze_panel(data)
    receipt = _protocol_payload(data, panel)
    ROOT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        existing = json.loads(PROTOCOL.read_text())
        if not _compare_frozen(existing, receipt):
            if not refresh:
                raise ValueError(f'{PROTOCOL} differs from this protocol; use a new study or --refresh')
            if any((_arm_output(name) / 'ckpt.pt').exists() for name in ARM_ORDER):
                raise ValueError('Cannot refresh the protocol after an arm has produced a checkpoint')
        else:
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


def dry_run():
    """Validate the 20B arithmetic, curriculum, WSD, and evaluation plan in memory."""
    configs = configurations()
    manifest_hash, manifest = _manifest()
    phase_checks = {}
    arms = []
    for name in ARM_ORDER:
        config = study.run_config(name)
        if config['max_iters'] * config['batch_size'] * config['gradient_accumulation_steps'] * BLOCK_SIZE != study.ACTUAL_CHARACTERS:
            raise ValueError(f'{name}: update count does not produce the frozen character horizon')
        if config['architecture'] == 'recurrent':
            active = []
            for index, start in enumerate(study.CURRICULUM_STARTS):
                before = study.CURRICULUM_STARTS[index + 1] - 1 if index + 1 < len(study.CURRICULUM_STARTS) else study.UPDATES - 1
                matrix_at_start = update_probabilities_at_step(config, start)
                matrix_before_next = update_probabilities_at_step(config, before)
                active.append(dict(
                    start_step=start,
                    max_update_probabilities=[
                        sum(matrix_at_start[i][j]
                            for i, temporal in enumerate(study.UPDATE_SUPPORT)
                            for j, depth in enumerate(study.UPDATE_SUPPORT)
                            if max(temporal, depth) == count)
                        for count in study.UPDATE_SUPPORT],
                    phase_end_matches=matrix_before_next == matrix_at_start,
                ))
            phase_checks[name] = active
        arms.append(dict(
            name=name, architecture=config['architecture'], recurrence_mode=config['recurrence_mode'],
            configuration_sha256=_canonical_hash(config),
            curve_evaluation_cells=study.evaluation_cells(name, 19_551),
            major_evaluation_cells=study.evaluation_cells(name, study.WSD_DECAY_START),
        ))
    recurrent_config = study.run_config('hybrid')
    trace_steps = (0, 1_999, 2_000, study.WSD_DECAY_START - 1,
                   study.WSD_DECAY_START, study.UPDATES - 1, study.UPDATES)
    lr_trace = {str(step): get_lr(step, recurrent_config) for step in trace_steps}
    if lr_trace['175954'] != study.LEARNING_RATE or lr_trace['195504'] != study.MIN_LEARNING_RATE:
        raise ValueError('WSD boundary checks did not resolve to the frozen peak and minimum rates')
    return dict(
        status='passed',
        protocol_frozen=PROTOCOL.is_file(),
        dataset_manifest_sha256=manifest_hash,
        dataset_materialized=all((DATASET / name).is_file() for name in ('train.bin', 'val.bin')),
        validation_rows=manifest['splits']['val']['rows'],
        training=dict(
            target_characters=study.CHARACTERS,
            actual_characters=study.ACTUAL_CHARACTERS,
            optimizer_updates=study.UPDATES,
            curriculum_phase_checks=phase_checks,
            lr_trace=lr_trace),
        arms=arms,
        temporal_memory_gate_init=study.TEMPORAL_GATE_INIT,
        files_written=False,
        training_started=False,
        configurations_sha256={name: _canonical_hash(value) for name, value in configs.items()},
    )


def _python_minor(version):
    return '.'.join(version.split()[0].split('.')[:2])


def record_environment(name, receipt):
    """Append this host's receipt to the arm's environment log.

    Each arm keeps its own log so arms on separate Pods never collide. A
    repeated identical receipt is not re-recorded. A new host is appended and
    marked as a host change; protocol drift or a change in a pinned runtime
    field is rejected.
    """
    path = _arm_output(name) / ENVIRONMENT_LOG
    entries = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
               if path.is_file() else [])
    if entries:
        first = entries[0]
        if receipt['protocol_sha256'] != first['protocol_sha256']:
            raise ValueError(f'{name}: frozen protocol differs from the one this arm started with')
        changed = [key for key in PINNED_RUNTIME_KEYS
                   if receipt['runtime'].get(key) != first['runtime'].get(key)]
        if _python_minor(receipt['runtime']['python']) != _python_minor(first['runtime']['python']):
            changed.append('python')
        if changed:
            raise ValueError(f'{name}: runtime differs from the arm\'s first host in {changed}')
        last = {key: value for key, value in entries[-1].items()
                if key not in ('recorded_utc', 'host_change')}
        if last == receipt:
            return entries[-1]
    entry = dict(receipt, recorded_utc=datetime.now(timezone.utc).isoformat(),
                 host_change=bool(entries))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(entry, allow_nan=False) + '\n')
    return entry


def preflight(name=None):
    """Verify a frozen study, complete dataset, and CUDA BF16 runtime.

    With an arm name, also record this host in that arm's environment log.
    """
    data = _materialized_data()
    if data is None:
        raise RuntimeError(f'{DATASET} is incomplete: train.bin and val.bin are required for CUDA preflight')
    recorded, _, _ = verify_protocol()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('The study requires exactly one CUDA GPU')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The study requires CUDA BF16 support')
    receipt = dict(protocol_sha256=file_hash(PROTOCOL),
                   protocol_branch=recorded['source']['branch'],
                   protocol_commit=recorded['source']['head_commit'],
                   host=os.environ.get('RUNPOD_POD_ID') or platform.node(),
                   runtime=runtime_environment())
    return record_environment(name, receipt) if name is not None else receipt


def _arm_output(name):
    return Path(study.run_config(name)['out_dir'])


def _arm_plan(name):
    config = study.run_config(name)
    return dict(study=study.STUDY_NAME, arm=name, protocol_sha256=file_hash(PROTOCOL),
                configuration_sha256=_canonical_hash(_json_ready(config)),
                configuration=_json_ready(config), expected_optimizer_updates=study.UPDATES,
                expected_characters=study.ACTUAL_CHARACTERS)


def _write_arm_plan(name):
    output = _arm_output(name)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'plan.json'
    expected = _arm_plan(name)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f'{path} differs from the frozen protocol; use a new arm directory')
    else:
        path.write_text(json.dumps(expected, indent=2, allow_nan=False) + '\n')
    return expected


def _append_history(name, record):
    # Per arm, so separate Pods never append to the same file.
    path = _arm_output(name) / 'run_history.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def train_arm(name, resume=False):
    """Train one arm from scratch, or with ``resume`` continue its checkpoint.

    A fresh start requires that no checkpoint exists. ``resume`` requires a
    valid checkpoint and never falls back to a fresh start.
    """
    if name not in ARM_ORDER:
        raise ValueError(name)
    preflight(name)
    plan = _write_arm_plan(name)
    config = study.run_config(name)
    checkpoint = Path(config['out_dir']) / 'ckpt.pt'
    resumed_from = None
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        _validate_checkpoint(saved, name)
        step = saved['iter_num']
        if step == study.UPDATES:
            print(f'{name}: already complete at step {step}')
            return checkpoint
        if not resume:
            raise RuntimeError(f'{name}: a partial checkpoint exists at step {step}; '
                               'continue it with --resume instead of starting fresh')
        if not EXACT_RESUME_GATE_PASSED:
            raise RuntimeError(
                'Resume is disabled until the production WSD/curriculum '
                'configuration passes the >=1,000-update exact-resume gate')
        config['init_from'] = 'resume'
        resumed_from = dict(step=step, checkpoint_sha256=file_hash(checkpoint))
    elif resume:
        raise RuntimeError(f'{name}: --resume found no checkpoint at {checkpoint}; refusing to start fresh')
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    result = train(config)
    elapsed = time.monotonic() - started
    final = torch.load(result, map_location='cpu', weights_only=False)
    if final['iter_num'] != study.UPDATES:
        raise RuntimeError(f'{name} stopped at {final["iter_num"]}, expected {study.UPDATES}')
    _append_history(name, dict(
        event='training', arm=name, started_utc=started_utc, resumed_from=resumed_from,
        final_step=final['iter_num'], elapsed_seconds=elapsed,
        checkpoint_sha256=file_hash(result), plan=plan))
    return result


def arm_integrity(name, results_dir=None, require_complete=False):
    """Check an arm's results against the frozen protocol; never raises on bad data.

    Used on the Pod each monitoring tick and locally after collection. Checks
    that the arm plan and every environment receipt carry the frozen protocol
    hash, and that ``ckpt.pt`` matches the frozen configuration, dataset and
    panel. With ``require_complete``, also requires the final step and an
    evaluation report for every retained checkpoint whose recorded hash matches
    that checkpoint.
    """
    results = Path(results_dir) if results_dir is not None else _arm_output(name)
    protocol_sha256 = file_hash(PROTOCOL)
    problems = []
    plan_path = results / 'plan.json'
    if not plan_path.is_file():
        problems.append('plan.json is missing')
    elif json.loads(plan_path.read_text()).get('protocol_sha256') != protocol_sha256:
        problems.append('plan.json records a different protocol')
    environment_path = results / ENVIRONMENT_LOG
    environment = ([json.loads(line) for line in environment_path.read_text().splitlines() if line.strip()]
                   if environment_path.is_file() else [])
    if not environment:
        problems.append(f'{ENVIRONMENT_LOG} is missing or empty')
    elif any(entry.get('protocol_sha256') != protocol_sha256 for entry in environment):
        problems.append(f'{ENVIRONMENT_LOG} records a different protocol')
    checkpoint = dict(exists=(results / 'ckpt.pt').is_file(), sha256=None, step=None, valid=False)
    if checkpoint['exists']:
        path = results / 'ckpt.pt'
        checkpoint['sha256'] = file_hash(path)
        try:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            checkpoint['step'] = saved['iter_num']
            _validate_checkpoint(saved, name)
            checkpoint['valid'] = True
        except Exception as error:  # report, never crash the monitor
            problems.append(f'ckpt.pt is invalid: {str(error)[:200]}')
    elif require_complete:
        problems.append('ckpt.pt is missing')
    if require_complete:
        if checkpoint['step'] != study.UPDATES:
            problems.append(f"final step is {checkpoint['step']}, expected {study.UPDATES}")
        for step in study.EVALUATION_CHECKPOINT_STEPS:
            retained = results / f'ckpt-step{step:06d}.pt'
            report = results / f'evaluation-selection-{retained.stem}.json'
            if not retained.is_file() or not report.is_file():
                problems.append(f'step {step}: checkpoint or evaluation report missing')
            elif json.loads(report.read_text()).get('checkpoint_sha256') != file_hash(retained):
                problems.append(f'step {step}: evaluation report does not match its checkpoint')
    return dict(arm=name, results=str(results), protocol_sha256=protocol_sha256,
                checkpoint=checkpoint, environment_hosts=len(environment),
                ok=not problems, problems=problems)


def _checkpoint_for(name, step):
    output = _arm_output(name)
    return output / ('ckpt.pt' if step is None else f'ckpt-step{step:06d}.pt')


def _evaluation_path(name, checkpoint, split):
    return _arm_output(name) / f'evaluation-{split}-{checkpoint.stem}.json'


def _validate_checkpoint(checkpoint, name, step=None):
    """Reject a misplaced arm or changed training protocol before using its weights."""
    protocol = recorded_protocol()
    expected = protocol['configurations'][name]
    actual = dict(checkpoint['config'])
    actual['init_from'] = 'scratch'  # A resumed run has the same scientific configuration.
    if _json_ready(actual) != expected:
        raise ValueError(f'Checkpoint configuration differs from the frozen {name} arm')
    if (checkpoint['manifest_hash'] != protocol['dataset']['manifest_hash'] or
            checkpoint.get('eval_panel_sha256') != protocol['panel']['sha256']):
        raise ValueError('Checkpoint dataset or panel differs from the frozen protocol')
    completed = checkpoint['iter_num']
    if (type(completed) is not int or not 0 <= completed <= study.UPDATES or
            (step is not None and completed != step)):
        raise ValueError('Checkpoint step differs from the requested study checkpoint')


def _panel_rows(data, panel):
    """Return the panel's validation rows as (inputs, targets) pairs."""
    rows = data.rows['val']
    pairs = []
    for index in panel['row_indices']:
        row = torch.from_numpy(rows[index, :BLOCK_SIZE + 1].astype('int64'))
        pairs.append((row[:-1], row[1:]))
    return pairs


def evaluate_live(model, name, step, rows):
    """Run the declared live-feedback settings; returns [] off major checkpoints."""
    reports = []
    for depth_steps, kv_strategy in study.live_evaluation_settings(name, step):
        started = time.monotonic()
        report = evaluate_teacher_forced(model, rows, depth_steps=depth_steps,
                                         kv_strategy=kv_strategy)
        report['seconds'] = time.monotonic() - started
        reports.append(report)
    return reports


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
    checkpoint_data, checkpoint_hash, checkpoint_data_source, model, recurrent = load_model(checkpoint, device)
    _validate_checkpoint(checkpoint_data, name, step)
    if checkpoint_data_source.manifest_hash != data.manifest_hash:
        raise ValueError('Checkpoint dataset differs from the frozen study dataset')
    batches, metadata = fixed_panel_batches(data, panel, study.run_config(name)['batch_size'])
    requested_cells = study.evaluation_cells(name, checkpoint_data['iter_num'])
    is_major_checkpoint = checkpoint_data['iter_num'] in study.MAJOR_CHECKPOINT_STEPS
    if recurrent:
        active_matrix = update_probabilities_at_step(checkpoint_data['config'], checkpoint_data['iter_num'])
        result = evaluate_grid(
            model, data, fixed_batches=batches, data_seed=None,
            batch_size=metadata['batch_size'], mask_seeds=MASK_SEEDS,
            training_update_probabilities=update_probability_map_at_step(
                checkpoint_data['config'], checkpoint_data['iter_num']),
            diagnostics=False, sampling=metadata['sampling'], cells=requested_cells,
            optional_cells=(set(requested_cells) -
                            set(study.PRIMARY_EVALUATION_CELLS[name])
                            if is_major_checkpoint else ()))
        result['next_update_probability_step'] = checkpoint_data['iter_num']
        result['next_update_probability_matrix'] = [list(row) for row in active_matrix]
        result['last_update_probability_matrix'] = (
            [list(row) for row in update_probabilities_at_step(
                checkpoint_data['config'], checkpoint_data['iter_num'] - 1)]
            if checkpoint_data['iter_num'] > 0 else None)
    else:
        result = evaluate_baseline(model, batches, device)
    numerical_stress = None
    diagnostic_failures = [dict(source='extended_evaluation_grid', **failure)
                           for failure in result.get('failed_cells', [])]
    if recurrent and is_major_checkpoint:
        stress_panel, checks = run_stress_checks(model, data, panel, device)
        numerical_stress = dict(panel=stress_panel, checks=checks, dtype='float32', device=device)
        diagnostic_failures.extend(
            dict(source='activation_stress', u_t=check['u_t'], u_d=check['u_d'],
                 error=check.get('diagnostic_failure') or 'stress check did not pass')
            for check in checks if not check.get('finite', False))
    # Sequential decoding is per row; the 82k-row confirmation split is out of reach.
    live = (evaluate_live(model, name, checkpoint_data['iter_num'], _panel_rows(data, panel))
            if recurrent and split == 'selection' else [])
    report = dict(
        study=study.STUDY_NAME, arm=name, split=split,
        execution='training_graph' if recurrent else 'baseline',
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        checkpoint_step=checkpoint_data['iter_num'], training_seed=checkpoint_data['config']['seed'],
        manifest_hash=data.manifest_hash, panel_sha256=panel['sha256'],
        panel_row_count=panel['row_count'], fixed_batches=metadata,
        requested_cells=[list(cell) for cell in requested_cells],
        extended_diagnostic=is_major_checkpoint,
        diagnostic_failures=diagnostic_failures,
        numerical_stress_checks=numerical_stress,
        mask_seeds=MASK_SEEDS, device=device, dtype='float32',
        protocol_sha256=file_hash(PROTOCOL), metrics=result, live_metrics=live)
    if file_hash(checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed during evaluation')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if recurrent:
        rows = [dict(execution='training_graph', u_t=cell['u_t'], u_d=cell['u_d'],
                     depth_steps=None, kv_strategy=None, nll=cell['nll_mean'],
                     accuracy=cell['accuracy_mean'], placement_count=cell['placement_count'])
                for cell in result['cells']]
        rows.extend(dict(execution='live', u_t=None, u_d=None, depth_steps=item['depth_steps'],
                         kv_strategy=item['kv_strategy'], nll=item['nll'],
                         accuracy=item['accuracy'], placement_count=1)
                    for item in live)
    else:
        rows = [dict(execution='baseline', u_t=0, u_d=0, depth_steps=None, kv_strategy=None,
                     nll=result['nll'], accuracy=result['accuracy'], placement_count=1)]
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return report


def status():
    recorded_protocol()
    for name in ARM_ORDER:
        checkpoint = Path(study.run_config(name)['out_dir']) / 'ckpt.pt'
        if not checkpoint.is_file():
            print(f'{name}: not started')
            continue
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        characters = saved['iter_num'] * 100 * BLOCK_SIZE
        print(f'{name}: step {saved["iter_num"]}/{study.UPDATES}, '
              f'characters {characters}/{study.ACTUAL_CHARACTERS}, checkpoint {file_hash(checkpoint)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--refresh', action='store_true',
                               help='Refresh only before any arm has produced a checkpoint')
    sub.add_parser('dry-run')
    preflight_parser = sub.add_parser('preflight')
    preflight_parser.add_argument('arm', nargs='?', choices=ARM_ORDER,
                                  help="Record this host in the arm's environment log")
    train_parser = sub.add_parser('train')
    train_parser.add_argument('arm', choices=ARM_ORDER)
    train_parser.add_argument('--resume', action='store_true',
                              help='Continue the existing checkpoint; never start fresh')
    integrity_parser = sub.add_parser('integrity')
    integrity_parser.add_argument('arm', choices=ARM_ORDER)
    integrity_parser.add_argument('--results-dir', help='Check a copied results directory instead')
    integrity_parser.add_argument('--complete', action='store_true',
                                  help='Also require the final step and every evaluation report')
    evaluate_parser = sub.add_parser('evaluate')
    evaluate_parser.add_argument('arm', choices=ARM_ORDER)
    evaluate_parser.add_argument('--step', type=int, choices=study.CHECKPOINT_STEPS)
    evaluate_parser.add_argument('--split', choices=('selection', 'confirmation'), default='selection')
    evaluate_parser.add_argument('--device', default='cuda')
    study_parser = sub.add_parser('study')
    study_parser.add_argument('--evaluate-checkpoints',
                              dest='evaluate_checkpoints', action='store_true',
                              help='Evaluate every retained nonzero checkpoint after each arm finishes')
    sub.add_parser('status')
    args = parser.parse_args()
    if args.command == 'freeze':
        print(json.dumps(freeze(args.refresh), indent=2))
    elif args.command == 'dry-run':
        print(json.dumps(dry_run(), indent=2))
    elif args.command == 'preflight':
        print(json.dumps(preflight(args.arm), indent=2))
    elif args.command == 'train':
        train_arm(args.arm, resume=args.resume)
    elif args.command == 'integrity':
        print(json.dumps(arm_integrity(args.arm, args.results_dir, args.complete)))
    elif args.command == 'evaluate':
        evaluate_arm(args.arm, args.step, args.split, args.device)
    elif args.command == 'study':
        for name in ARM_ORDER:
            train_arm(name)
            if args.evaluate_checkpoints:
                for checkpoint_step in study.EVALUATION_CHECKPOINT_STEPS:
                    evaluate_arm(name, checkpoint_step)
    else:
        status()


if __name__ == '__main__':
    main()
