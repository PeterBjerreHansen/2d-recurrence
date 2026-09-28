"""Run the post-training evaluation battery for the four recurrence arms.

Stages, each skipped when its outputs already exist (rerun to resume):

``training_graph``  per-position losses for every arm's cells on the
                    training-graph rows (``evaluation.position_losses``)
``live``            live NLL and legal-move probability, sharded across worker
                    processes (``evaluation.live_legality``); the live rows are
                    the first rows of the same seeded permutation, so they are a
                    subset of the training-graph rows
``compare``         paired, stratified reports (``evaluation.compare_losses``);
                    refused until every configured cell and live shard exists,
                    unless ``--allow-partial`` writes regenerated ``*.partial``
                    reports instead

Presets name the checkpoints. ``5B`` is the dry run on the completed 5B arms;
its transformer is a stand-in (the 20B transformer at 4B characters) because
the 5B transformer arm was not completed. ``20B`` uses the final 20B
checkpoints, or the 18B pre-decay ones with ``--step 175954``.

Example:
    python -m experiments.evaluation_battery.run --preset 20B --tg-device cuda \
        --live-device cuda --live-workers 8
"""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

from data_loader import file_hash


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path('experiments/evaluation_battery/results')
PANEL = 'experiments/long_runs/20B_recurrence/results/panel.json'
FIVE_B = 'experiments/long_runs/5B_axis/runpod_live_backups/{arm}/results/ckpt-step048876.pt'
TWENTY_B = 'experiments/long_runs/20B_recurrence/{arm}_20B/results/ckpt-step{step:06d}.pt'

TRAINING_GRAPH_CELLS = {
    'transformer': [(0, 0)],
    'temporal': [(0, 0), (1, 0), (3, 0), (7, 0), (15, 0)],
    'depth': [(0, 0), (0, 1), (0, 3), (0, 7), (0, 15)],
    # The single-axis and asymmetric cells ask whether one hybrid run serves as each single-axis model.
    'hybrid': [(0, 0), (1, 1), (3, 3), (7, 7), (15, 15), (3, 0), (0, 3), (3, 1), (1, 3), (1, 0), (0, 1)],
    # Post-hoc live-aligned checkpoints (experiments/ablations/live_warm_start), with --aligned-dir.
    'temporal_aligned': [(0, 0), (1, 0), (3, 0), (7, 0), (15, 0)],
    'hybrid_aligned': [(0, 0), (1, 1), (3, 3), (7, 7), (15, 15), (3, 0), (0, 3)],
}
ALIGNED = ('temporal_aligned', 'hybrid_aligned')
# (arm, J, KV strategy, legality): the deployed settings plus NLL-only points of the hybrid depth dial.
LIVE_SETTINGS = [
    ('transformer', 1, 'ordinary', True),
    ('temporal', 1, 'final_depth', True),
    ('hybrid', 1, 'depth_specialized', True),
    ('hybrid', 4, 'depth_specialized', True),
    ('depth', 4, 'depth_specialized', True),
    ('hybrid', 2, 'depth_specialized', False),
    ('hybrid', 8, 'depth_specialized', False),
    ('temporal_aligned', 1, 'final_depth', True),
    ('hybrid_aligned', 1, 'depth_specialized', True),
    ('hybrid_aligned', 4, 'depth_specialized', True),
]
TRAINING_GRAPH_PAIRS = [
    ('hybrid_3_3', 'temporal_3_0'), ('hybrid_3_3', 'depth_0_3'), ('temporal_3_0', 'depth_0_3'),
    ('hybrid_3_0', 'temporal_3_0'), ('hybrid_0_3', 'depth_0_3'), ('hybrid_3_3', 'transformer_0_0'),
    ('temporal_3_0', 'transformer_0_0'), ('depth_0_3', 'transformer_0_0'),
    ('temporal_aligned_3_0', 'temporal_3_0'), ('hybrid_aligned_3_3', 'hybrid_3_3'),
]
LIVE_PAIRS = [
    ('hybrid_live_J1', 'temporal_live_J1'), ('hybrid_live_J4', 'temporal_live_J1'),
    ('hybrid_live_J4', 'depth_live_J4'), ('hybrid_live_J4', 'hybrid_live_J1'),
    ('hybrid_live_J1', 'transformer_live_J1'), ('temporal_live_J1', 'transformer_live_J1'),
    ('temporal_live_J1', 'temporal_3_0'), ('hybrid_live_J4', 'hybrid_3_3'), ('depth_live_J4', 'depth_0_3'),
    ('hybrid_live_J4', 'transformer_live_J1'), ('depth_live_J4', 'transformer_live_J1'),
    ('temporal_aligned_live_J1', 'temporal_live_J1'), ('temporal_aligned_live_J1', 'transformer_live_J1'),
    ('temporal_aligned_live_J1', 'temporal_aligned_3_0'), ('hybrid_aligned_live_J1', 'temporal_aligned_live_J1'),
    ('hybrid_aligned_live_J1', 'hybrid_live_J1'), ('hybrid_aligned_live_J4', 'hybrid_live_J4'),
    ('hybrid_aligned_live_J4', 'depth_live_J4'), ('hybrid_aligned_live_J1', 'transformer_live_J1'),
]


def checkpoints(preset, step, aligned_dir=None):
    if aligned_dir:
        paths = checkpoints(preset, step)
        return {**paths, **{arm: str(Path(aligned_dir) / f'{arm}.pt') for arm in ALIGNED}}
    if preset == '5B':
        paths = {arm: FIVE_B.format(arm=arm) for arm in ('temporal', 'depth', 'hybrid')}
        paths['transformer'] = TWENTY_B.format(arm='transformer', step=39_101)
        return paths
    return {arm: TWENTY_B.format(arm=arm, step=step) for arm in ('transformer', 'temporal', 'depth', 'hybrid')}


def run(command, log=None):
    print('$ ' + ' '.join(command), flush=True)
    if log is None:
        subprocess.run(command, check=True, cwd=ROOT)
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(command, cwd=ROOT, stdout=log.open('a'), stderr=subprocess.STDOUT)


def row_arguments(limit, seed):
    arguments = ['--panel-file', PANEL, '--panel-split', 'confirmation', '--subset-seed', str(seed)]
    return arguments + (['--limit', str(limit)] if limit else [])


def configured_cells(args, arm):
    return [cell for cell in TRAINING_GRAPH_CELLS[arm] if max(cell) <= args.max_passes - 1]


def configured_live(args, paths):
    return [setting for setting in LIVE_SETTINGS if setting[1] <= args.max_live_depth and setting[0] in paths]


def shard_paths(args, root, arm, depth_steps):
    return [root / arm / f'live_J{depth_steps}.shard{index}of{args.live_workers}.npz'
            for index in range(args.live_workers)]


def training_graph(args, paths, root):
    for arm, path in paths.items():
        cells = configured_cells(args, arm)
        missing = [cell for cell in cells if not (root / arm / f'cell_{cell[0]}_{cell[1]}.npz').exists()]
        if not missing:
            continue
        # One invocation per cell keeps each completed cell even if a later one fails.
        for cell in missing:
            run([sys.executable, '-m', 'evaluation.position_losses', '--checkpoint', path,
                 '--output-dir', str(root / arm), '--cell', f'{cell[0]},{cell[1]}',
                 '--device', args.tg_device, '--batch-size', str(args.tg_batch_size),
                 *row_arguments(args.tg_limit, args.subset_seed)])


def live(args, paths, root):
    for arm, depth_steps, strategy, legality in configured_live(args, paths):
        name = f'live_J{depth_steps}'
        final = shard_paths(args, root, arm, depth_steps)
        workers = []
        for index, output in enumerate(final):
            if output.exists():
                continue
            command = [sys.executable, '-m', 'evaluation.live_legality', '--checkpoint', paths[arm],
                       '--output', str(output), '--device', args.live_device,
                       '--num-threads', str(args.live_threads), '--shard', f'{index}/{args.live_workers}',
                       *row_arguments(args.live_limit, args.subset_seed)]
            if arm != 'transformer':
                command += ['--depth-steps', str(depth_steps), '--kv-strategy', strategy]
            if not legality:
                command.append('--no-legality')
            workers.append(run(command, log=output.with_suffix('.log')))
        started = time.monotonic()
        for worker in workers:
            if worker.wait():
                raise RuntimeError(f'{arm} {name}: a live worker failed; see the shard logs in {root / arm}')
        if workers:
            print(f'{arm} {name}: {len(workers)} shards in {time.monotonic() - started:.0f}s', flush=True)


def compare(args, paths, root):
    tg_runs = {f'{arm}_{u_t}_{u_d}': root / arm / f'cell_{u_t}_{u_d}.npz'
               for arm in paths for u_t, u_d in configured_cells(args, arm)}
    live_runs = {f'{arm}_live_J{depth_steps}': str(root / arm / f'live_J{depth_steps}.shard*of{args.live_workers}.npz')
                 for arm, depth_steps, _, _ in configured_live(args, paths)}
    missing = [str(path) for path in tg_runs.values() if not path.exists()]
    missing += [str(path) for arm, depth_steps, _, _ in configured_live(args, paths)
                for path in shard_paths(args, root, arm, depth_steps) if not path.exists()]
    if missing and not args.allow_partial:
        raise SystemExit(f'Refusing final comparisons; {len(missing)} outputs are missing, e.g. {missing[0]}. '
                         'Finish the stages or pass --allow-partial.')
    if missing:
        # Partial reports cover only complete runs and are regenerated on every call.
        complete = lambda arm, depth: all(path.exists() for path in shard_paths(args, root, arm, depth))
        tg_runs = {label: path for label, path in tg_runs.items() if path.exists()}
        live_runs = {f'{arm}_live_J{depth}': live_runs[f'{arm}_live_J{depth}']
                     for arm, depth, _, _ in configured_live(args, paths) if complete(arm, depth)}
    suffix = '.partial' if missing else ''
    reports = [('training_graph', tg_runs, TRAINING_GRAPH_PAIRS)]
    primary = {label: tg_runs[label] for label in ('transformer_0_0', 'temporal_3_0', 'depth_0_3', 'hybrid_3_3',
                                                   'temporal_aligned_3_0', 'hybrid_aligned_3_3')
               if label in tg_runs}
    reports.append(('live', {**live_runs, **primary}, LIVE_PAIRS))
    for name, runs, pairs in reports:
        output = root / f'compare_{name}{suffix}.json'
        markdown = output.with_suffix('.md')
        if missing:
            output.unlink(missing_ok=True)
            markdown.unlink(missing_ok=True)
        elif output.exists():
            continue
        if not runs:
            continue
        pairs = [pair for pair in pairs if set(pair) <= set(runs)]
        command = [sys.executable, '-m', 'evaluation.compare_losses', '--output', str(output),
                   '--markdown', str(markdown), '--bootstrap', str(args.bootstrap)]
        command += list(itertools.chain.from_iterable(('--run', f'{label}={path}') for label, path in runs.items()))
        command += list(itertools.chain.from_iterable(('--pair', f'{a},{b}') for a, b in pairs))
        run(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=['5B', '20B'], required=True)
    parser.add_argument('--step', type=int, default=195_504, help='20B checkpoint step')
    parser.add_argument('--stages', nargs='+', default=['training_graph', 'live', 'compare'],
                        choices=['training_graph', 'live', 'compare'])
    parser.add_argument('--tg-limit', type=int, help='Training-graph rows; default every confirmation row')
    parser.add_argument('--live-limit', type=int, default=1000)
    parser.add_argument('--subset-seed', type=int, default=0)
    parser.add_argument('--max-passes', type=int, default=16, help='Skip training-graph cells above this')
    parser.add_argument('--max-live-depth', type=int, default=8)
    parser.add_argument('--tg-device', default='cuda')
    parser.add_argument('--tg-batch-size', type=int, default=8)
    parser.add_argument('--live-device', default='cpu')
    parser.add_argument('--live-workers', type=int, default=6)
    parser.add_argument('--live-threads', type=int, default=1)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--allow-partial', action='store_true',
                        help='Write regenerated compare_*.partial reports from whatever is complete')
    parser.add_argument('--aligned-dir', help='Also evaluate temporal_aligned.pt and hybrid_aligned.pt from here')
    parser.add_argument('--output-root', help='Defaults to experiments/evaluation_battery/results/<preset>-step<step>')
    args = parser.parse_args()
    paths = checkpoints(args.preset, args.step, args.aligned_dir)
    missing = [path for path in paths.values() if not (ROOT / path).is_file()]
    if missing:
        parser.error(f'Missing checkpoints: {missing}')
    label = '5B-dry-run' if args.preset == '5B' else f'20B-step{args.step}'
    root = Path(args.output_root) if args.output_root else RESULTS / label
    root.mkdir(parents=True, exist_ok=True)
    settings = root / 'battery.json'
    # Everything that decides which outputs exist or what the reports contain is frozen here.
    recorded = dict(preset=args.preset, step=args.step, checkpoints=paths,
                    checkpoint_sha256={arm: file_hash(ROOT / path) for arm, path in paths.items()},
                    tg_limit=args.tg_limit, live_limit=args.live_limit, subset_seed=args.subset_seed,
                    live_workers=args.live_workers, max_passes=args.max_passes,
                    max_live_depth=args.max_live_depth, bootstrap=args.bootstrap,
                    panel=PANEL, panel_split='confirmation')
    if settings.exists() and json.loads(settings.read_text()) != recorded:
        parser.error(f'{settings} records different battery settings; choose a new --output-root')
    settings.write_text(json.dumps(recorded, indent=2) + '\n')
    for stage in args.stages:
        globals()[stage](args, paths, root)


if __name__ == '__main__':
    main()
