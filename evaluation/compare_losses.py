"""Stratified, paired comparison of saved per-position losses and legality records.

Inputs are ``label=path`` runs written by ``evaluation.position_losses`` or
``evaluation.live_legality``; a path may be a glob, whose files (for example
live shards) are merged by validation row. Only rows present in every run are
compared, so all differences are paired on identical targets.

NLL is reported overall and by character class, ply bucket, row-position bucket
and game ordinal (see ``evaluation.pgn_annotations``). Each ``--pair A,B``
reports ``A - B`` with a percentile bootstrap interval that resamples rows. The
interval covers evaluation-row sampling only, not training-seed variation.

Example:
    python -m evaluation.compare_losses --run hybrid=.../hybrid/cell_3_3.npz \
        --run temporal=.../temporal/cell_3_0.npz --pair hybrid,temporal \
        --output .../comparison.json --markdown .../comparison.md
"""

import argparse
from glob import glob
import json
from pathlib import Path
import warnings

import numpy as np

from data_loader import ChessData
from evaluation.pgn_annotations import CLASSES, annotate_row


PLY_EDGES = (0, 10, 20, 40, 60, 80, 100)
POSITION_EDGES = (0, 128, 256, 512, 768)
GAME_EDGES = (0, 1, 2, 3)
LEGALITY_METRICS = ('legal_mass', 'top_legal_match', 'greedy_legal', 'greedy_match', 'actual_logprob')


def bucket_labels(edges):
    return [f'{low}-{high - 1}' for low, high in zip(edges, edges[1:])] + [f'{edges[-1]}+']


IDENTITY_FIELDS = ('checkpoint_sha256', 'manifest_hash', 'execution', 'recurrence_mode', 'u_t', 'u_d',
                   'depth_steps', 'kv_strategy', 'legality')
SHARD_FIELDS = ('shard', 'shard_row_count', 'row_count')


def check_run_identity(pattern, metas):
    """Refuse to merge files from different checkpoints, settings or row sets, or an incomplete shard set."""
    def identity(meta):
        rows = {key: value for key, value in (meta.get('rows') or {}).items() if key not in SHARD_FIELDS}
        return json.dumps(dict({field: meta.get(field) for field in IDENTITY_FIELDS}, rows=rows), sort_keys=True)
    if len({identity(meta) for meta in metas}) != 1:
        raise ValueError(f'{pattern}: files come from different checkpoints, settings or row sets')
    shards = (metas[0].get('rows') or {}).get('shards', 1)
    present = sorted((meta.get('rows') or {}).get('shard', 0) for meta in metas)
    if present != list(range(shards)):
        raise ValueError(f'{pattern}: expected shards 0..{shards - 1}, found {present}')


def load_run(pattern):
    paths = sorted(glob(pattern)) or ([pattern] if Path(pattern).exists() else [])
    if not paths:
        raise FileNotFoundError(f'No run files match {pattern!r}')
    parts = [np.load(path) for path in paths]
    check_run_identity(pattern, [json.loads(str(part['meta'])) for part in parts])
    rows = np.concatenate([part['row_indices'] for part in parts])
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f'{pattern}: files repeat validation rows')
    order = np.argsort(rows)
    run = dict(paths=paths, rows=rows[order],
               losses=np.concatenate([part['losses'] for part in parts])[order],
               correct=np.concatenate([part['correct'] for part in parts])[order],
               meta=[json.loads(str(part['meta'])) for part in parts])
    if all(meta.get('legality') for meta in run['meta']):
        run['moves'] = {name: np.concatenate([part[name] for part in parts])
                        for name in ('move_row', 'move_start', 'move_ply', *LEGALITY_METRICS)}
    return run


def strata(data, rows):
    """Return {family: (codes [rows, targets], labels)} for the given validation rows."""
    itos = data.meta['itos']
    klass, ply, game = [], [], []
    for index in rows:
        annotation = annotate_row(''.join(itos[int(value)] for value in data.rows['val'][index]),
                                  legal_moves=False)
        klass.append(annotation.klass[:data.context_length])
        ply.append(annotation.ply[:data.context_length])
        game.append(annotation.game[:data.context_length])
    klass, ply, game = np.stack(klass), np.stack(ply), np.stack(game)
    ply_bucket = np.where(ply >= 0, np.digitize(ply, PLY_EDGES) - 1, -1)
    position = np.broadcast_to(np.digitize(np.arange(data.context_length), POSITION_EDGES) - 1, klass.shape)
    game_bucket = np.where(game >= 0, np.digitize(game, GAME_EDGES) - 1, -1)
    ply_labels = bucket_labels(PLY_EDGES)
    cross = np.where((klass >= 0) & (ply_bucket >= 0), klass * len(ply_labels) + ply_bucket, -1)
    return dict(
        overall=(np.zeros(klass.shape, dtype=np.int64), ['all']),
        character_class=(klass, list(CLASSES)),
        ply=(ply_bucket, ply_labels),
        row_position=(position, bucket_labels(POSITION_EDGES)),
        game_ordinal=(game_bucket, bucket_labels(GAME_EDGES)),
        class_by_ply=(cross, [f'{name} | ply {bucket}' for name in CLASSES for bucket in ply_labels]),
    )


def row_sums(values, codes, count):
    """Per-row sums and counts for each category: two arrays [rows, count]."""
    rows = values.shape[0]
    valid = codes >= 0
    flat = (np.arange(rows)[:, None] * count + codes)[valid]
    sums = np.bincount(flat, weights=values[valid], minlength=rows * count).reshape(rows, count)
    counts = np.bincount(flat, minlength=rows * count).reshape(rows, count).astype(np.float64)
    return sums, counts


def bootstrap_weights(row_count, replicates, seed):
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(row_count, size=row_count), minlength=row_count).astype(np.float64)
            for _ in range(replicates)]


def paired_difference(left, right, counts, weights):
    """Mean difference per category with a 95% percentile row-bootstrap interval."""
    difference = left - right
    totals = counts.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        estimate = difference.sum(axis=0) / totals
        samples = np.stack([(w @ difference) / (w @ counts) for w in weights])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # empty strata have no interval
        low, high = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return [dict(difference=_float(e), ci_low=_float(a), ci_high=_float(b), count=int(n))
            for e, a, b, n in zip(estimate, low, high, totals)]


def _float(value):
    return None if not np.isfinite(value) else float(value)


def legality_tables(runs, labels, pairs, weights, row_order):
    """Align move records across runs and summarize legality overall and by ply."""
    keyed = {}
    for label in labels:
        moves = runs[label]['moves']
        keyed[label] = {(int(r), int(s)): index for index, (r, s) in
                        enumerate(zip(moves['move_row'], moves['move_start']))}
    row_position = {row: index for index, row in enumerate(row_order)}
    common = sorted(key for key in set.intersection(*(set(keys) for keys in keyed.values()))
                    if key[0] in row_position)
    if not common:
        return None
    move_rows = np.array([row_position[row] for row, _ in common])
    first = runs[labels[0]]['moves']
    plies = first['move_ply'][[keyed[labels[0]][key] for key in common]]
    ply_codes = np.digitize(plies, PLY_EDGES) - 1
    ply_labels = bucket_labels(PLY_EDGES)
    categories = [('overall', np.zeros(len(common), dtype=np.int64), ['all']), ('ply', ply_codes, ply_labels)]
    aggregated = {}
    for label in labels:
        moves = runs[label]['moves']
        selected = [keyed[label][key] for key in common]
        aggregated[label] = {}
        for metric in LEGALITY_METRICS:
            values = moves[metric][selected].astype(np.float64)
            aggregated[label][metric] = {}
            for family, codes, names in categories:
                sums = np.zeros((len(row_order), len(names)))
                counts = np.zeros((len(row_order), len(names)))
                np.add.at(sums, (move_rows, codes), values)
                np.add.at(counts, (move_rows, codes), 1)
                aggregated[label][metric][family] = (sums, counts, names)
    result = dict(move_count=len(common), metrics={}, pairs={})
    for label in labels:
        result['metrics'][label] = {
            metric: {family: dict(zip(names, (_float(v) for v in sums.sum(0) / np.maximum(counts.sum(0), 1))))
                     for family, (sums, counts, names) in aggregated[label][metric].items()}
            for metric in LEGALITY_METRICS}
    for left, right in pairs:
        result['pairs'][f'{left} - {right}'] = {
            metric: {family: dict(zip(names, paired_difference(
                aggregated[left][metric][family][0], aggregated[right][metric][family][0],
                aggregated[left][metric][family][1], weights)))
                for family, (_, _, names) in aggregated[left][metric].items()}
            for metric in LEGALITY_METRICS}
    return result


def markdown(report):
    lines = ['# Loss comparison', '',
             f"Rows compared: {report['row_count']}. Intervals are 95% row-bootstrap percentiles "
             f"({report['bootstrap_replicates']} replicates); they do not include training-seed variation.", '']
    labels = list(report['runs'])
    for family, table in report['nll'].items():
        lines += [f'## NLL by {family.replace("_", " ")}', '',
                  '| stratum | targets | ' + ' | '.join(labels) + ' |',
                  '|---|---:|' + '---:|' * len(labels)]
        for name, row in table.items():
            lines.append(f"| {name} | {row['count']} | " +
                         ' | '.join('–' if row[label] is None else f'{row[label]:.5f}' for label in labels) + ' |')
        lines.append('')
        for pair, differences in report['pairs'].items():
            lines += [f'**{pair}** by {family.replace("_", " ")}', '', '| stratum | Δ NLL | 95% CI |', '|---|---:|---|']
            for name, item in differences[family].items():
                # The full class-by-ply cross is in the JSON; show the two most telling classes here.
                if family == 'class_by_ply' and not name.startswith(('move_first', 'check_slot')):
                    continue
                if item['difference'] is not None:
                    lines.append(f"| {name} | {item['difference']:+.5f} | "
                                 f"[{item['ci_low']:+.5f}, {item['ci_high']:+.5f}] |")
            lines.append('')
    legality = report.get('legality')
    if legality:
        lines += ['## Legality at move starts', '', f"Moves compared: {legality['move_count']}.", '',
                  '| run | ' + ' | '.join(LEGALITY_METRICS) + ' |', '|---|' + '---:|' * len(LEGALITY_METRICS)]
        for label, metrics in legality['metrics'].items():
            lines.append(f'| {label} | ' + ' | '.join(f"{metrics[m]['overall']['all']:.4f}"
                                                       for m in LEGALITY_METRICS) + ' |')
        lines.append('')
        for pair, metrics in legality['pairs'].items():
            lines += [f'**{pair}**', '', '| metric | Δ | 95% CI |', '|---|---:|---|']
            for metric in LEGALITY_METRICS:
                item = metrics[metric]['overall']['all']
                lines.append(f"| {metric} | {item['difference']:+.5f} | "
                             f"[{item['ci_low']:+.5f}, {item['ci_high']:+.5f}] |")
            lines.append('')
            lines += [f'**{pair}**: legal-mass difference by ply', '',
                      '| ply | moves | Δ legal mass | 95% CI |', '|---|---:|---:|---|']
            for name, item in metrics['legal_mass']['ply'].items():
                if item['difference'] is not None:
                    lines.append(f"| {name} | {item['count']} | {item['difference']:+.5f} | "
                                 f"[{item['ci_low']:+.5f}, {item['ci_high']:+.5f}] |")
            lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True, help='label=path or label=glob')
    parser.add_argument('--pair', action='append', default=[], help='A,B reports A - B')
    parser.add_argument('--data-dir', default='data/chess_8M_v1')
    parser.add_argument('--context-length', type=int, default=1023)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--output', required=True, help='New JSON report path')
    parser.add_argument('--markdown', help='Optional Markdown summary path')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or (args.markdown and Path(args.markdown).exists()):
        parser.error('Outputs exist; choose new paths')
    specs = []
    for raw in args.run:
        label, separator, path = raw.partition('=')
        if not separator or not label or not path:
            parser.error('--run must use label=path')
        specs.append((label, path))
    labels = [label for label, _ in specs]
    if len(set(labels)) != len(labels):
        parser.error('Run labels must be distinct')
    pairs = [tuple(raw.split(',')) for raw in args.pair]
    if any(len(pair) != 2 or not set(pair) <= set(labels) for pair in pairs):
        parser.error('--pair must name two run labels: A,B')

    runs = {label: load_run(path) for label, path in specs}
    common = np.array(sorted(set.intersection(*(set(run['rows'].tolist()) for run in runs.values()))))
    if not len(common):
        parser.error('The runs share no validation rows')
    data = ChessData(args.data_dir, args.context_length, verify_hashes=False)
    manifests = {meta['manifest_hash'] for run in runs.values() for meta in run['meta']}
    if manifests != {data.manifest_hash}:
        parser.error('Run files and --data-dir use different dataset manifests')
    families = strata(data, common)
    weights = bootstrap_weights(len(common), args.bootstrap, args.seed)
    sums = {}
    report = dict(row_count=len(common), rows_per_run={label: len(run['rows']) for label, run in runs.items()},
                  bootstrap_replicates=args.bootstrap, bootstrap_seed=args.seed,
                  runs={label: dict(files=run['paths'], execution=run['meta'][0].get('execution'),
                                    checkpoint_step=run['meta'][0].get('checkpoint_step'),
                                    recurrence_mode=run['meta'][0].get('recurrence_mode'),
                                    cell=[run['meta'][0].get('u_t'), run['meta'][0].get('u_d')],
                                    depth_steps=run['meta'][0].get('depth_steps'))
                        for label, run in runs.items()},
                  nll={}, pairs={f'{a} - {b}': {} for a, b in pairs})
    for label, run in runs.items():
        selected = np.searchsorted(run['rows'], common)
        losses = run['losses'][selected].astype(np.float64)
        report['runs'][label]['accuracy'] = float(run['correct'][selected].mean())
        sums[label] = {family: row_sums(losses, codes, len(names)) for family, (codes, names) in families.items()}
    for family, (_, names) in families.items():
        counts = sums[labels[0]][family][1]
        totals = counts.sum(axis=0)
        report['nll'][family] = {
            name: dict(count=int(totals[index]), **{
                label: _float(sums[label][family][0][:, index].sum() / totals[index]) if totals[index] else None
                for label in labels})
            for index, name in enumerate(names)}
        for left, right in pairs:
            report['pairs'][f'{left} - {right}'][family] = dict(zip(names, paired_difference(
                sums[left][family][0], sums[right][family][0], counts, weights)))
    legality_labels = [label for label in labels if 'moves' in runs[label]]
    if legality_labels:
        legality_pairs = [pair for pair in pairs if set(pair) <= set(legality_labels)]
        report['legality'] = legality_tables(runs, legality_labels, legality_pairs,
                                             weights, common.tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.markdown:
        Path(args.markdown).write_text(markdown(report) + '\n')
    for label in labels:
        print(f"{label}: NLL {report['nll']['overall']['all'][label]:.5f} on {len(common)} rows")
    for pair, families_ in report['pairs'].items():
        item = families_['overall']['all']
        print(f"{pair}: {item['difference']:+.5f} [{item['ci_low']:+.5f}, {item['ci_high']:+.5f}]")


if __name__ == '__main__':
    main()
