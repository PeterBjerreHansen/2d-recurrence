"""Evaluate a recurrent checkpoint on fixed data and distinct write placements.

Run with: python -m evaluation.recurrence_grid --checkpoint ... --output ...
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

import torch
import torch.nn.functional as F

from data_loader import ChessData, file_hash
from evaluation.panels import fixed_panel_batches, load_panel
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import (RecurrenceSchedule, update_probability_map_at_step,
                                 update_probabilities_at_step)
from training_utils import provenance

PILOT_SUPPORT = (0, 1, 3)


def supported_evaluation_cells(config, support=PILOT_SUPPORT):
    """Return the recurrence cells available to a configured model mode."""
    support = tuple(support)
    if config.recurrence_mode == 'hybrid':
        return list(itertools.product(support, repeat=2))
    if config.recurrence_mode == 'temporal':
        return [(u_t, 0) for u_t in support]
    if config.recurrence_mode == 'depth':
        return [(0, u_d) for u_d in support]
    raise ValueError(f'Unsupported recurrence_mode: {config.recurrence_mode}')


def validate_evaluation_cells(config, cells=None, support=PILOT_SUPPORT):
    """Resolve the default grid or validate a caller-selected ordered cell list.

    Explicit lists must begin with ``(0, 0)`` because reported NLL deltas and
    prediction-change rates use that cell as their common reference.
    """
    if cells is None:
        return supported_evaluation_cells(config, support)
    if not isinstance(cells, (list, tuple)) or not cells:
        raise ValueError('cells must be a nonempty ordered list of (u_t, u_d) pairs')
    resolved = []
    seen = set()
    for cell in cells:
        if (not isinstance(cell, (list, tuple)) or len(cell) != 2 or
                any(type(count) is not int or count < 0 for count in cell)):
            raise ValueError('Each evaluation cell must contain two nonnegative integer update counts')
        cell = tuple(cell)
        if cell in seen:
            raise ValueError(f'Duplicate evaluation cell: {cell}')
        seen.add(cell)
        u_t, u_d = cell
        if ((config.recurrence_mode == 'temporal' and u_d != 0) or
                (config.recurrence_mode == 'depth' and u_t != 0)):
            raise ValueError(f'Evaluation cell {cell} is incompatible with recurrence_mode={config.recurrence_mode!r}')
        if config.recurrence_mode not in ('hybrid', 'temporal', 'depth'):
            raise ValueError(f'Unsupported recurrence_mode: {config.recurrence_mode}')
        resolved.append(cell)
    if resolved[0] != (0, 0):
        raise ValueError('Explicit evaluation cells must begin with the reference cell (0, 0)')
    return resolved


def distinct_schedules(u_t, u_d, seeds):
    """Sample placements without replacement; deterministic cells run only once."""
    slots = max(u_t, u_d)
    masks = lambda count: [tuple(i in chosen for i in range(slots))
                           for chosen in itertools.combinations(range(slots), count)]
    remaining = [RecurrenceSchedule(t, d) for t in masks(u_t) for d in masks(u_d)]
    selected = []
    for seed in seeds:
        if not remaining:
            break
        selected.append((seed, remaining.pop(random.Random(seed).randrange(len(remaining)))))
    return selected


def compute_estimate(config, schedule, length):
    """Forward matrix-multiply FLOPs per sequence; multiply-add counts as two."""
    d = config.n_embd
    blocks = config.n_prelude + schedule.rounds * (config.n_buffer + config.n_core) + (schedule.u_t + 1) * config.n_source + config.n_coda
    # A write after pass i is consumed from pass i+1 onward, even while held.
    def reads(mask):
        return len(mask) - mask.index(True) if any(mask) else 0
    temporal_reads, depth_reads = reads(schedule.temporal_write_mask), reads(schedule.depth_write_mask)
    flops = (blocks * (24 * length * d * d + 4 * length * length * d)
             + temporal_reads * 12 * length * d * d
             + depth_reads * 4 * length * d * d
             + 2 * length * d * config.vocab_size)
    return dict(block_applications=blocks, temporal_reads=temporal_reads, depth_reads=depth_reads,
                estimated_forward_matmul_flops_per_sequence=flops)


def trajectory_diagnostics(model, x, y, schedule):
    """One eval-mode backward probe; autograd.grad leaves parameter .grad untouched."""
    core_end = model.config.core_end
    source_index = model.config.temporal_source_output_index
    norms = {
        'prelude_output_rms': [], 'core_output_rms': [], 'source_output_rms': [],
        'mixer_input_rms': {'temporal': [], 'depth': []},
        'mixer_output_rms': {'temporal': [], 'depth': []},
        'successive_pass_relative_change': {'core': [], 'source': []},
        'sampled_cross_token_cosine_similarity': {'core': [], 'source': []},
        'gate_value_contributions': {'temporal': [], 'depth': []},
    }
    handles = []
    previous = {'core': None, 'source': None}

    def rms(value):
        value = value.detach().float()
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite diagnostic activation')
        return value.square().mean().sqrt().item()

    def sampled_cosine(value):
        value = value.detach().float()
        if value.shape[1] < 2:
            return None
        stride = max(1, (value.shape[1] - 1) // 128)
        left, right = value[:, :-1:stride], value[:, 1::stride]
        return F.cosine_similarity(left, right, dim=-1).mean().item()

    def relative(value, previous_value):
        if previous_value is None:
            return None
        value = value.detach().float()
        previous_value = previous_value.detach().float()
        return ((value - previous_value).square().mean().sqrt() /
                (previous_value.square().mean().sqrt() + 1e-8)).item()

    def record_block(group, output):
        if group == 'prelude':
            norms['prelude_output_rms'].append(rms(output))
            return
        norms[f'{group}_output_rms'].append(rms(output))
        norms['successive_pass_relative_change'][group].append(relative(output, previous[group]))
        norms['sampled_cross_token_cosine_similarity'][group].append(sampled_cosine(output))
        previous[group] = output.detach()

    if model.config.n_prelude:
        handles.append(model.transformer.h[model.config.n_prelude - 1].register_forward_hook(
            lambda module, inputs, output: record_block('prelude', output)))
    handles.append(model.transformer.h[core_end].register_forward_hook(
        lambda module, inputs, output: record_block('core', output)))
    handles.append(model.transformer.h[source_index].register_forward_hook(
        lambda module, inputs, output: record_block('source', output)))

    def record_temporal(module, inputs, output):
        prelude, shifted_memory = inputs
        with torch.no_grad():
            memory = module.memory_norm(shifted_memory)
            anchor = module.prelude_norm(prelude)
            alpha, beta = module.gates(torch.cat((memory, anchor), dim=-1)).sigmoid().chunk(2, dim=-1)
            memory_value = module.memory_value(memory)
            prelude_value = module.prelude_value(anchor)
            memory_contribution = alpha * memory_value
            prelude_contribution = beta * prelude_value
            norms['mixer_input_rms']['temporal'].append(
                dict(prelude=rms(prelude), shifted_memory=rms(shifted_memory)))
            norms['mixer_output_rms']['temporal'].append(rms(output))
            norms['gate_value_contributions']['temporal'].append(dict(
                alpha_mean=alpha.float().mean().item(), beta_mean=beta.float().mean().item(),
                memory_value_rms=rms(memory_value), prelude_value_rms=rms(prelude_value),
                memory_contribution_rms=rms(memory_contribution),
                prelude_contribution_rms=rms(prelude_contribution)))

    def record_depth(module, inputs, output):
        state, anchor = inputs
        with torch.no_grad():
            state_value = module.state_value(module.state_norm(state))
            anchor_value = module.anchor_value(module.anchor_norm(anchor))
            norms['mixer_input_rms']['depth'].append(
                dict(state=rms(state), anchor=rms(anchor)))
            norms['mixer_output_rms']['depth'].append(rms(output))
            norms['gate_value_contributions']['depth'].append(dict(
                state_value_rms=rms(state_value), anchor_value_rms=rms(anchor_value),
                state_contribution_rms=rms(state_value), anchor_contribution_rms=rms(anchor_value)))

    if model.config.uses_temporal_recurrence:
        handles.append(model.temporal_mixer.register_forward_hook(record_temporal))
    if model.config.uses_depth_recurrence:
        handles.append(model.depth_mixer.register_forward_hook(record_depth))
    try:
        with torch.enable_grad():
            _, loss = model(x, y, schedule=schedule)
            named = list(model.named_parameters())
            gradients = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
        groups = {'temporal_mixer': None, 'depth_mixer': None}
        for (name, _), gradient in zip(named, gradients):
            if name.startswith('transformer.h.'):
                index = int(name.split('.')[2])
                group = ('prelude' if index < model.config.n_prelude else
                         'buffer' if index < model.config.core_start else
                         'core' if index < model.config.core_stop else
                         'source' if index < model.config.source_stop else 'coda')
            else:
                group = name.split('.')[0] if 'mixer' in name else 'embedding_and_head'
            groups.setdefault(group, None)
            if gradient is not None:
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f'Non-finite gradient: {name}')
                groups[group] = (groups[group] or 0.0) + gradient.float().square().sum().item()
        return dict(**norms, gradient_l2={k: None if v is None else v ** .5 for k, v in groups.items()},
                    loss=loss.item(), mode='eval', batch_count=1,
                    diagnostic_scope='one fixed validation batch; observational eval-mode probe')
    finally:
        for handle in handles:
            handle.remove()


def evaluate_grid(model, data, *, batches=8, batch_size=2, data_seed=2027,
                  mask_seeds=(11, 23, 37), training_update_probabilities=None, diagnostics=True,
                  fixed_batches=None, sampling=None, cells=None, optional_cells=()):
    if batches < 1 or batch_size < 1 or not mask_seeds or len(set(mask_seeds)) != len(mask_seeds):
        raise ValueError('Positive batch counts and nonempty distinct mask seeds are required')
    device = next(model.parameters()).device
    if fixed_batches is None:
        generator = torch.Generator().manual_seed(data_seed)
        fixed = [data.batch('val', batch_size, 'cpu', generator) for _ in range(batches)]
        sampling = sampling or 'fixed row-aligned batches sampled with replacement'
    else:
        fixed = list(fixed_batches)
        if not fixed:
            raise ValueError('fixed_batches must be nonempty')
        batches = len(fixed)
        sampling = sampling or 'caller-provided fixed batches'
    digest = hashlib.sha256()
    for x, y in fixed:
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    selected_cells = validate_evaluation_cells(model.config, cells)
    optional_cells = {tuple(cell) for cell in optional_cells}
    if ((0, 0) in optional_cells or not optional_cells <= set(selected_cells)):
        raise ValueError('optional_cells must be selected diagnostic cells, not the (0, 0) reference')
    if diagnostics and any(max(cell) > 7 for cell in selected_cells):
        raise ValueError('Sixteen-pass grids require diagnostics=False; use evaluation.stress_checks for per-pass RMS checks')
    was_training = model.training
    model.eval()
    results, failed_cells, baseline_predictions = [], [], []
    try:
        for u_t, u_d in selected_cells:
            placements = []
            cell_failure = None
            for seed, schedule in distinct_schedules(u_t, u_d, mask_seeds):
                total_loss, correct, changed, count = 0., 0, 0, 0
                with torch.no_grad():
                    for index, (cpu_x, cpu_y) in enumerate(fixed):
                        x, y = cpu_x.to(device), cpu_y.to(device)
                        try:
                            logits, loss = model(x, y, schedule=schedule)
                        except torch.cuda.OutOfMemoryError as error:
                            if (u_t, u_d) not in optional_cells:
                                raise
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            cell_failure = dict(u_t=u_t, u_d=u_d,
                                                error_type=type(error).__name__, error=str(error))
                            break
                        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                            error = FloatingPointError(f'Non-finite grid cell {(u_t, u_d)}')
                            if (u_t, u_d) not in optional_cells:
                                raise error
                            cell_failure = dict(u_t=u_t, u_d=u_d,
                                                error_type=type(error).__name__, error=str(error))
                            break
                        predictions = logits.argmax(-1).cpu()
                        if (u_t, u_d) == (0, 0):
                            baseline_predictions.append(predictions)
                        changed += (predictions != baseline_predictions[index]).sum().item()
                        total_loss += loss.item() * y.numel()
                        correct += (predictions == cpu_y).sum().item()
                        count += y.numel()
                if cell_failure:
                    break
                placements.append(dict(mask_seed=seed, temporal_write_mask=schedule.temporal_write_mask,
                                       depth_write_mask=schedule.depth_write_mask, nll=total_loss / count,
                                       accuracy=correct / count, prediction_change_rate=changed / count,
                                       **compute_estimate(model.config, schedule, data.context_length)))
            if cell_failure:
                failed_cells.append(cell_failure)
                continue
            stats = {}
            for key in ['nll', 'accuracy', 'prediction_change_rate', 'estimated_forward_matmul_flops_per_sequence']:
                values = [p[key] for p in placements]
                stats[key + '_mean'] = statistics.mean(values)
                stats[key + '_std'] = statistics.pstdev(values)
            cell = dict(u_t=u_t, u_d=u_d, core_passes=max(u_t, u_d) + 1,
                        training_update_probability=(training_update_probabilities.get((u_t, u_d), 0.)
                                                     if training_update_probabilities is not None else None),
                        possible_placements=math.comb(max(u_t, u_d), u_t) * math.comb(max(u_t, u_d), u_d),
                        placement_count=len(placements), evaluated_characters_per_placement=count,
                        target_count=count,
                        **stats, placements=placements)
            cell['all_placements_evaluated'] = len(placements) == cell['possible_placements']
            cell['nll_delta_vs_00'] = stats['nll_mean'] - (results[0]['nll_mean'] if results else stats['nll_mean'])
            if diagnostics:
                first = placements[0]
                schedule = RecurrenceSchedule(first['temporal_write_mask'], first['depth_write_mask'])
                cell['diagnostics'] = dict(mask_seed=first['mask_seed'], **trajectory_diagnostics(
                    model, fixed[0][0].to(device), fixed[0][1].to(device), schedule))
            results.append(cell)
    finally:
        model.train(was_training)
    target_count = sum(y.numel() for _, y in fixed)
    return dict(execution='training_graph', split='val', data_seed=data_seed, mask_seeds=list(mask_seeds),
                sampling=sampling, batch_size=batch_size,
                batches=batches, batch_sha256=digest.hexdigest(), batch_fingerprint=digest.hexdigest(),
                target_count=target_count, context_length=data.context_length,
                variation='population standard deviation across distinct mask placements, not training seeds or batches',
                compute_convention='Forward matmul estimate per sequence, multiply-add=2; dense T-by-T attention. '
                'Includes transformer projections/MLPs, attention, temporal gates/values, depth values, LM head. '
                'Excludes normalization, softmax, activations, elementwise operations, backward, and kernel overhead; '
                'not measured hardware FLOPs or a complete compute comparison.', cells=results,
                failed_cells=failed_cells)


def evaluate_checkpoint(checkpoint_path, *, device='cpu', dataset=None, panel_file=None,
                        panel_split='selection', **kwargs):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = file_hash(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['config'].get('architecture') != 'recurrent':
        raise ValueError('The recurrence grid requires a recurrent checkpoint')
    config = checkpoint['config']
    data = ChessData(Path('data') / (dataset or config['dataset']), checkpoint['model_args']['block_size'])
    if data.manifest_hash != checkpoint['manifest_hash'] or data.meta != checkpoint['meta']:
        raise ValueError('Evaluation data or vocabulary differs from the training checkpoint')
    model = Recurrent2DGPT(RecurrentGPTConfig.from_checkpoint(checkpoint['model_args'])).to(device)
    model.load_state_dict(checkpoint['model'])
    active_matrix = update_probabilities_at_step(config, checkpoint['iter_num'])
    probabilities = update_probability_map_at_step(config, checkpoint['iter_num'])
    panel = None
    if panel_file:
        panel = load_panel(panel_file, data, split=panel_split)
        fixed, fixed_metadata = fixed_panel_batches(data, panel, kwargs.pop('batch_size', 2))
        kwargs.update(fixed_batches=fixed, batch_size=fixed_metadata['batch_size'], data_seed=None,
                      sampling=fixed_metadata['sampling'])
    report = evaluate_grid(model, data, training_update_probabilities=probabilities, **kwargs)
    if file_hash(checkpoint_path) != checkpoint_hash:
        raise ValueError('Checkpoint changed during evaluation; use a retained step checkpoint')
    report.update(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_hash,
                  checkpoint_step=checkpoint['iter_num'], manifest_hash=data.manifest_hash,
                  model_args=checkpoint['model_args'], training_seed=config['seed'],
                  recurrence_mode=RecurrentGPTConfig.from_checkpoint(checkpoint['model_args']).recurrence_mode,
                  training_schedule_seed=config['recurrence_seed'], device=device,
                  dtype='float32', provenance=provenance())
    report['next_update_probability_step'] = checkpoint['iter_num']
    report['next_update_probability_matrix'] = [list(row) for row in active_matrix]
    report['last_update_probability_matrix'] = (
        [list(row) for row in update_probabilities_at_step(config, checkpoint['iter_num'] - 1)]
        if checkpoint['iter_num'] > 0 else None)
    report['dataset_identity'] = dict(dataset=config['dataset'], manifest_hash=data.manifest_hash,
                                      validation_row_count=len(data.rows['val']),
                                      training_row_count=len(data.rows['train']))
    if panel:
        report.update(panel_file=panel['path'], panel_file_sha256=panel['sha256'],
                      panel_split=panel['split'], row_indices=panel['row_indices'],
                      row_count=panel['row_count'], target_count=report['target_count'],
                      batch_fingerprint=report['batch_sha256'], fixed_panel=fixed_metadata)
    return report


def _parse_cell(value):
    try:
        pair = tuple(int(part) for part in value.split(','))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('cells must use U_T,U_D integer syntax') from None
    if len(pair) != 2 or any(count < 0 for count in pair):
        raise argparse.ArgumentTypeError('cells must use two nonnegative counts: U_T,U_D')
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='JSON report; a flat CSV is written alongside it')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dataset', help='Optional relocated data directory, checked against the saved manifest')
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--data-seed', type=int, default=2027)
    parser.add_argument('--panel-file', help='Frozen validation panel JSON')
    parser.add_argument('--panel-split', choices=['selection', 'confirmation'], default='selection')
    parser.add_argument('--mask-seeds', type=int, nargs='+', default=[11, 23, 37])
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--no-diagnostics', action='store_true',
                        help='Disable gradient diagnostics; required for 16-pass cells, which use stress_checks')
    parser.add_argument('--cell', action='append', type=_parse_cell,
                        help='Explicit evaluation cell U_T,U_D; repeat in order and begin with 0,0')
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix != '.json':
        parser.error('--output must end with .json')
    if output.exists() or output.with_suffix('.csv').exists():
        parser.error('Output exists; choose a new report path')
    torch.set_num_threads(args.num_threads)
    report = evaluate_checkpoint(args.checkpoint, device=args.device, dataset=args.dataset,
                                 batches=args.batches, batch_size=args.batch_size, data_seed=args.data_seed,
                                 panel_file=args.panel_file, panel_split=args.panel_split,
                                 mask_seeds=args.mask_seeds, diagnostics=not args.no_diagnostics,
                                 cells=args.cell)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    rows = [{k: v for k, v in cell.items() if k not in ['placements', 'diagnostics']} for cell in report['cells']]
    with output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for cell in report['cells']:
        print(f"({cell['u_t']},{cell['u_d']}): NLL {cell['nll_mean']:.5f} ± {cell['nll_std']:.5f}, "
              f"accuracy {cell['accuracy_mean']:.3f}, ΔNLL {cell['nll_delta_vs_00']:+.5f}")


if __name__ == '__main__':
    main()
