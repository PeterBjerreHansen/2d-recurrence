import copy
import itertools
import json
import random

import pytest
import torch

from data_loader import ChessData
from evaluation.feedback_diagnostic import donor_permutation
from evaluation.panels import fixed_panel_batches, load_panel
from evaluation.recurrence_grid import compute_estimate, distinct_schedules, evaluate_checkpoint, evaluate_grid
from evaluation.stress_checks import check as run_stress_checks
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import RecurrenceSchedule, sample_schedule
from sample import generate
from train import train


def test_feedback_donor_permutation_preserves_actual_identity_including_final_row():
    source_outputs = torch.tensor([10., 11., 20., 21.])
    swapped = source_outputs[torch.tensor(donor_permutation(2))]
    assert swapped[:2].tolist() == [20., 21.]
    assert swapped[2:].tolist() == [10., 11.]
    final_batch = torch.tensor([30., 40.])
    assert final_batch[torch.tensor(donor_permutation(1))].tolist() == [40., 30.]


def model(recurrence_mode='hybrid'):
    torch.manual_seed(17)
    return Recurrent2DGPT(RecurrentGPTConfig(n_layer=4, n_prelude=1, n_buffer=0, n_core=1,
                                            n_coda=1, n_head=2, n_embd=8, block_size=8,
                                            recurrence_mode=recurrence_mode))


def test_distinct_placements_cover_pilot_without_fake_replication():
    for t, d in itertools.product([0, 1, 3], repeat=2):
        selected = distinct_schedules(t, d, [11, 23, 37])
        schedules = [s for _, s in selected]
        assert len(schedules) == (3 if (t, d) in [(1, 3), (3, 1)] else 1)
        assert len(set(schedules)) == len(schedules)
        assert all((s.u_t, s.u_d) == (t, d) for s in schedules)
        assert selected == distinct_schedules(t, d, [11, 23, 37])


def test_compute_counts_held_reads_not_just_writes():
    config = model().config
    early = RecurrenceSchedule((True, False, False), (True, True, True))
    late = RecurrenceSchedule((False, False, True), (True, True, True))
    a, b = [compute_estimate(config, s, 8) for s in [early, late]]
    assert a['block_applications'] == b['block_applications'] == 8
    assert a['temporal_reads'] == 3 and b['temporal_reads'] == 1
    assert a['depth_reads'] == b['depth_reads'] == 3
    assert a['estimated_forward_matmul_flops_per_sequence'] - b['estimated_forward_matmul_flops_per_sequence'] == 2 * 12 * 8 * 8 * 8
    plain = compute_estimate(config, sample_schedule(0, 0, random.Random(0)), 8)
    # Four blocks, each 24*T*D^2 + 4*T^2*D, plus the full LM head.
    assert plain['estimated_forward_matmul_flops_per_sequence'] == 4 * (24 * 8 * 64 + 4 * 64 * 8) + 2 * 8 * 8 * 32


def test_grid_fixed_data_metrics_diagnostics_and_no_mutation(prepared_data):
    net = model().train()
    data = ChessData(prepared_data, 8)
    parameters = copy.deepcopy(net.state_dict())
    first_parameter = next(net.parameters())
    first_parameter.grad = torch.ones_like(first_parameter)
    rng = torch.get_rng_state().clone()
    report = evaluate_grid(net, data, batches=2, batch_size=2,
                           training_update_probabilities={(0, 0): .1})
    assert net.training and torch.equal(rng, torch.get_rng_state())
    assert torch.equal(first_parameter.grad, torch.ones_like(first_parameter))
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    assert len(report['cells']) == 9
    baseline = report['cells'][0]
    assert baseline['training_update_probability'] == .1
    assert baseline['nll_std'] == baseline['prediction_change_rate_mean'] == 0
    assert baseline['diagnostics']['gradient_l2']['temporal_mixer'] is None
    assert baseline['diagnostics']['gradient_l2']['depth_mixer'] is None
    for cell in report['cells']:
        assert len(cell['diagnostics']['core_output_rms']) == cell['core_passes']
        assert len(cell['diagnostics']['source_output_rms']) == cell['u_t'] + 1
        assert cell['evaluated_characters_per_placement'] == 32
        assert cell['nll_delta_vs_00'] == pytest.approx(cell['nll_mean'] - baseline['nll_mean'])
    generator = torch.Generator().manual_seed(2027)
    losses, accuracies = [], []
    net.eval()
    with torch.no_grad():
        for _ in range(2):
            x, y = data.batch('val', 2, 'cpu', generator)
            logits, _ = net(x, y, schedule=sample_schedule(0, 0, random.Random(0)))
            losses.append(torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
            accuracies.append((logits.argmax(-1) == y).float().mean().item())
    assert baseline['nll_mean'] == pytest.approx(sum(losses) / 2)
    assert baseline['accuracy_mean'] == pytest.approx(sum(accuracies) / 2)
    again = evaluate_grid(net, data, batches=2, batch_size=2, diagnostics=False,
                          training_update_probabilities={(0, 0): .1})
    assert again['batch_sha256'] == report['batch_sha256']
    for cell, original in zip(again['cells'], report['cells']):
        assert cell == {k: v for k, v in original.items() if k != 'diagnostics'}


@pytest.mark.parametrize('mode, cells', [
    ('hybrid', [(0, 0), (15, 0), (0, 15), (15, 15)]),
    ('temporal', [(0, 0), (15, 0)]),
    ('depth', [(0, 0), (0, 15)]),
])
def test_explicit_sixteen_pass_cells_are_evaluated_in_requested_order(prepared_data, mode, cells):
    net = model(mode).eval()
    data = ChessData(prepared_data, 8)

    report = evaluate_grid(net, data, batches=1, batch_size=1, mask_seeds=[11],
                           diagnostics=False, cells=cells)

    assert [(cell['u_t'], cell['u_d']) for cell in report['cells']] == cells
    assert [cell['core_passes'] for cell in report['cells']] == [
        max(u_t, u_d) + 1 for u_t, u_d in cells]
    assert all(torch.isfinite(torch.tensor(cell['nll_mean'])) for cell in report['cells'])


def test_optional_extrapolation_nonfinite_cell_is_recorded_without_losing_primary_metrics(
        prepared_data, monkeypatch):
    net = model('temporal').eval()
    data = ChessData(prepared_data, 8)
    original_forward = net.forward

    def nonfinite_at_sixteen_passes(x, y=None, *, schedule=None):
        logits, loss = original_forward(x, y, schedule=schedule)
        if len(schedule.temporal_write_mask) == 15:
            logits = logits * float('nan')
        return logits, loss

    monkeypatch.setattr(net, 'forward', nonfinite_at_sixteen_passes)

    report = evaluate_grid(
        net, data, batches=1, batch_size=1, mask_seeds=[11], diagnostics=False,
        cells=[(0, 0), (3, 0), (15, 0)], optional_cells=[(15, 0)])

    assert [(cell['u_t'], cell['u_d']) for cell in report['cells']] == [(0, 0), (3, 0)]
    assert torch.isfinite(torch.tensor(report['cells'][1]['nll_mean']))
    assert report['failed_cells'] == [{
        'u_t': 15, 'u_d': 0, 'error_type': 'FloatingPointError',
        'error': 'Non-finite grid cell (15, 0)',
    }]


def test_nonfinite_primary_grid_cell_remains_fatal_even_when_other_cells_are_optional(
        prepared_data, monkeypatch):
    net = model('temporal').eval()
    data = ChessData(prepared_data, 8)
    original_forward = net.forward

    def nonfinite_at_four_passes(x, y=None, *, schedule=None):
        logits, loss = original_forward(x, y, schedule=schedule)
        if len(schedule.temporal_write_mask) == 3:
            logits = logits * float('nan')
        return logits, loss

    monkeypatch.setattr(net, 'forward', nonfinite_at_four_passes)

    with pytest.raises(FloatingPointError, match=r'Non-finite grid cell \(3, 0\)'):
        evaluate_grid(
            net, data, batches=1, batch_size=1, mask_seeds=[11], diagnostics=False,
            cells=[(0, 0), (3, 0), (15, 0)], optional_cells=[(15, 0)])


def test_optional_extrapolation_oom_is_recorded_without_losing_primary_metrics(
        prepared_data, monkeypatch):
    net = model('temporal').eval()
    data = ChessData(prepared_data, 8)
    original_forward = net.forward

    def out_of_memory_at_sixteen_passes(x, y=None, *, schedule=None):
        if len(schedule.temporal_write_mask) == 15:
            raise torch.cuda.OutOfMemoryError('simulated diagnostic OOM')
        return original_forward(x, y, schedule=schedule)

    monkeypatch.setattr(net, 'forward', out_of_memory_at_sixteen_passes)

    report = evaluate_grid(
        net, data, batches=1, batch_size=1, mask_seeds=[11], diagnostics=False,
        cells=[(0, 0), (3, 0), (15, 0)], optional_cells=[(15, 0)])

    assert [(cell['u_t'], cell['u_d']) for cell in report['cells']] == [(0, 0), (3, 0)]
    assert report['failed_cells'][0]['error_type'] == 'OutOfMemoryError'


def test_explicit_evaluation_cells_reject_incompatible_or_ambiguous_lists(prepared_data):
    net = Recurrent2DGPT(RecurrentGPTConfig(
        n_layer=4, n_prelude=1, n_buffer=0, n_core=1, n_source=1, n_coda=1,
        n_head=2, n_embd=8, block_size=8, recurrence_mode='temporal'))
    data = ChessData(prepared_data, 8)

    with pytest.raises(ValueError, match='recurrence_mode'):
        evaluate_grid(net, data, batches=1, batch_size=1, diagnostics=False,
                      cells=[(0, 0), (0, 15)])
    with pytest.raises(ValueError, match='Duplicate'):
        evaluate_grid(net, data, batches=1, batch_size=1, diagnostics=False,
                      cells=[(0, 0), (0, 0)])
    with pytest.raises(ValueError, match='reference cell'):
        evaluate_grid(net, data, batches=1, batch_size=1, diagnostics=False,
                      cells=[(1, 0), (0, 0)])
    with pytest.raises(ValueError, match='nonnegative integer'):
        evaluate_grid(net, data, batches=1, batch_size=1, diagnostics=False,
                      cells=[(0, 0), (True, 0)])


def test_sixteen_pass_grid_requires_memory_safe_diagnostics(prepared_data):
    net = model().eval()
    data = ChessData(prepared_data, 8)

    with pytest.raises(ValueError, match='diagnostics=False'):
        evaluate_grid(net, data, batches=1, batch_size=1,
                      cells=[(0, 0), (15, 15)])


@pytest.mark.parametrize('mode, cells', [
    ('hybrid', ((15, 0), (0, 15), (15, 15))),
    ('temporal', ((15, 0),)),
    ('depth', ((0, 15),)),
])
def test_numerical_stress_check_reports_sixteen_pass_state_trajectories(
        prepared_data, tmp_path, mode, cells):
    data = ChessData(prepared_data, 8)
    selected = [0, 1]
    panel_path = tmp_path / 'panel.json'
    panel_path.write_text(json.dumps({
        'dataset_manifest_hash': data.manifest_hash,
        'validation_row_count': len(data.rows['val']),
        'selection_seed': 2027,
        'selection_indices': selected,
        'confirmation_indices': [i for i in range(len(data.rows['val'])) if i not in selected],
    }))
    panel = load_panel(panel_path, data, split='selection')

    _, checks = run_stress_checks(model(mode).eval(), data, panel, 'cpu')

    by_cell = {(item['u_t'], item['u_d']): item for item in checks}
    for cell in cells:
        assert by_cell[cell]['finite']
        assert by_cell[cell]['expected_passes'] == 16
        assert len(by_cell[cell]['state_rms_by_pass']) == 16


def test_numerical_stress_nonfinite_result_is_serializable_diagnostic(
        prepared_data, tmp_path, monkeypatch):
    data = ChessData(prepared_data, 8)
    panel_path = tmp_path / 'panel.json'
    panel_path.write_text(json.dumps({
        'dataset_manifest_hash': data.manifest_hash,
        'validation_row_count': len(data.rows['val']),
        'selection_seed': 2027,
        'selection_indices': [0, 1],
        'confirmation_indices': [i for i in range(2, len(data.rows['val']))],
    }))
    panel = load_panel(panel_path, data, split='selection')
    net = model('temporal').eval()
    original_forward = net.forward

    def nonfinite_at_sixteen_passes(x, y=None, *, schedule=None):
        logits, loss = original_forward(x, y, schedule=schedule)
        if len(schedule.temporal_write_mask) == 15:
            logits = logits * float('nan')
        return logits, loss

    monkeypatch.setattr(net, 'forward', nonfinite_at_sixteen_passes)

    _, checks = run_stress_checks(net, data, panel, 'cpu')

    sixteen_pass = next(item for item in checks if item['u_t'] == 15)
    assert not sixteen_pass['finite']
    assert sixteen_pass['diagnostic_failure']
    json.dumps(checks, allow_nan=False)


def test_numerical_stress_oom_is_serializable_diagnostic(prepared_data, tmp_path, monkeypatch):
    data = ChessData(prepared_data, 8)
    panel_path = tmp_path / 'panel.json'
    panel_path.write_text(json.dumps({
        'dataset_manifest_hash': data.manifest_hash,
        'validation_row_count': len(data.rows['val']),
        'selection_seed': 2027,
        'selection_indices': [0, 1],
        'confirmation_indices': [i for i in range(2, len(data.rows['val']))],
    }))
    panel = load_panel(panel_path, data, split='selection')
    net = model('temporal').eval()
    original_forward = net.forward

    def out_of_memory_at_sixteen_passes(x, y=None, *, schedule=None):
        if len(schedule.temporal_write_mask) == 15:
            raise torch.cuda.OutOfMemoryError('simulated stress OOM')
        return original_forward(x, y, schedule=schedule)

    monkeypatch.setattr(net, 'forward', out_of_memory_at_sixteen_passes)

    _, checks = run_stress_checks(net, data, panel, 'cpu')

    sixteen_pass = next(item for item in checks if item['u_t'] == 15)
    assert not sixteen_pass['finite']
    assert sixteen_pass['diagnostic_failure'].startswith('OutOfMemoryError:')
    assert sixteen_pass['nll'] is None
    json.dumps(checks, allow_nan=False)


def test_snapshot_checkpoint_evaluation_and_manifest_guard(prepared_data, tmp_path):
    config = dict(architecture='recurrent', update_support=[0, 1, 3],
                  update_probabilities=[[.1, .12, .04], [.12, .26, .08], [.04, .08, .16]],
                  n_layer=4, n_prelude=1, n_buffer=0, n_core=1, n_coda=1, n_head=2, n_embd=8,
                  dataset=str(prepared_data), block_size=8, batch_size=1,
                  gradient_accumulation_steps=1, max_iters=0, eval_interval=1, eval_iters=1,
                  log_interval=1, warmup_iters=0, lr_decay_iters=2, compile=False,
                  device='cpu', dtype='float32', num_threads=1, keep_checkpoints=True,
                  out_dir=str(tmp_path / 'run'))
    latest = train(config)
    initial = latest.parent / 'ckpt-step000000.pt'
    initial_bytes = initial.read_bytes()
    train({**config, 'init_from': 'resume', 'max_iters': 2})
    assert initial.read_bytes() == initial_bytes
    assert (latest.parent / 'ckpt-step000001.pt').exists()
    report = evaluate_checkpoint(latest.parent / 'ckpt-step000002.pt', batches=1, batch_size=1)
    assert report['checkpoint_step'] == 2
    assert report['training_seed'] == 1337
    assert report['recurrence_mode'] == 'hybrid'
    assert sum(c['training_update_probability'] for c in report['cells']) == pytest.approx(1.)
    extended = evaluate_checkpoint(
        latest.parent / 'ckpt-step000002.pt', batches=1, batch_size=1,
        mask_seeds=[11], diagnostics=False, cells=[(0, 0), (15, 15)])
    assert [(cell['u_t'], cell['u_d']) for cell in extended['cells']] == [(0, 0), (15, 15)]
    assert extended['cells'][-1]['core_passes'] == 16
    broken = torch.load(latest, weights_only=False)
    broken['manifest_hash'] = 'wrong'
    torch.save(broken, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='differs'):
        evaluate_checkpoint(tmp_path / 'bad.pt', batches=1)


def test_frozen_panel_isolation_and_exact_partial_batch(prepared_data, tmp_path):
    data = ChessData(prepared_data, 8)
    validation_count = len(data.rows['val'])
    selection = [0, 2, 4]
    confirmation = [index for index in range(validation_count) if index not in selection]
    panel_path = tmp_path / 'panels.json'
    panel_path.write_text(__import__('json').dumps({
        'dataset_manifest_hash': data.manifest_hash,
        'validation_row_count': validation_count,
        'selection_seed': 2027,
        'selection_indices': selection,
        'confirmation_indices': confirmation,
    }))
    panel = load_panel(panel_path, data, 'selection')
    batches, metadata = fixed_panel_batches(data, panel, batch_size=2)
    assert metadata['row_indices'] == selection
    assert metadata['batch_count'] == 2
    assert metadata['target_count'] == len(selection) * data.context_length
    assert [batch[0].shape[0] for batch in batches] == [2, 1]
    flattened = torch.cat([batch[0][:, 0] for batch in batches]).tolist()
    expected = [int(data.rows['val'][index, 0]) for index in selection]
    assert flattened == expected


def test_recurrent_generation_labels_and_fixed_schedule(prepared_data):
    net = model()
    meta = ChessData(prepared_data, 8).meta
    schedule = sample_schedule(1, 3, random.Random(11))
    schedules = []
    handle = net.register_forward_pre_hook(lambda module, args, kwargs: schedules.append(kwargs['schedule']), with_kwargs=True)
    try:
        result = generate(net, meta, max_new_tokens=3, temperature=0, schedule=schedule)
    finally:
        handle.remove()
    assert result['execution'] == 'training_graph'
    assert result['prefill'] == 'full_prefix_recomputation_each_character'
    assert result['depth_budget'] == dict(kind='training_graph_core_passes', value=4)
    assert schedules and all(s == schedule for s in schedules)
    assert result['temporal_write_mask'] == schedule.temporal_write_mask
