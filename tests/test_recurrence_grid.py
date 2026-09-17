import copy
import itertools
import random

import pytest
import torch

from data_loader import ChessData
from evaluation.recurrence_grid import compute_estimate, distinct_schedules, evaluate_checkpoint, evaluate_grid
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import RecurrenceSchedule, sample_schedule
from sample import generate
from train import train


def model():
    torch.manual_seed(17)
    return Recurrent2DGPT(RecurrentGPTConfig(n_layer=4, n_prelude=1, n_core=1,
                                            n_coda=1, n_head=2, n_embd=8, block_size=8))


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
                           training_probabilities={(0, 0): .1})
    assert net.training and torch.equal(rng, torch.get_rng_state())
    assert torch.equal(first_parameter.grad, torch.ones_like(first_parameter))
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    assert len(report['cells']) == 9
    baseline = report['cells'][0]
    assert baseline['training_probability'] == .1
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
                          training_probabilities={(0, 0): .1})
    assert again['batch_sha256'] == report['batch_sha256']
    for cell, original in zip(again['cells'], report['cells']):
        assert cell == {k: v for k, v in original.items() if k != 'diagnostics'}


def test_snapshot_checkpoint_evaluation_and_manifest_guard(prepared_data, tmp_path):
    config = dict(architecture='recurrent', recurrence_support=[0, 1, 3],
                  recurrence_probabilities=[[.1, .12, .04], [.12, .26, .08], [.04, .08, .16]],
                  n_layer=4, n_prelude=1, n_core=1, n_coda=1, n_head=2, n_embd=8,
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
    assert sum(c['training_probability'] for c in report['cells']) == pytest.approx(1.)
    broken = torch.load(latest, weights_only=False)
    broken['manifest_hash'] = 'wrong'
    torch.save(broken, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='differs'):
        evaluate_checkpoint(tmp_path / 'bad.pt', batches=1)


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
