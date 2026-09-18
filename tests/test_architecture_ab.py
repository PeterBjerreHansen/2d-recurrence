"""Architectural boundaries, held-state semantics, and exact spot recovery."""
import copy
from collections import Counter
import random

import pytest
import torch

from evaluation.recurrence_grid import compute_estimate, trajectory_diagnostics
from experiments.ablations.architecture_sites.run import configuration, training_rows
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import RecurrenceSchedule, sample_schedule
from train import train


def make_model(variant):
    settings = dict(n_layer=8, n_prelude=1, n_coda=1, n_head=2, n_embd=8,
                    block_size=6, vocab_size=32, dropout=0.0)
    layouts = {
        'separated': dict(n_buffer=1, n_core=4, n_source=1),
        'coincident': dict(n_buffer=0, n_core=6, n_source=0),
        # C keeps a destination buffer but shares the core output as both
        # state candidates, isolating the buffer relative to B.
        'buffered_coincident': dict(n_buffer=1, n_core=5, n_source=0),
    }
    settings.update(layouts[variant])
    return Recurrent2DGPT(RecurrentGPTConfig(**settings))


def reference(model, x, schedule, variant):
    # Deliberately execute all eight physical blocks even for an unused candidate.
    # Select reads from full histories rather than maintaining mutable state.
    layers = model.transformer.h
    p = layers[0](model.embed(x))
    depths, memories = [], []
    for b in range(schedule.rounds):
        tw = [i for i in range(b) if schedule.temporal_write_mask[i]]
        dw = [i for i in range(b) if schedule.depth_write_mask[i]]
        a = p
        if tw:
            m = memories[tw[-1]]
            a = model.temporal_mixer(p, torch.nn.functional.pad(m[:, :-1], (0, 0, 1, 0)))
        if variant == 'separated':
            a = layers[1](a)
        h = model.depth_mixer(depths[dw[-1]], a) if dw else a
        for layer in layers[2:6] if variant == 'separated' else layers[1:7]:
            h = layer(h)
        depths.append(h)
        memories.append(layers[6](h) if variant == 'separated' else h)
    return model.readout(layers[7](memories[-1]), x)


@pytest.mark.parametrize('variant', ['separated', 'coincident'])
@pytest.mark.parametrize('masks', [((True, True, True), (True, False, False)),
                                  ((False, False, True), (True, True, True)),
                                  ((True, False, False), (True, True, True)),
                                  ((False, False, False), (True, True, True)),
                                  ((True, True, True), (False, False, False))])
def test_layout_and_write_masks_match_history_gradients(variant, masks):
    torch.manual_seed(22)
    model = make_model(variant)
    other = copy.deepcopy(model)
    x = torch.randint(32, (2, 6))
    schedule = RecurrenceSchedule(*masks)
    counts = Counter()
    handles = [layer.register_forward_hook(lambda m, a, o, i=i: counts.update([i]))
               for i, layer in enumerate(model.transformer.h)]
    logits, loss = model(x, x, schedule=schedule)
    for handle in handles:
        handle.remove()
    expected, expected_loss = reference(other, x, schedule, variant)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    loss.backward()
    expected_loss.backward()
    for (name, p), (other_name, q) in zip(model.named_parameters(), other.named_parameters()):
        assert name == other_name
        if q.grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    expected_counts = {0: 1, 7: 1, **{i: 4 for i in range(1, 7)}}
    if variant == 'separated':
        expected_counts[6] = schedule.u_t + 1
    assert dict(counts) == expected_counts
    assert compute_estimate(model.config, schedule, 6)['block_applications'] == sum(counts.values())
    changed = x.clone()
    changed[:, 3:] = (changed[:, 3:] + 1) % 32
    alternative, _ = model(changed, changed, schedule=schedule)
    torch.testing.assert_close(logits[:, :3], alternative[:, :3], rtol=0, atol=0)


@pytest.mark.parametrize('variant', ['separated', 'coincident'])
def test_zero_updates_and_source_diagnostics(variant):
    model = make_model(variant)
    base = GPT(GPTConfig(n_layer=8, n_head=2, n_embd=8, block_size=6, vocab_size=32))
    base.transformer.load_state_dict(model.transformer.state_dict())
    x = torch.randint(32, (1, 6))
    actual, _ = model(x, x, schedule=RecurrenceSchedule((), ()))
    expected, _ = base(x, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    diag = trajectory_diagnostics(model, x, x, sample_schedule(0, 3, random.Random(1)))
    assert len(diag['core_output_rms']) == 4
    assert len(diag['source_output_rms']) == (1 if variant == 'separated' else 4)
    if variant == 'coincident':
        assert diag['source_output_rms'] == diag['core_output_rms']


@pytest.mark.parametrize('variant', ['separated', 'coincident', 'buffered_coincident'])
def test_zero_updates_equivalence_a_b_c_including_gradients(variant):
    """Every named layout reduces exactly to the ordinary stack at zero updates."""
    torch.manual_seed(41)
    model = make_model(variant)
    base = GPT(GPTConfig(n_layer=8, n_head=2, n_embd=8, block_size=6, vocab_size=32))
    model.transformer.load_state_dict(base.transformer.state_dict())
    x = torch.randint(32, (2, 6))
    schedule = RecurrenceSchedule((), ())

    expected_logits, expected_loss = base(x, x)
    actual_logits, actual_loss = model(x, x, schedule=schedule)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)

    expected_loss.backward()
    actual_loss.backward()
    actual_parameters = dict(model.named_parameters())
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(actual_parameters[name].grad, parameter.grad, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.temporal_mixer.parameters())
    assert all(parameter.grad is None for parameter in model.depth_mixer.parameters())


@pytest.mark.parametrize('variant', ['separated', 'coincident'])
def test_exact_resume_for_both_architectures(variant, prepared_data, tmp_path):
    config = configuration(variant)
    config.update(dataset=str(prepared_data), eval_panel_path='', n_embd=8, n_head=2,
                  block_size=6, device='cpu', num_threads=1, batch_size=1,
                  gradient_accumulation_steps=2, max_iters=4, eval_interval=2,
                  eval_iters=1, checkpoint_steps=[0, 2, 4], log_interval=1)
    full = train({**config, 'out_dir': str(tmp_path / 'full')})
    resumed_dir = str(tmp_path / 'resumed')
    train({**config, 'out_dir': resumed_dir, 'max_iters': 2})
    resumed = train({**config, 'out_dir': resumed_dir, 'init_from': 'resume'})
    a, b = [torch.load(path, weights_only=False) for path in (full, resumed)]
    torch.testing.assert_close(a['model'], b['model'], rtol=0, atol=0)
    torch.testing.assert_close(a['optimizer'], b['optimizer'], rtol=0, atol=0)
    assert a['recurrence_sampler'] == b['recurrence_sampler']
    # A shared-source checkpoint must not be silently resumed as a separate source.
    with pytest.raises(ValueError, match='model configuration'):
        train({**config, 'out_dir': resumed_dir, 'init_from': 'resume',
               'n_source': 1 - config['n_source'], 'n_core': 5})


def test_replayed_training_updates_and_matched_configs(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_text('{"event":"train","step":1,"seconds":2}\n'
                    '{"event":"train","step":2,"seconds":4}\n'
                    '{broken\n{"event":"train","step":2,"seconds":3}\n')
    rows = training_rows(path)
    assert rows[2]['seconds'] == 3
    a, b = [configuration(v) for v in ('separated', 'coincident')]
    assert {key for key in a if a[key] != b[key]} == {'n_buffer', 'n_core', 'n_source', 'out_dir'}


@pytest.mark.parametrize('b_nll,selected', [(0.49, 'coincident'), (0.499, 'separated')])
def test_runner_scores_late_checkpoints_and_checks_report_identity(tmp_path, monkeypatch, b_nll, selected):
    import json
    from data_loader import file_hash
    import experiments.ablations.architecture_sites.run as runner
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    receipt = dict(panel_sha256='panel', dataset_manifest_hash='dataset')
    monkeypatch.setattr(runner, 'recorded_protocol', lambda: receipt)
    for variant in runner.VARIANTS:
        directory = tmp_path / variant
        directory.mkdir()
        with (directory / 'metrics.jsonl').open('w') as stream:
            for step in range(1, 10001):
                stream.write(json.dumps(dict(event='train', step=step, seconds=.1)) + '\n')
        for step in runner.STEPS:
            checkpoint = directory / f'ckpt-step{step:06d}.pt'
            checkpoint.write_bytes(f'{variant}-{step}'.encode())
            report = dict(checkpoint_sha256=file_hash(checkpoint), checkpoint_step=step,
                          panel_split='selection', panel_file_sha256='panel', manifest_hash='dataset',
                          mask_seeds=[11, 23, 37], device='cuda', batch_fingerprint='fixed-batch',
                          cells=[dict(u_t=t, u_d=d, nll_mean=.5 if variant == 'separated' else b_nll,
                                      accuracy_mean=.8, training_probability=1/9,
                                      estimated_forward_matmul_flops_per_sequence_mean=100)
                                 for t in (0, 1, 3) for d in (0, 1, 3)])
            runner.report_path(variant, step).write_text(json.dumps(report))
    runner.summarize()
    result = json.loads((tmp_path / 'analysis/decision.json').read_text())
    assert result['selected'] == selected
    assert len((tmp_path / 'analysis/curves.csv').read_text().splitlines()) == 91
    runner.benchmark()
    measured = json.loads((tmp_path / 'analysis/benchmark.json').read_text())
    assert measured['separated']['mean_seconds_per_update'] == pytest.approx(.1)
    runner.evaluate('separated', 1000, 'selection')  # Reuses matching reports without launching a process.
    path = runner.report_path('separated', 1000)
    report = json.loads(path.read_text())
    report['checkpoint_sha256'] = 'wrong'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='does not match'):
        runner.evaluate('separated', 1000, 'selection')


def test_new_default_and_legacy_checkpoint_layouts_are_distinct():
    current = RecurrentGPTConfig()
    assert (current.n_prelude, current.n_buffer, current.n_core, current.n_source, current.n_coda) == (1, 1, 4, 1, 1)
    assert current.temporal_source_output_index == current.source_index == current.coda_start - 1
    original = RecurrentGPTConfig.from_checkpoint(dict(n_layer=8, n_prelude=2, n_core=4, n_coda=1))
    assert original.n_buffer == 0
    assert original.temporal_source_output_index == original.source_index == 6
    a = RecurrentGPTConfig.from_checkpoint(dict(n_layer=8, n_prelude=1, n_buffer=1, n_core=4, n_coda=1))
    b = RecurrentGPTConfig.from_checkpoint(dict(n_layer=8, n_prelude=1, n_core=6, n_source=0, n_coda=1))
    assert a == current
    assert b.n_buffer == 0
    assert b.core_end == b.temporal_source_output_index == b.source_index == 6


@pytest.mark.parametrize('counts', [(1, 2, 2, 2, 1), (0, 1, 3, 0, 0), (2, 0, 2, 3, 1)])
def test_flexible_ordered_boundaries_preserve_backbone_and_call_counts(counts):
    prelude, buffer, core, source, coda = counts
    config = RecurrentGPTConfig(n_layer=sum(counts), n_prelude=prelude, n_buffer=buffer,
                                n_core=core, n_source=source, n_coda=coda,
                                n_head=2, n_embd=8, block_size=6, vocab_size=32)
    model = Recurrent2DGPT(config)
    base = GPT(GPTConfig(n_layer=sum(counts), n_head=2, n_embd=8, block_size=6, vocab_size=32))
    base.transformer.load_state_dict(model.transformer.state_dict())
    x = torch.randint(32, (1, 6))
    expected, _ = base(x, x)
    actual, _ = model(x, x, schedule=RecurrenceSchedule((), ()))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    calls = Counter()
    handles = [layer.register_forward_hook(lambda m, a, o, i=i: calls.update([i]))
               for i, layer in enumerate(model.transformer.h)]
    schedule = RecurrenceSchedule((True, False, False), (True, True, True))
    _, loss = model(x, x, schedule=schedule)
    for handle in handles:
        handle.remove()
    expected_calls = [1]*prelude + [4]*(buffer+core) + [2]*source + [1]*coda
    assert [calls[i] for i in range(sum(counts))] == expected_calls
    assert compute_estimate(config, schedule, 6)['block_applications'] == sum(expected_calls)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
