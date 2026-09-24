import copy
import json
import random
from collections import Counter

import pytest
import torch

from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig, TemporalMixer
from recurrence.schedule import RecurrenceSchedule, RecurrenceScheduleSampler, sample_schedule
from train import train

UPDATE_SUPPORT = [0, 1, 3]
UPDATE_PROBABILITIES = [[.10, .12, .04], [.12, .26, .08], [.04, .08, .16]]


def tiny_model(**overrides):
    return Recurrent2DGPT(RecurrentGPTConfig(n_layer=4, n_prelude=1, n_buffer=0, n_core=1,
                                            n_coda=1, n_embd=16, n_head=2,
                                            block_size=12, **overrides))


def history_reference(model, x, y, schedule):
    """Select the latest written pass from histories, independently of the state loop."""
    p = model.embed(x)
    start = model.config.n_prelude
    source = start + model.config.n_core
    for block in model.transformer.h[:start]:
        p = block(p)
    depths, memories = [], []
    for b in range(schedule.rounds):
        temporal_writes = [i for i in range(b) if schedule.temporal_write_mask[i]]
        depth_writes = [i for i in range(b) if schedule.depth_write_mask[i]]
        a = p
        if temporal_writes:
            memory = memories[temporal_writes[-1]]
            shifted = torch.nn.functional.pad(memory[:, :-1], (0, 0, 1, 0))
            a = model.temporal_mixer(p, shifted)
        h = model.depth_mixer(depths[depth_writes[-1]], a) if depth_writes else a
        for block in model.transformer.h[start:source]:
            h = block(h)
        depths.append(h)
        memories.append(model.transformer.h[source](h) if b == schedule.rounds - 1 or schedule.temporal_write_mask[b] else None)
    h = memories[-1]
    for block in model.transformer.h[source + 1:]:
        h = block(h)
    return model.readout(h, y)


def test_zero_updates_exactly_matches_baseline_and_gradients():
    torch.manual_seed(23)
    base = GPT(GPTConfig(n_layer=8, n_head=2, n_embd=16, block_size=12))
    model = Recurrent2DGPT(RecurrentGPTConfig(n_layer=8, n_head=2, n_embd=16, block_size=12))
    model.transformer.load_state_dict(base.transformer.state_dict())
    x = torch.randint(32, (2, 12))
    logits, loss = base(x, x)
    actual, actual_loss = model(x, x, schedule=sample_schedule(0, 0, random.Random(0)))
    torch.testing.assert_close(actual, logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, loss, rtol=0, atol=0)
    loss.backward()
    actual_loss.backward()
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(dict(model.named_parameters())[name].grad, parameter.grad, rtol=0, atol=0)
    assert all(p.grad is None for p in model.temporal_mixer.parameters())
    assert all(p.grad is None for p in model.depth_mixer.parameters())
    assert model.lm_head.weight is model.transformer.wte.weight


def test_deep_supervision_is_normalized_and_has_no_one_pass_term():
    torch.manual_seed(31)
    model = tiny_model()
    reference = copy.deepcopy(model)
    x = torch.randint(32, (2, 12))
    two_pass = RecurrenceSchedule((True,), (True,))

    plain_logits, plain_loss = reference(x, x, schedule=two_pass)
    logits, objective, components = model(
        x, x, schedule=two_pass, deep_supervision=True,
        deep_supervision_lambda=.25, return_components=True)
    torch.testing.assert_close(logits, plain_logits, rtol=0, atol=0)
    torch.testing.assert_close(components['final_loss'], plain_loss, rtol=0, atol=0)
    assert components['intermediate_passes'] == 1
    expected = (plain_loss + .25 * components['intermediate_loss']) / 1.25
    torch.testing.assert_close(objective, expected, rtol=0, atol=0)

    one_reference_logits, one_reference_loss = reference(
        x, x, schedule=RecurrenceSchedule((), ()))
    one_pass_logits, one_pass_loss, one_pass_components = model(
        x, x, schedule=RecurrenceSchedule((), ()), deep_supervision=True,
        deep_supervision_lambda=.25, return_components=True)
    assert one_pass_components['intermediate_loss'] is None
    assert one_pass_components['intermediate_passes'] == 0
    torch.testing.assert_close(one_pass_logits, one_reference_logits, rtol=0, atol=0)
    torch.testing.assert_close(one_pass_loss, one_pass_components['final_loss'], rtol=0, atol=0)
    torch.testing.assert_close(one_pass_loss, one_reference_loss, rtol=0, atol=0)


def test_zero_auxiliary_weight_skips_auxiliary_forwards_and_rng_consumption():
    torch.manual_seed(37)
    model = tiny_model(dropout=0.2).train()
    reference = copy.deepcopy(model).train()
    x = torch.randint(32, (2, 12))
    schedule = RecurrenceSchedule((False,), (True,))
    calls = Counter()
    handle = model.transformer.h[model.config.temporal_source_output_index].register_forward_hook(
        lambda module, args, output: calls.update(['temporal_source']))
    rng_state = torch.get_rng_state()
    try:
        torch.set_rng_state(rng_state)
        expected_logits, expected_loss = reference(x, x, schedule=schedule)
        expected_rng_state = torch.get_rng_state()
        torch.set_rng_state(rng_state)
        actual_logits, actual_loss, components = model(
            x, x, schedule=schedule, deep_supervision=True,
            deep_supervision_lambda=0.0, return_components=True)
        actual_rng_state = torch.get_rng_state()
    finally:
        handle.remove()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert torch.equal(actual_rng_state, expected_rng_state)
    assert calls['temporal_source'] == 1
    assert components['intermediate_loss'] is None
    assert components['intermediate_passes'] == 0


@pytest.mark.parametrize('weight', [True, False])
def test_boolean_deep_supervision_weight_is_rejected(weight):
    model = tiny_model()
    x = torch.randint(32, (2, 12))
    with pytest.raises(ValueError, match='deep_supervision_lambda'):
        model(x, x, schedule=RecurrenceSchedule((True,), (True,)),
              deep_supervision=True, deep_supervision_lambda=weight)


@pytest.mark.parametrize('masks', [((True, True, True), (True, False, False)),
                                  ((True, False, False), (True, True, True)),
                                  ((False, False, True), (True, True, True)),
                                  ((True, True, True), (False, False, False))])
def test_held_states_match_history_reference_including_gradients(masks):
    torch.manual_seed(10)
    model = tiny_model()
    reference = copy.deepcopy(model)
    schedule = RecurrenceSchedule(*masks)
    x = torch.randint(32, (2, 12))
    actual, loss = model(x, x, schedule=schedule)
    expected, expected_loss = history_reference(reference, x, x, schedule)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    loss.backward()
    expected_loss.backward()
    for name, parameter in model.named_parameters():
        expected_grad = dict(reference.named_parameters())[name].grad
        if expected_grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=0, atol=0)
    assert model.transformer.h[2].attn.c_attn.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('u_t,u_d', [(t, d) for t in UPDATE_SUPPORT for d in UPDATE_SUPPORT] + [(7, 2), (2, 7)])
def test_calls_causality_and_finite_gradients(u_t, u_d):
    torch.manual_seed(4)
    model = tiny_model().eval()
    counts = Counter()
    handles = [block.register_forward_hook(lambda module, args, output, i=i: counts.update([i]))
               for i, block in enumerate(model.transformer.h)]
    x = torch.randint(32, (1, 12))
    schedule = sample_schedule(u_t, u_d, random.Random(1))
    logits, loss = model(x, x, schedule=schedule)
    assert counts == {0: 1, 1: max(u_t, u_d) + 1, 2: u_t + 1, 3: 1}
    for handle in handles:
        handle.remove()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    changed = x.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 32
    other, _ = model(changed, changed, schedule=schedule)
    torch.testing.assert_close(logits[:, :6], other[:, :6], rtol=0, atol=0)
    # Each call starts with absent states, with no persistent memory from prior batches.
    again, _ = model(x, x, schedule=schedule)
    torch.testing.assert_close(logits, again, rtol=0, atol=0)


def test_temporal_controller_uses_unprojected_sources_and_bypasses_boundary():
    torch.manual_seed(1)
    mixer = TemporalMixer(4)
    with torch.no_grad():
        mixer.gates.weight.normal_()
        mixer.memory_value.weight.normal_()
        mixer.prelude_value.weight.normal_()
    p = torch.randn(2, 5, 4)
    memory = torch.zeros_like(p)  # Real zero-valued memories still use mixing.
    memory[:, 2:] = torch.randn(2, 3, 4)
    nr, np = mixer.memory_norm(memory), mixer.prelude_norm(p)
    alpha, beta = torch.sigmoid(mixer.gates(torch.cat([nr, np], -1))).chunk(2, -1)
    expected = alpha * mixer.memory_value(nr) + beta * mixer.prelude_value(np)
    actual = mixer(p, memory)
    torch.testing.assert_close(actual[:, 0], p[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 1:], expected[:, 1:], rtol=0, atol=0)
    assert not torch.equal(actual[:, 1], p[:, 1])


def test_temporal_gate_initialization_is_configurable_and_complementary():
    mixer = TemporalMixer(4, memory_gate_init=.25)
    assert torch.allclose(mixer.gates.weight, torch.zeros_like(mixer.gates.weight))
    alpha, beta = torch.sigmoid(mixer.gates.bias).chunk(2)
    torch.testing.assert_close(alpha, torch.full_like(alpha, .25), rtol=0, atol=1e-6)
    torch.testing.assert_close(beta, torch.full_like(beta, .75), rtol=0, atol=1e-6)
    assert tiny_model(temporal_memory_gate_init=.25).config.temporal_memory_gate_init == .25


@pytest.mark.parametrize('value', [True, 0, 1, -0.1, 1.1, float('nan')])
def test_temporal_gate_initialization_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='temporal_memory_gate_init|memory_gate_init'):
        tiny_model(temporal_memory_gate_init=value)


def test_sampler_distribution_masks_and_resume():
    sampler = RecurrenceScheduleSampler(UPDATE_SUPPORT, UPDATE_PROBABILITIES, 19)
    placements = Counter()
    for _ in range(12000):
        s = sampler.sample()
        assert s.rounds == max(s.u_t, s.u_d) + 1
        assert (s.u_t, s.u_d) in [(t, d) for t in UPDATE_SUPPORT for d in UPDATE_SUPPORT]
        if (s.u_t, s.u_d) == (1, 3):
            placements[s.temporal_write_mask.index(True)] += 1
    for i, t in enumerate(UPDATE_SUPPORT):
        for j, d in enumerate(UPDATE_SUPPORT):
            assert abs(sampler.pair_histogram[t, d] / 12000 - UPDATE_PROBABILITIES[i][j]) < .015
    assert all(abs(n / sum(placements.values()) - 1 / 3) < .05 for n in placements.values())
    resumed = RecurrenceScheduleSampler(UPDATE_SUPPORT, UPDATE_PROBABILITIES, 999)
    resumed.load_state_dict(sampler.state_dict())
    random.seed(983)
    assert [sampler.sample() for _ in range(100)] == [resumed.sample() for _ in range(100)]
    assert sampler.state_dict() == resumed.state_dict()


@pytest.mark.parametrize('masks', [((False,), (False,)), ((True,), ()), ([True], (True,)), ((1,), (True,))])
def test_invalid_masks(masks):
    with pytest.raises(ValueError):
        RecurrenceSchedule(*masks)


@pytest.mark.parametrize('counts', [(-1, 0), (True, 0), (1.5, 3)])
def test_invalid_counts(counts):
    with pytest.raises(ValueError):
        sample_schedule(*counts, random.Random())


def test_recurrent_training_exact_resume(prepared_data, tmp_path):
    config = dict(architecture='recurrent', update_support=UPDATE_SUPPORT,
                  update_probabilities=UPDATE_PROBABILITIES, recurrence_seed=19,
                  n_layer=4, n_prelude=1, n_buffer=0, n_core=1, n_coda=1, n_head=2, n_embd=16,
                  dataset=str(prepared_data), block_size=12, batch_size=2,
                  gradient_accumulation_steps=3, max_iters=4, eval_interval=2,
                  eval_iters=2, eval_u_t=3, eval_u_d=1, log_interval=2,
                  warmup_iters=0, lr_decay_iters=4, compile=False, device='cpu',
                  dtype='float32', dropout=.2, num_threads=2)
    full = train({**config, 'out_dir': str(tmp_path / 'full')})
    resumed_dir = str(tmp_path / 'resumed')
    train({**config, 'max_iters': 2, 'out_dir': resumed_dir})
    resumed = train({**config, 'init_from': 'resume', 'out_dir': resumed_dir})
    a, b = [torch.load(path, weights_only=False) for path in (full, resumed)]
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    torch.testing.assert_close(a['optimizer'], b['optimizer'], rtol=0, atol=0)
    assert a['recurrence_sampler'] == b['recurrence_sampler']
    assert a['recurrence_sampler']['draw_count'] == 12
    assert a['best_val_loss'] == b['best_val_loss']


def test_recurrent_training_exact_resume_across_probability_crossover(prepared_data, tmp_path):
    phase_zero = [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    phase_one = [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]
    config = dict(architecture='recurrent', update_support=UPDATE_SUPPORT,
                  update_probabilities=[], update_probability_schedule={
                      'type': 'piecewise_constant',
                      'phases': [{'start_step': 0, 'update_probabilities': phase_zero},
                                 {'start_step': 2, 'update_probabilities': phase_one}]},
                  recurrence_seed=19, n_layer=4, n_prelude=1, n_buffer=0, n_core=1,
                  n_coda=1, n_head=2, n_embd=16, dataset=str(prepared_data), block_size=12,
                  batch_size=2, gradient_accumulation_steps=1, max_iters=4, eval_interval=2,
                  eval_iters=2, eval_u_t=3, eval_u_d=1, log_interval=2,
                  warmup_iters=0, lr_decay_iters=4, compile=False, device='cpu',
                  dtype='float32', dropout=.2, num_threads=2)
    full = train({**config, 'out_dir': str(tmp_path / 'full')})
    resumed_dir = str(tmp_path / 'resumed')
    train({**config, 'max_iters': 2, 'out_dir': resumed_dir})
    resumed = train({**config, 'init_from': 'resume', 'out_dir': resumed_dir})
    a, b = [torch.load(path, weights_only=False) for path in (full, resumed)]
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    torch.testing.assert_close(a['optimizer'], b['optimizer'], rtol=0, atol=0)
    assert a['recurrence_sampler'] == b['recurrence_sampler']
    assert a['recurrence_sampler']['pair_histogram'] == {(0, 0): 2, (3, 3): 2}
    assert a['recurrence_sampler']['draw_count'] == 4
    assert a['best_val_loss'] == b['best_val_loss']
    events = [json.loads(line) for line in (tmp_path / 'full' / 'metrics.jsonl').read_text().splitlines()]
    evaluations = [record for record in events if record['event'] == 'evaluation']
    by_step = {record['step']: record for record in evaluations}
    assert by_step[0]['next_update_probability_step'] == 0
    assert by_step[0]['next_update_probability_matrix'] == phase_zero
    assert by_step[0]['last_update_probability_matrix'] is None
    assert by_step[2]['next_update_probability_step'] == 2
    assert by_step[2]['next_update_probability_matrix'] == phase_one
    assert by_step[2]['last_update_probability_matrix'] == phase_zero


def test_noncommuting_mixers_read_held_depth_and_raw_shifted_source():
    class Affine(torch.nn.Module):
        def __init__(self, scale, offset):
            super().__init__()
            self.scale, self.offset = scale, offset

        def forward(self, x):
            return self.scale * x + self.offset

    class Temporal(torch.nn.Module):
        def forward(self, p, memory):
            return p + 5 * memory

    class Depth(torch.nn.Module):
        def forward(self, state, anchor):
            return 7 * state + 11 * anchor

    model = tiny_model()
    model.embed = lambda x: x.float().unsqueeze(-1)
    model.readout = lambda h, targets: (h, None)
    model.transformer.h = torch.nn.ModuleList([torch.nn.Identity(), Affine(2, 1),
                                              Affine(3, 2), torch.nn.Identity()])
    model.temporal_mixer = Temporal()
    model.depth_mixer = Depth()
    # First core = [3,5,7], source = [11,17,23]. Depth remains that first core.
    # Second anchor = [1,57,88], core = [65,1325,2035], source = [197,3977,6107].
    # Third anchor = [1,987,19888], core = [65,21785,437635].
    output, _ = model(torch.tensor([[1, 2, 3]]),
                      schedule=RecurrenceSchedule((True, True), (True, False)))
    torch.testing.assert_close(output.flatten(), torch.tensor([197., 65357., 1312907.]), rtol=0, atol=0)
    # Hold temporal memory instead: each later read uses [0,11,17], never a second shift.
    # Second depth = [65,1325,2035]; third core = [933,19805,30427].
    output, _ = model(torch.tensor([[1, 2, 3]]),
                      schedule=RecurrenceSchedule((True, False), (True, True)))
    torch.testing.assert_close(output.flatten(), torch.tensor([2801., 59417., 91283.]), rtol=0, atol=0)


@pytest.mark.parametrize('u_t,u_d', [(0, 3), (3, 0)])
def test_single_axis_matches_reference(u_t, u_d):
    model = tiny_model()
    x = torch.randint(32, (1, 12))
    schedule = sample_schedule(u_t, u_d, random.Random(0))
    actual, loss = model(x, x, schedule=schedule)
    expected, _ = history_reference(model, x, x, schedule)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    loss.backward()
    unused = model.temporal_mixer if u_t == 0 else model.depth_mixer
    used = model.depth_mixer if u_t == 0 else model.temporal_mixer
    assert all(p.grad is None for p in unused.parameters())
    assert all(p.grad is not None for p in used.parameters())
