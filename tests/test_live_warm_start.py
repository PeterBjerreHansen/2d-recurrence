from importlib import import_module

import numpy as np
import pytest
import torch

from evaluation.live_inference import evaluate_teacher_forced
from inference.live import create_live_state, decode_live_step, validate_live_inference_spec
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig

align = import_module('experiments.ablations.live_warm_start.align')


def _model(mode):
    torch.manual_seed(11)
    return Recurrent2DGPT(RecurrentGPTConfig(
        n_layer=5, n_prelude=1, n_buffer=1, n_core=1, n_source=1, n_coda=1,
        n_head=2, n_embd=16, block_size=24, recurrence_mode=mode)).eval()


def _live_memory(model, x, depth_steps, strategy):
    spec = validate_live_inference_spec(model.config, depth_steps, strategy)
    state = create_live_state(model, spec.depth_steps, spec.kv_strategy)
    memories = []
    for token in x[0]:
        decode_live_step(model, token[None], state, spec)
        memories.append(state.temporal_memory[:, 0])
    return torch.stack(memories, 1)


@pytest.mark.parametrize('mode, depth_steps, strategy', [
    ('temporal', 1, 'final_depth'), ('hybrid', 1, 'depth_specialized'), ('hybrid', 3, 'depth_specialized')])
def test_settled_memory_and_final_pass_equal_live_execution(mode, depth_steps, strategy):
    model = _model(mode)
    x = torch.randint(0, 32, (1, 12))
    y = torch.randint(0, 32, (1, 12))
    with torch.no_grad():
        p = align.prelude(model, x)
        memory = align.settle(model, p, x.shape[1], depth_steps)  # Jacobi is exact after T passes
        torch.testing.assert_close(memory, _live_memory(model, x, depth_steps, strategy), atol=1e-5, rtol=1e-5)
        losses = align.final_losses(model, p, memory, y, depth_steps)
    live = evaluate_teacher_forced(model, [(x[0], y[0])], depth_steps=depth_steps, kv_strategy=strategy)
    assert float(losses.mean()) == pytest.approx(live['nll'], abs=1e-5)


def test_one_pass_matches_the_model_training_graph():
    model = _model('temporal')
    x = torch.randint(0, 32, (2, 10))
    y = torch.randint(0, 32, (2, 10))
    schedule = import_module('recurrence.schedule').RecurrenceSchedule((True,) * 3, (False,) * 3)
    with torch.no_grad():
        _, expected = model(x, y, schedule=schedule)
        p = align.prelude(model, x)
        losses = align.final_losses(model, p, align.settle(model, p, 3), y)
    assert float(losses.mean()) == pytest.approx(float(expected), abs=1e-5)


@pytest.mark.parametrize('mode', ['temporal', 'hybrid'])
def test_align_changes_only_the_temporal_mixer(mode, monkeypatch):
    monkeypatch.setitem(align.PROCEDURE, 'updates', 2)
    monkeypatch.setitem(align.PROCEDURE, 'passes', 3)
    model = _model(mode)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rows = np.random.default_rng(0).integers(0, 32, (40, 25)).astype(np.uint8)
    model.train()
    history = align.align(model, rows, mode, 'cpu', log=lambda _: None)
    assert len(history) == 2
    changed = {name for name, value in model.state_dict().items() if not torch.equal(value, before[name])}
    assert changed and all(name.startswith('temporal_mixer.') for name in changed)


def test_aligned_export_is_evaluation_only():
    model = _model('temporal')
    source = dict(model=model.state_dict(), config={}, optimizer={'state': 1}, scaler={}, recurrence_sampler={},
                  rng_by_rank=[], training_seconds=1.0, last_eval_step=0, best_val_loss=1.0, iter_num=5)
    exported = align.evaluation_checkpoint(source, model.state_dict(), dict(procedure='x'))
    assert exported['evaluation_only'] and exported['alignment'] == dict(procedure='x')
    assert not set(align.RESUME_KEYS) & set(exported)
    assert exported['iter_num'] == 5 and 'optimizer' in source  # the source is not modified
    assert align.weights_sha256(exported['model']) == align.weights_sha256(model.state_dict())


def test_study_validation_rejects_aligned_checkpoints():
    runner = import_module('experiments.long_runs.20B_recurrence.run')
    with pytest.raises(ValueError, match='aligned'):
        runner._validate_checkpoint({'evaluation_only': True, 'alignment': {}}, 'temporal')


def test_model_settles_temporal_memory_to_live_execution():
    model = _model('temporal')
    x = torch.randint(0, 32, (1, 12))
    memory, passes = model.settle_temporal_memory(x, max_passes=12, tolerance=0.0)
    assert passes == 12
    torch.testing.assert_close(memory, _live_memory(model, x, 1, 'final_depth'), atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        _, early = model.settle_temporal_memory(x, max_passes=40, tolerance=1e9)
    assert early == 2  # stops at the first comparison when any change is allowed


def test_warm_started_forward_reads_the_given_memory():
    model = _model('temporal')
    x = torch.randint(0, 32, (2, 10))
    y = torch.randint(0, 32, (2, 10))
    schedule = import_module('recurrence.schedule').RecurrenceSchedule((), ())
    with torch.no_grad():
        memory, _ = model.settle_temporal_memory(x, max_passes=10, tolerance=0.0)
        _, loss = model(x, y, schedule=schedule, initial_temporal_state=memory)
        expected = align.final_losses(model, align.prelude(model, x), memory, y).mean()
    assert float(loss) == pytest.approx(float(expected), abs=1e-6)


def test_training_resumes_with_warm_start_batches(prepared_data, tmp_path):
    from train import train
    base = import_module('tests.test_recurrence_modes').mode_training_config(
        'temporal', tmp_path / 'run', dataset=str(prepared_data), gradient_accumulation_steps=2,
        max_iters=4, eval_interval=2, log_interval=1)
    train({**base, 'max_iters': 2})
    final = train({**base, 'init_from': 'resume', 'warm_start_fraction': 0.5, 'warm_start_max_passes': 4})
    assert torch.load(final, weights_only=False)['iter_num'] == 4
    import json
    records = [json.loads(line) for line in (tmp_path / 'run' / 'metrics.jsonl').read_text().splitlines()]
    warm = [[s.get('warm_passes') for s in r['schedules']] for r in records if r['event'] == 'train']
    assert all(passes is None for passes in warm[0])           # before the resume: no warm start
    assert all(p[0] is not None and 1 <= p[0] <= 4 and p[1] is None for p in warm[-2:])


def test_warm_start_is_refused_outside_the_temporal_mode(tmp_path):
    from train import train
    config = import_module('tests.test_recurrence_modes').mode_training_config(
        'hybrid', tmp_path / 'run', warm_start_fraction=0.25)
    with pytest.raises(ValueError, match='temporal mode'):
        train(config)
