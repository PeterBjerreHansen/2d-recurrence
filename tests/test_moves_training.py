"""Move-token models and training: untied readout, execution checks, losses and the trainer."""

from importlib import import_module
import json

import pytest
import torch

from inference.live import create_live_state, decode_live_step
from inference.reference import create_reference_state, decode_reference_step
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from moves.data import MoveData
from moves.vocab import MOVE_COUNT, VOCAB_SIZE
from recurrence.schedule import RecurrenceSchedule
from train import train

align = import_module('experiments.ablations.live_warm_start.align')

MOVE_ARGS = dict(vocab_size=VOCAB_SIZE, output_size=MOVE_COUNT, tie_weights=False)


def _model(mode, block_size=12):
    torch.manual_seed(31)
    return Recurrent2DGPT(RecurrentGPTConfig(
        n_layer=5, n_prelude=1, n_buffer=1, n_core=1, n_source=1, n_coda=1, n_head=2, n_embd=16,
        block_size=block_size, recurrence_mode=mode, **MOVE_ARGS)).eval()


def _tokens(length, batch=1):
    return torch.randint(VOCAB_SIZE, (batch, length))


def test_move_models_have_an_untied_readout_over_moves():
    model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=8, block_size=12, **MOVE_ARGS))
    assert model.lm_head.weight.shape == (MOVE_COUNT, 8)
    assert model.transformer.wte.weight.shape == (VOCAB_SIZE, 8)
    assert model.lm_head.weight is not model.transformer.wte.weight
    logits, _ = model(_tokens(5))
    assert logits.shape[-1] == MOVE_COUNT
    with pytest.raises(ValueError, match='Tied'):
        GPT(GPTConfig(n_layer=1, n_head=1, n_embd=4, vocab_size=VOCAB_SIZE, output_size=MOVE_COUNT))


# --- the three execution checks (docs/engine_policy_plan.md, Pitfalls 1) ---

@pytest.mark.parametrize('mode, depth_steps, strategy', [
    ('temporal', 1, 'final_depth'), ('depth', 3, 'depth_specialized'),
    ('hybrid', 1, 'depth_specialized'), ('hybrid', 3, 'depth_specialized'), ('hybrid', 3, 'final_depth')])
def test_cached_live_execution_equals_the_slow_reference(mode, depth_steps, strategy):
    model = _model(mode)
    tokens = _tokens(9)
    cached = create_live_state(model, depth_steps, strategy)
    reference = create_reference_state(model, depth_steps, strategy)
    for position in range(tokens.shape[1]):
        torch.testing.assert_close(decode_live_step(model, tokens[:, position], cached),
                                   decode_reference_step(model, tokens[:, position], reference),
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('mode, depth_steps', [('temporal', 1), ('hybrid', 1), ('hybrid', 3)])
def test_the_settled_training_graph_equals_live_execution(mode, depth_steps):
    model = _model(mode)
    tokens = _tokens(10)
    with torch.no_grad():
        p = align.prelude(model, tokens)
        # Jacobi passes are exact after as many writes as there are positions.
        memory = align.settle(model, p, tokens.shape[1], depth_steps)
        h = align.one_pass(model, p, memory, depth_steps)
        settled, _ = model.readout(model.coda(model.temporal_source(h)), torch.zeros_like(tokens))
    state = create_live_state(model, depth_steps, 'depth_specialized')
    live = torch.stack([decode_live_step(model, tokens[:, t], state) for t in range(tokens.shape[1])], 1)
    torch.testing.assert_close(live, settled, rtol=1e-4, atol=1e-5)


def test_depth_only_live_execution_equals_its_training_cell():
    model = _model('depth')
    tokens = _tokens(8)
    for depth_steps in (1, 2, 4):
        schedule = RecurrenceSchedule((False,) * (depth_steps - 1), (True,) * (depth_steps - 1))
        training, _ = model(tokens, torch.zeros_like(tokens), schedule=schedule)
        state = create_live_state(model, depth_steps, 'depth_specialized')
        live = torch.stack([decode_live_step(model, tokens[:, t], state) for t in range(tokens.shape[1])], 1)
        torch.testing.assert_close(live, training, rtol=1e-4, atol=1e-5)


def test_the_loss_reads_only_the_final_pass(monkeypatch, move_data):
    model = _model('hybrid', block_size=40).train()
    data = MoveData(move_data, 40)
    batch = data.batch('train', 2, 'cpu', torch.Generator().manual_seed(0), objective='legal')
    calls = []
    original = type(model).readout

    def counted(self, h, targets=None):
        calls.append(targets is not None)
        return original(self, h, targets)

    monkeypatch.setattr(type(model), 'readout', counted)
    schedule = RecurrenceSchedule((True, True, True), (False, True, False))
    _, loss = model(batch.x, batch.targets, schedule=schedule)
    assert calls == [True] and torch.isfinite(loss)


# --- training ---

def _config(move_data, output, objective, **overrides):
    config = dict(data_format='moves', objective=objective, dataset=str(move_data), n_layer=2, n_head=2,
                  n_embd=16, block_size=40, batch_size=3, gradient_accumulation_steps=2, max_iters=4,
                  eval_interval=2, eval_iters=1, log_interval=1, warmup_iters=0, lr_decay_iters=4,
                  compile=False, device='cpu', dtype='float32', num_threads=1, out_dir=str(output))
    config.update(overrides)
    return config


def _recurrent(mode, **overrides):
    base = import_module('tests.test_recurrence_modes').mode_training_config(mode, 'unused')
    keys = ('architecture', 'recurrence_mode', 'update_support', 'update_probabilities', 'recurrence_seed',
            'n_layer', 'n_prelude', 'n_buffer', 'n_core', 'n_source', 'n_coda', 'eval_u_t', 'eval_u_d')
    return {**{key: base[key] for key in keys}, **overrides}


def _records(output):
    return [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('objective, architecture', [
    ('human', {}), ('legal', {}), ('legal', _recurrent('temporal')), ('human', _recurrent('hybrid'))])
def test_move_objectives_train_and_count_their_volume(move_data, tmp_path, objective, architecture):
    output = tmp_path / 'run'
    train(_config(move_data, output, objective, random_game_fraction=0.5, **architecture))
    records = _records(output)
    evaluation = [r for r in records if r['event'] == 'evaluation'][-1]
    assert {'train_loss', 'val_loss', 'random_loss'} <= set(evaluation)
    if objective == 'legal':
        assert {'val_exact_set', 'val_precision', 'val_recall', 'val_constant_bce'} <= set(evaluation)
    mode = architecture.get('recurrence_mode')
    depths = [] if mode is None else [1] if mode == 'temporal' else [1, 2, 4]
    for depth in depths:
        assert f'live_J{depth}_val_loss' in evaluation
    assert not any(key.startswith('live_J') for key in evaluation) or depths
    last = [r for r in records if r['event'] == 'train'][-1]
    assert last['row_tokens'] == 4 * 2 * 3 * 40
    assert last['supervised_positions'] == last['human_plies'] + last['random_plies']
    assert last['human_plies'] > 0 and last['random_plies'] > 0
    checkpoint = torch.load(output / 'ckpt.pt', weights_only=False)
    assert checkpoint['move_counts']['row_tokens'] == last['row_tokens']
    assert 'characters_processed' not in last


@pytest.mark.parametrize('objective', ['human', 'legal'])
def test_live_evaluation_is_paired_with_the_training_graph(move_data, tmp_path, objective):
    # A depth-only model with depth_specialized caches runs live exactly as its training cell (0, J-1).
    output = tmp_path / 'run'
    train(_config(move_data, output, objective, max_iters=2, **_recurrent('depth', eval_u_d=1),
                  live_eval_depths=[2]))
    evaluation = [r for r in _records(output) if r['event'] == 'evaluation'][-1]
    assert evaluation['live_J2_val_loss'] == pytest.approx(evaluation['val_loss'], rel=1e-4)
    key = 'accuracy' if objective == 'human' else 'exact_set'
    assert evaluation[f'live_J2_val_{key}'] == pytest.approx(evaluation[f'val_{key}'])


def test_move_training_resumes_exactly_with_any_number_of_loader_workers(move_data, tmp_path):
    config = _config(move_data, tmp_path / 'full', 'legal', random_game_fraction=0.5, **_recurrent('temporal'))
    full = train(config)
    part = {**config, 'out_dir': str(tmp_path / 'part')}
    train({**part, 'max_iters': 2, 'loader_workers': 2})
    resumed = train({**part, 'init_from': 'resume'})
    a, b = (torch.load(path, weights_only=False) for path in (full, resumed))
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['move_counts'] == b['move_counts']


def test_the_engine_stage_continues_a_legal_trunk(move_data, engine_data, tmp_path):
    legal = train(_config(move_data, tmp_path / 'legal', 'legal', max_iters=2, **_recurrent('hybrid')))
    engine = train(_config(engine_data, tmp_path / 'engine', 'engine', max_iters=2, init_from='continue',
                           continue_from=str(legal), **_recurrent('hybrid')))
    result = torch.load(engine, weights_only=False)
    assert result['continued_from']['objective'] == 'legal'
    assert result['continued_from']['iter_num'] == 2
    run = json.loads((tmp_path / 'engine' / 'run.json').read_text())
    assert run['continued_from']['sha256'] == result['continued_from']['sha256']
    evaluation = [r for r in _records(tmp_path / 'engine') if r['event'] == 'evaluation']
    assert 'val_regret' in evaluation[0] and 'random_loss' not in evaluation[0]


def test_a_continued_run_resumes_exactly(move_data, engine_data, tmp_path):
    legal = train(_config(move_data, tmp_path / 'legal', 'legal', max_iters=2, **_recurrent('temporal')))
    config = _config(engine_data, tmp_path / 'full', 'engine', init_from='continue', continue_from=str(legal),
                     **_recurrent('temporal'))
    full = train(config)
    part = {**config, 'out_dir': str(tmp_path / 'part')}
    train({**part, 'max_iters': 2})
    resumed = train({**part, 'init_from': 'resume'})
    a, b = (torch.load(path, weights_only=False) for path in (full, resumed))
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert b['continued_from'] == a['continued_from'] and b['continued_from']['objective'] == 'legal'
    with pytest.raises(ValueError, match='continue_from'):
        train({**part, 'init_from': 'resume', 'continue_from': str(tmp_path / 'other.pt')})


def test_continuing_copies_the_trunk_and_reinitialises_the_readout(move_data, engine_data, tmp_path):
    legal = train(_config(move_data, tmp_path / 'legal', 'legal', max_iters=2))
    engine = train(_config(engine_data, tmp_path / 'engine', 'engine', max_iters=0, init_from='continue',
                           continue_from=str(legal)))
    source, result = (torch.load(path, weights_only=False) for path in (legal, engine))
    for key, value in source['model'].items():
        if key == 'lm_head.weight':
            assert not torch.equal(value, result['model'][key])
        else:
            torch.testing.assert_close(value, result['model'][key], rtol=0, atol=0)


@pytest.mark.parametrize('overrides, message', [
    (dict(objective='engine', random_game_fraction=0.5), 'engine-labelled human games'),
    (dict(objective='cheating'), 'objective must be one of'),
    (dict(init_from='continue'), 'continue_from'),
    (dict(continue_from='elsewhere.pt'), 'continue_from'),
    (dict(data_format='characters'), 'data_format=moves'),
    (dict(live_eval_depths=[1]), 'transformer runs live exactly'),
    (dict(loader_workers=-1), 'loader_workers'),
])
def test_invalid_move_configurations_are_refused(move_data, tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        train({**_config(move_data, tmp_path / 'run', 'legal'), **overrides})
