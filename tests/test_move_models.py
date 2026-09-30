"""Move-token models: the untied readout, the three execution checks, and a final-pass-only loss."""

from importlib import import_module

import pytest
import torch

from inference.live import create_live_state, decode_live_step
from inference.reference import create_reference_state, decode_reference_step
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from moves.objectives import LegalTargets
from moves.vocab import MOVE_COUNT, VOCAB_SIZE
from recurrence.schedule import RecurrenceSchedule

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


def test_the_loss_reads_only_the_final_pass(monkeypatch):
    model = _model('hybrid').train()
    targets = LegalTargets(positions=torch.tensor([0, 3, 8]), owner=torch.tensor([0, 0, 1, 2, 2]),
                           moves=torch.tensor([5, 9, 700, 12, 1300]))
    calls = []
    original = type(model).readout

    def counted(self, h, targets=None):
        calls.append(targets is not None)
        return original(self, h, targets)

    monkeypatch.setattr(type(model), 'readout', counted)
    schedule = RecurrenceSchedule((True, True, True), (False, True, False))
    _, loss = model(_tokens(10), targets, schedule=schedule)
    assert calls == [True] and torch.isfinite(loss)
