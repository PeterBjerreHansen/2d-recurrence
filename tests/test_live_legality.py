import math

import numpy as np
import pytest
import torch

from evaluation.live_legality import evaluate_row
from evaluation.pgn_annotations import annotate_row
from inference.live import create_live_state, decode_live_step, validate_live_inference_spec
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig

ITOS = dict(enumerate(' #+-.0123456789;=BKNOQRabcdefghx'))
STOI = {char: index for index, char in ITOS.items()}
TEXT = ';1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 '


def _model(mode):
    torch.manual_seed(307)
    if mode == 'baseline':
        return GPT(GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=40)).eval()
    return Recurrent2DGPT(RecurrentGPTConfig(
        n_layer=5, n_prelude=1, n_buffer=1, n_core=1, n_source=1, n_coda=1,
        n_head=2, n_embd=16, block_size=40, recurrence_mode=mode)).eval()


def _brute_force_logprob(model, spec, prefix, san):
    state = create_live_state(model, spec.depth_steps, spec.kv_strategy)
    text = prefix + san
    total, logprobs = 0.0, None
    for index, char in enumerate(text):
        if logprobs is not None and index >= len(prefix):
            total += float(logprobs[STOI[char]])
        logprobs = decode_live_step(model, torch.tensor([STOI[char]]), state)[0].log_softmax(-1)
    return total + float(torch.logsumexp(logprobs[[STOI[' '], STOI[';']]], dim=0))


@pytest.mark.parametrize('mode, depth_steps', [('baseline', None), ('temporal', 1), ('hybrid', 2)])
def test_trie_scores_match_brute_force_live_decoding(mode, depth_steps):
    model = _model(mode)
    spec = validate_live_inference_spec(model.config, depth_steps, None)
    values = np.array([STOI[char] for char in TEXT])
    losses, correct, records, skipped = evaluate_row(
        model, values[:-1], values[1:], TEXT, spec, STOI, ITOS)
    moves = [move for move in annotate_row(TEXT).moves if move.complete]
    assert len(records) == len(moves) == 6 and skipped == dict(context_limit=0, replay=0, noncanonical=0)
    for move, record in zip(moves, records):
        scores = {san: _brute_force_logprob(model, spec, TEXT[:move.start], san) for san in move.legal_sans}
        assert record['move_start'] == move.start and record['n_legal'] == len(move.legal_sans)
        assert record['legal_mass'] == pytest.approx(sum(math.exp(v) for v in scores.values()), rel=1e-4)
        assert record['actual_logprob'] == pytest.approx(scores[move.san], abs=1e-4)
        assert record['top_legal_match'] == (max(scores, key=scores.get) == move.san)
    assert losses.shape == (len(TEXT) - 1,) and 0 <= correct.mean() <= 1


def test_moves_beyond_the_context_limit_are_skipped():
    model = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=len(TEXT) - 1)).eval()
    spec = validate_live_inference_spec(model.config, None, None)
    values = np.array([STOI[char] for char in TEXT])
    _, _, records, skipped = evaluate_row(model, values[:-1], values[1:], TEXT, spec, STOI, ITOS)
    assert skipped['context_limit'] >= 1 and len(records) + skipped['context_limit'] == 6


@pytest.mark.parametrize('text, reason', [
    (';1.e4 e5 2.Ke3 Nc6 ', 'noncanonical'),                 # illegal move: the board before it is known
    (';1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6 4.Qxf7 ', 'noncanonical'),  # mate written without its #
])
def test_unscorable_moves_are_counted_not_fatal(text, reason):
    model = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=64)).eval()
    spec = validate_live_inference_spec(model.config, None, None)
    values = np.array([STOI[char] for char in text])
    _, _, records, skipped = evaluate_row(model, values[:-1], values[1:], text, spec, STOI, ITOS)
    assert skipped[reason] == 1
    assert all(record['n_legal'] > 0 for record in records)


def test_non_finite_scores_are_rejected():
    model = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=64)).eval()
    with torch.no_grad():
        model.lm_head.weight.fill_(float('nan'))
    spec = validate_live_inference_spec(model.config, None, None)
    values = np.array([STOI[char] for char in TEXT])
    with pytest.raises(FloatingPointError):
        evaluate_row(model, values[:-1], values[1:], TEXT, spec, STOI, ITOS)
