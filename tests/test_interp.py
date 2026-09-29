from importlib import import_module

import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from interp.boards import encode, probe_points
from interp.probes import fit
from interp.sites import SiteRunner
from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from recurrence.schedule import RecurrenceSchedule

align = import_module('experiments.ablations.live_warm_start.align')


def _recurrent(mode):
    torch.manual_seed(11)
    return Recurrent2DGPT(RecurrentGPTConfig(
        n_layer=6, n_prelude=1, n_buffer=1, n_core=2, n_source=1, n_coda=1,
        n_head=2, n_embd=16, block_size=24, recurrence_mode=mode)).eval()


def _nll(logits, y):
    return float(F.cross_entropy(logits.transpose(1, 2), y))


def test_transformer_sites_reproduce_the_forward_pass():
    torch.manual_seed(3)
    model = GPT(GPTConfig(n_layer=3, n_head=2, n_embd=16, block_size=24)).eval()
    x, y = torch.randint(0, 32, (2, 12)), torch.randint(0, 32, (2, 12))
    runner = SiteRunner(model)
    assert runner.site_names == ['emb', 'L1', 'L2', 'L3']
    with torch.no_grad():
        _, expected = model(x, y)
    assert _nll(runner.run(x).logits, y) == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.parametrize('mode, depth_steps', [('temporal', 1), ('hybrid', 1), ('hybrid', 3)])
def test_settled_sites_reproduce_the_alignment_fixed_point(mode, depth_steps):
    model = _recurrent(mode)
    x, y = torch.randint(0, 32, (2, 12)), torch.randint(0, 32, (2, 12))
    with torch.no_grad():
        p = align.prelude(model, x)
        expected = align.final_losses(model, p, align.settle(model, p, 12, depth_steps), y, depth_steps)
    run = SiteRunner(model, depth_steps).run(x, passes=12)
    assert run.passes == 12
    assert _nll(run.logits, y) == pytest.approx(float(expected.mean()), abs=1e-5)


def test_tolerance_settling_matches_exact_settling():
    model = _recurrent('temporal')
    x = torch.randint(0, 32, (2, 12))
    runner = SiteRunner(model, tolerance=1e-7, max_passes=64)
    settled, exact = runner.run(x), runner.run(x, passes=12)
    assert settled.memory_change < 1e-7
    torch.testing.assert_close(settled.logits, exact.logits, atol=1e-5, rtol=1e-5)


def test_depth_sites_reproduce_the_training_graph():
    model = _recurrent('depth')
    x, y = torch.randint(0, 32, (2, 12)), torch.randint(0, 32, (2, 12))
    schedule = RecurrenceSchedule((False,) * 3, (True,) * 3)
    with torch.no_grad():
        _, expected = model(x, y, schedule=schedule)
    runner = SiteRunner(model, depth_steps=4)
    assert runner.sites[-1].applications == 2 + 4 * 2 + 2
    assert 'Dmix@4' in runner.site_names and 'Tmix' not in runner.site_names
    assert _nll(runner.run(x).logits, y) == pytest.approx(float(expected), abs=1e-6)


def test_captures_and_edits_act_at_named_sites():
    model = _recurrent('hybrid')
    x = torch.randint(0, 32, (2, 12))
    runner = SiteRunner(model, depth_steps=2)
    index = (torch.tensor([0, 1]), torch.tensor([3, 7]))
    base = runner.run(x, capture=['Tmix', 'L3@2'], index=index, passes=4)
    assert base.captures['Tmix'].shape == (2, 16)
    identity = runner.run(x, edits={'L3@2': lambda h: h}, passes=4)
    torch.testing.assert_close(identity.logits, base.logits)

    push = torch.randn(16, generator=torch.Generator().manual_seed(1)) * 5

    def edit_last(h):
        h = h.clone()
        h[:, 11] += push
        return h

    # An edit at the final position is visible at that position only.
    edited = runner.run(x, edits={'L4@1': edit_last}, passes=4)
    torch.testing.assert_close(edited.logits[:, :11], base.logits[:, :11])
    assert not torch.allclose(edited.logits[:, 11], base.logits[:, 11])

    # A T-source edit reaches later positions only through temporal memory.
    def edit_early(h):
        h = h.clone()
        h[:, 2] += push
        return h

    edited = runner.run(x, edits={'L5': edit_early}, passes=4)
    assert not torch.allclose(edited.logits[:, 3], base.logits[:, 3])


def test_probe_points_label_boards_and_side_to_move():
    text = ';1.e4 e5 2.Nf3 Nc6 3.Bb5 a6;1.d4 d5 2.c4'
    points = probe_points(text)
    for point in points:
        assert text[point.index] == ('.' if point.kind == 'dot' else ' ')
        assert point.board.turn == (point.kind != 'space_black')
    first = [p for p in points if p.game == 0]
    assert [p.kind for p in first[:4]] == ['dot', 'space_black', 'space_white', 'dot']
    assert [p.next_san for p in first[:4]] == ['e4', 'e5', 'Nf3', 'Nf3']
    after_e4 = first[1].board
    assert after_e4.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    # The final token of the second game is incomplete; its board is still labelled.
    assert points[-1].kind == 'dot' and points[-1].next_san is None and points[-1].game == 1
    assert probe_points(text, limit=10)[-1].index < 10


def test_relative_encoding_puts_the_side_to_move_first():
    board = chess.Board()
    board.push_san('e4')
    absolute, relative = encode(board, False), encode(board, True)
    assert absolute[chess.E4] == 1 and relative[chess.E4] == 7
    assert absolute[chess.E8] == 12 and relative[chess.E8] == 6


def test_probe_fits_a_linear_rule():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 8)).astype(np.float32)
    y = np.stack([(x[:, 0] > 0), (x[:, 1] + x[:, 2] > 0)], 1).astype(np.int64)
    probe = fit(x, y, 2, epochs=20, batch=128, lr=1e-2)
    assert (probe.predict(torch.from_numpy(x)).numpy() == y).mean() > 0.97


def test_probe_coefficients_set_the_requested_margin():
    from interp.probes import Probe
    from interp.steering import probe_coefficients
    torch.manual_seed(0)
    probe = Probe(torch.randn(8), torch.randn(8, 3 * 4), torch.randn(3 * 4), 3, 4)
    x = torch.randn(5, 8)
    head, source, target = torch.tensor([0, 1, 2, 0, 1]), torch.tensor([1, 2, 3, 0, 2]), torch.zeros(5, dtype=torch.long)
    head[3], source[3], target[3] = 2, 1, 3
    coefficients, directions = probe_coefficients(probe, x, head, source, target, 4.0)
    logits = probe.logits(x + coefficients[:, None] * directions)
    rows = torch.arange(5)
    margin = logits[rows, head, target] - logits[rows, head, source]
    reached = coefficients > 0
    torch.testing.assert_close(margin[reached], torch.full_like(margin[reached], 4.0), atol=1e-4, rtol=0)
    assert (margin[~reached] >= 4.0 - 1e-4).all()


def test_additive_edit_starts_at_each_rows_point():
    from interp.steering import additive_edit
    h = torch.zeros(2, 5, 3)
    edited = additive_edit(torch.ones(2, 3), torch.tensor([1, 3]))(h)
    assert edited[0, :, 0].tolist() == [0, 1, 1, 1, 1]
    assert edited[1, :, 0].tolist() == [0, 0, 0, 1, 1]


def test_greedy_moves_match_unbatched_decoding():
    from interp.steering import greedy_moves
    torch.manual_seed(5)
    model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=32)).eval()
    runner = SiteRunner(model)
    itos = {i: chr(ord('a') + i) for i in range(32)}
    itos[0] = ' '
    sequences = [torch.randint(1, 32, (n,)).tolist() for n in (3, 6, 4)]
    batched = greedy_moves(runner, sequences, itos, 'cpu', max_chars=5)
    assert batched == [greedy_moves(runner, [s], itos, 'cpu', max_chars=5)[0] for s in sequences]


def test_move_cycle_roles_and_latest_move_squares():
    from experiments.interp.board_state.cycle import ROLES, label_row
    text = ';1.e4 e5 2.Nf3+ Nc6 10.O-O'
    role, current, previous, valid = label_row(text, len(text))
    roles = {i: ROLES[r] for i, r in enumerate(role) if valid[i]}
    assert [roles[i] for i in range(1, 9)] == [
        'digit_last', 'dot', 'first_white', 'last_white', 'space_black', 'first_black', 'last_black', 'space_number']
    ten = text.index('10.')
    assert (roles[ten], roles[ten + 1], roles[ten + 2]) == ('digit', 'digit_last', 'dot')
    assert roles[text.index('+')] == 'check_white'
    # The move is applied at its last character, not before.
    e4_first, e4_last = text.index('e4'), text.index('e4') + 1
    changed = lambda i: {chess.square_name(s) for s in np.flatnonzero(current[i] != previous[i])}
    assert changed(e4_last) == {'e2', 'e4'}
    assert not (current[e4_first] != encode(chess.Board(), False)).any()
    # Black's first character still has White's e4 as the latest move.
    assert changed(text.index('e5')) == {'e2', 'e4'}
    # The incomplete final move (no delimiter after it) is not labelled.
    assert not valid[text.index('O-O'):].any()


def test_update_pairs_each_decision_with_the_previous_one():
    from experiments.interp.board_state.update import paired_points
    labels = dict(kind=np.array([0, 2, 0, 1, 2]), row=np.zeros(5, int), game=np.zeros(5, int),
                  ply=np.array([0, 1, 2, 2, 3]))
    current, previous = paired_points(labels)
    assert sorted(zip(current.tolist(), previous.tolist())) == [(1, 0), (2, 1), (4, 2)]


def test_scaled_memory_and_mixer_terms():
    from experiments.interp.board_state.memory_scale import scaled_memory
    from experiments.interp.board_state.mixer_terms import mixer_terms
    model = _recurrent('temporal')
    runner = SiteRunner(model)
    x = torch.randint(0, 32, (2, 12))
    index = (torch.tensor([0, 1]), torch.tensor([5, 9]))
    base = runner.run(x, capture=['L1', 'Tmix'], index=index, passes=12)
    with scaled_memory(model.temporal_mixer, 1.0):
        same = runner.run(x, passes=12)
    torch.testing.assert_close(same.logits, base.logits)
    with scaled_memory(model.temporal_mixer, 2.0):
        assert not torch.allclose(runner.run(x, passes=12).logits, base.logits)
    torch.testing.assert_close(runner.run(x, passes=12).logits, base.logits)  # restored
    # The two terms add up to the mixer output. Memory is the T-source output (L5 in this
    # six-layer model, L7 in the 20B arms) one position back.
    memory = runner.run(x, capture=['L5'], index=(index[0], index[1] - 1), passes=12).captures['L5']
    m_term, c_term = mixer_terms(model.temporal_mixer, memory.float().numpy(),
                                 base.captures['L1'].float().numpy(), 'cpu')
    np.testing.assert_allclose(m_term + c_term, base.captures['Tmix'].float().numpy(), atol=2e-3)


def test_blocked_attention_and_attention_to_previous():
    from experiments.interp.board_state.attention import attention_to_previous, blocked_attention
    torch.manual_seed(4)
    model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=24)).eval()
    runner = SiteRunner(model)
    x = torch.randint(0, 32, (1, 10))
    base = runner.run(x).logits
    forbid = torch.zeros(1, 24, 24, dtype=torch.bool)
    with blocked_attention(model, [0, 1], lambda: forbid):
        torch.testing.assert_close(runner.run(x).logits, base, atol=1e-5, rtol=1e-5)
    forbid[0, 6, 5] = True
    with blocked_attention(model, [0, 1], lambda: forbid):
        blocked = runner.run(x).logits
    torch.testing.assert_close(blocked[:, :6], base[:, :6], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(blocked[:, 6], base[:, 6])
    # Attention weights match a direct softmax over the causal scores.
    block, h = model.transformer.h[0], model.embed(x)[0]
    got = attention_to_previous(block, h, torch.tensor([4, 7]))
    q, k, _ = block.attn.c_attn(block.ln_1(h)).split(16, dim=-1)
    for qi, pos in enumerate([4, 7]):
        for head in range(2):
            s = q[pos, head * 8:(head + 1) * 8] @ k[:pos + 1, head * 8:(head + 1) * 8].T / 8 ** 0.5
            torch.testing.assert_close(got[head, qi], s.softmax(-1)[pos - 1], atol=1e-5, rtol=1e-5)
