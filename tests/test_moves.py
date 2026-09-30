"""Move tokens, parsing, the legality loss and the engine teacher."""

import os
from pathlib import Path
import random
import re
import shutil

import chess
import chess.engine
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from moves.games import MAX_PLIES, parse_row, random_game
from moves.objectives import LegalTargets
from moves.teacher import CHECKMATE, EXACT, LOWER_BOUND, RULE_DRAW, UPPER_BOUND, Teacher, history_key
from moves.values import mate_after_move, q_from_centipawns, q_from_mate
from moves.vocab import GAME_START, MOVE_COUNT, MOVES, PAD, id_move, legal_ids, move_id
from tests.move_fakes import MaterialEngine

ROOT = Path(__file__).resolve().parents[1]
REAL_ROWS = ROOT / 'data/chess_8M_v1/train.bin'


MOVE_NUMBER = re.compile(r'^\d+\.')


def san_plies(text):
    """The complete SAN plies of each game in a row, as written (for round-trip checks)."""
    segments = text.split(';')[1:]
    games = []
    for index, segment in enumerate(segments):
        tokens = segment.split(' ')
        if index == len(segments) - 1:
            tokens = tokens[:-1]
        plies = [MOVE_NUMBER.sub('', token) for token in tokens]
        games.append([ply for ply in plies if ply])
    return [game for game in games if game]


def to_san(game):
    """SAN of each ply of a game given as ply ids."""
    board, plies = chess.Board(), []
    for token in game:
        move = id_move(token)
        plies.append(board.san(move))
        board.push(move)
    return plies


def _real_rows(count):
    import pickle
    if not REAL_ROWS.exists():
        pytest.skip('The Lichess character data is not present')
    with open(ROOT / 'data/chess_8M_v1/meta.pkl', 'rb') as stream:
        itos = pickle.load(stream)['itos']
    rows = np.memmap(REAL_ROWS, dtype=np.uint8, mode='r').reshape(-1, 1024)
    return [''.join(itos[int(c)] for c in rows[index]) for index in range(count)]


# --- vocabulary ---

def test_vocabulary_is_chessbench_sized_sorted_and_complete():
    assert MOVE_COUNT == len(set(MOVES)) == 1968
    assert list(MOVES) == sorted(MOVES)
    assert (GAME_START, PAD) == (1968, 1969)
    for uci in ('e1g1', 'e1c1', 'e8g8', 'e8c8', 'e5d6', 'e7e8q', 'e7e8', 'a2a1n', 'b7a8r', 'g1f3'):
        assert uci in MOVES
    assert all(move_id(id_move(token)) == token for token in range(MOVE_COUNT))


@pytest.mark.parametrize('fen', [
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',        # castling both ways, both sides
    'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',            # en passant
    '1n2k3/P1P5/8/8/8/8/p1p5/1N2K3 w - - 0 1',      # promotions with and without capture
    '1n2k3/P1P5/8/8/8/8/p1p5/1N2K3 b - - 0 1',
])
def test_special_moves_are_in_the_vocabulary(fen):
    board = chess.Board(fen)
    assert len(legal_ids(board)) == board.legal_moves.count()


def test_random_games_use_only_vocabulary_moves_and_end_by_rule_or_cap():
    rng = random.Random(5)
    for _ in range(30):
        game, sets = random_game(rng)
        board = chess.Board()
        for token, legal in zip(game, sets):
            assert legal == legal_ids(board) and token in legal
            board.push(id_move(token))
        assert len(game) == MAX_PLIES or board.is_game_over()


# --- parsing and conversion ---

def test_parsing_drops_the_ply_cut_off_by_the_row_end():
    assert parse_row(';1.e4 e5 2.Nf3 Nc') == ([[move_id(chess.Move.from_uci(u)) for u in ('e2e4', 'e7e5', 'g1f3')]], 0)
    # A complete-looking final ply without a delimiter is dropped too: it may be cut (Qh5 of Qh5+).
    assert len(parse_row(';1.e4 e5 2.Qh5')[0][0]) == 2
    games, failures = parse_row(';1.e4 e5 ;1.d4 d5 2.c4 ')
    assert failures == 0 and [len(game) for game in games] == [2, 3]


def test_san_round_trip_on_fixture_rows(prepared_data):
    import pickle
    with open(prepared_data / 'meta.pkl', 'rb') as stream:
        itos = pickle.load(stream)['itos']
    rows = np.memmap(prepared_data / 'train.bin', dtype=np.uint8, mode='r').reshape(-1, 1024)
    for row in rows:
        text = ''.join(itos[int(c)] for c in row)
        games, failures = parse_row(text)
        assert failures == 0
        assert [to_san(game) for game in games] == san_plies(text)


def test_san_round_trip_on_lichess_rows():
    for text in _real_rows(300):
        games, failures = parse_row(text)
        assert failures == 0
        assert [to_san(game) for game in games] == san_plies(text)
        assert all(len(game) <= MAX_PLIES for game in games)


# --- losses ---

def _sparse(positions, legal):
    owner = torch.cat([torch.full((len(moves),), index) for index, moves in enumerate(legal)])
    moves = torch.cat([torch.tensor(moves) for moves in legal])
    return dict(positions=torch.tensor(positions), owner=owner, moves=moves)


def test_legal_loss_is_mean_binary_cross_entropy_at_supervised_positions_only():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7, requires_grad=True)
    legal = [[1, 4], [0, 2, 6]]
    targets = LegalTargets(**_sparse([1, 5], legal))
    dense = torch.zeros(2, 7)
    for row, moves in enumerate(legal):
        dense[row, moves] = 1
    expected = F.binary_cross_entropy_with_logits(logits.reshape(-1, 7)[[1, 5]], dense)
    loss = targets.loss(logits)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    touched = logits.grad.reshape(-1, 7).abs().sum(-1) > 0
    assert touched.tolist() == [False, True, False, False, False, True]
    metrics = targets.metrics(logits.detach())
    assert metrics['positions'] == 2 and metrics['legal_moves'] == 5


# --- values and the teacher ---

def test_value_scale_orders_mates_outside_centipawns():
    assert q_from_centipawns(0) == 0.5
    assert q_from_centipawns(5000) == q_from_centipawns(1500) == pytest.approx(0.996, abs=5e-4)
    assert q_from_centipawns(-1500) == pytest.approx(0.004, abs=5e-4)
    wins = [q_from_mate(n) for n in (1, 2, 3, 20, 200)]
    losses = [q_from_mate(-n) for n in (1, 2, 3, 20, 200)]
    assert wins == sorted(wins, reverse=True) and len(set(wins)) == 5
    assert losses == sorted(losses) and len(set(losses)) == 5
    assert min(wins) > q_from_centipawns(1500) and max(losses) < q_from_centipawns(-1500)
    assert q_from_mate(1) == pytest.approx(0.999) and q_from_mate(-1) == pytest.approx(0.001)
    with pytest.raises(ValueError):
        q_from_mate(0)


def test_mate_distance_counts_the_move_itself():
    white, black = chess.WHITE, chess.BLACK
    # After White's move, Black to move is mated in 1 more White move: the move is mate in 2.
    assert mate_after_move(chess.engine.PovScore(chess.engine.Mate(-1), black).pov(white)) == 2
    # After White's move, Black mates in 3: White is mated in 3.
    assert mate_after_move(chess.engine.PovScore(chess.engine.Mate(3), black).pov(white)) == -3
    assert mate_after_move(chess.engine.PovScore(chess.engine.Cp(50), black).pov(white)) is None


@pytest.mark.parametrize('fen', ['4k3/8/8/8/8/8/8/3QK3 w - - 0 1', '3qk3/8/8/8/8/8/8/4K3 b - - 0 1'])
def test_values_are_from_the_movers_perspective(fen):
    ids, labels = Teacher(MaterialEngine(), 5).label(chess.Board(fen))
    # The side to move is a queen up: every move that keeps the queen is winning for it.
    assert (labels['q'] > 0.9).sum() >= len(ids) - 3
    assert ids == legal_ids(chess.Board(fen))


def test_rule_outcomes_are_labelled_without_search():
    engine = MaterialEngine()
    board = chess.Board('6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1')    # Rd8 mates
    ids, labels = Teacher(engine, 5).label(board)
    mate = labels[ids.index(move_id(chess.Move.from_uci('d1d8')))]
    assert mate['kind'] == CHECKMATE and mate['q'] == pytest.approx(0.999)
    assert len(engine.calls) == len(ids) - 1
    stalemate = chess.Board('7k/8/6Q1/8/8/8/8/K7 w - - 0 1')      # Qf7 stalemates
    ids, labels = Teacher(MaterialEngine(), 5).label(stalemate)
    draw = labels[ids.index(move_id(chess.Move.from_uci('g6f7')))]
    assert draw['kind'] == RULE_DRAW and draw['q'] == 0.5


@pytest.mark.parametrize('reported, stored', [(None, EXACT), ('lowerbound', UPPER_BOUND),
                                              ('upperbound', LOWER_BOUND)])
def test_score_bounds_are_kept_and_flipped_to_the_movers_side(reported, stored):
    ids, labels = Teacher(MaterialEngine(bound=reported), 5).label(chess.Board())
    assert set(labels['bound'].tolist()) == {stored}


def test_the_engine_sees_the_history_and_a_new_game_per_search():
    engine = MaterialEngine()
    board = chess.Board()
    board.push_san('e4')
    Teacher(engine, 5).label(board)
    assert all(stack == 2 for _, stack, _ in engine.calls)
    assert len({id(game) for _, _, game in engine.calls}) == len(engine.calls)


def test_repetition_history_changes_the_label_of_the_same_position():
    # White is a queen up; knight shuffles make g1f3 a threefold repetition in one history only.
    start = '4k3/8/8/8/8/8/8/Q3K1N1 w - - 0 1'
    shuffle = ['g1f3', 'e8d8', 'f3g1', 'd8e8']
    repeated, fresh = chess.Board(start), chess.Board(start)
    for uci in shuffle * 2:
        repeated.push_uci(uci)
    assert repeated.fen().split(' ')[:4] == fresh.fen().split(' ')[:4]
    assert history_key(repeated) != history_key(fresh)
    teacher = Teacher(MaterialEngine(), 5)
    move = move_id(chess.Move.from_uci('g1f3'))
    ids, labels = teacher.label(repeated)
    capped = labels[ids.index(move)]
    ids, labels = teacher.label(fresh)
    free = labels[ids.index(move)]
    assert capped['claimable'] and capped['q'] == 0.5
    assert not free['claimable'] and free['q'] > 0.9


# --- the real engine (skipped when Stockfish is not installed) ---

STOCKFISH = os.environ.get('STOCKFISH_PATH') or shutil.which('stockfish')


@pytest.mark.skipif(not STOCKFISH, reason='Stockfish is not installed')
def test_stockfish_labels_are_reproducible_in_any_order():
    boards = []
    for moves in (['e2e4', 'e7e5', 'g1f3'], ['d2d4'], ['e2e4', 'c7c5']):
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
        boards.append(board)
    teacher = Teacher.open(STOCKFISH, 2000)
    try:
        forward = [teacher.label(board) for board in boards]
        backward = [teacher.label(board) for board in reversed(boards)][::-1]
    finally:
        teacher.close()
    for (ids_a, labels_a), (ids_b, labels_b) in zip(forward, backward):
        assert ids_a == ids_b and np.array_equal(labels_a, labels_b)


@pytest.mark.skipif(not STOCKFISH, reason='Stockfish is not installed')
@pytest.mark.parametrize('fen, best', [('6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1', 'd1d8'),
                                       ('3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1', 'd8d1')])
def test_stockfish_values_mates_for_either_colour(fen, best):
    teacher = Teacher.open(STOCKFISH, 2000)
    try:
        ids, labels = teacher.label(chess.Board(fen))
    finally:
        teacher.close()
    assert labels['q'][ids.index(move_id(chess.Move.from_uci(best)))] == pytest.approx(0.999)
    assert labels['q'].max() == pytest.approx(0.999)
