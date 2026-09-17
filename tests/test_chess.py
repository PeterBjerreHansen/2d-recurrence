import chess
import pytest

from evaluation.chess import MoveValidator


def validate(text):
    validator = MoveValidator()
    for char in text:
        validator.feed(char)
    return validator.finish()


def test_stop_on_first_illegal_move_preserves_failure_board():
    result = validate(';1.e4 e5 2.Bh6 2.Nf3 ')
    board = chess.Board()
    board.push_san('e4')
    board.push_san('e5')
    assert result['termination_reason'] == 'illegal_move'
    assert result['offending_move'] == 'Bh6'
    assert result['board_before_failure'] == board.fen()
    assert result['legal_moves'] == 2 and result['attempted_moves'] == 3
    assert result['text'] == ';1.e4 e5 2.Bh6 '
    assert not result['pgn_parse_success']


def test_incomplete_character_prefix_is_truncated_not_illegal():
    result = validate(';1.e4 e5 2.N')
    assert result['termination_reason'] == 'generation_limit'
    assert result['legal_moves'] == result['attempted_moves'] == 2
    assert result['incomplete_move'] == '2.N'


def test_malformed_move_and_null_move():
    assert validate(';1.xyz ')['termination_reason'] == 'malformed_move'
    assert validate(';1.-- ')['termination_reason'] == 'illegal_move'


def test_checkmate_stops_without_waiting_for_more_output():
    result = validate(';1.f3 e5 2.g4 Qh4# 3.a3 ')
    assert result['termination_reason'] == 'board_terminal'
    assert result['legal_moves'] == 4 and result['valid_termination']
    assert result['pgn_parse_success']
    assert result['text'].endswith('Qh4# ')


def test_castling_and_promotion():
    result = validate(';1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.O-O ')
    assert result['legal_moves'] == 7 and result['offending_move'] is None
    validator = MoveValidator()
    validator.board = chess.Board('7k/P7/8/8/8/8/8/K7 w - - 0 1')
    for char in 'a8=Q+ ':
        validator.feed(char)
    assert validator.legal_moves == 1


def test_prompt_moves_are_not_counted_as_generated():
    validator = MoveValidator()
    validator.read_prompt(';1.e4 e5 2.')
    for char in 'Nf3 ':
        validator.feed(char)
    assert validator.prompt_moves == 2 and validator.legal_moves == 1
    with pytest.raises(ValueError, match='boundary'):
        MoveValidator().read_prompt(';1.e4')


def test_game_boundary_and_declared_result_have_distinct_reasons():
    assert validate(';1.e4 e5;')['termination_reason'] == 'game_boundary'
    assert validate(';1.e4 e5 1-0 ')['termination_reason'] == 'declared_result'
