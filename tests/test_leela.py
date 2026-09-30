"""Leela training data: record layout, game reconstruction and conversion to move tokens."""

import gzip
import io
import os
import random
import tarfile

import chess
import numpy as np
import pytest

from moves.leela import V6, SkippedGame, board_in_frame, convert_game, policy_move, read_archive
from moves.leela_policy import POLICY_INDEX
from moves.vocab import MOVE_TO_ID, legal_ids


def _leela_index(move, board):
    """Inverse of ``policy_move``: the policy index of ``move`` in ``board``, as Leela encodes it."""
    uci = move.uci()
    if move.promotion == chess.KNIGHT:
        uci = uci[:4]
    if board.is_castling(move):
        # Leela writes castling as the king taking its own rook.
        rook_file = 'h' if board.is_kingside_castling(move) else 'a'
        uci = uci[:2] + rook_file + uci[3]
    if board.turn == chess.BLACK:
        uci = ''.join(c if c.isalpha() else str(9 - int(c)) for c in uci)
    return POLICY_INDEX.index(uci)


def _record(board, played, rng):
    """One V6 record of ``board`` with a random distribution over its legal moves."""
    record = np.zeros(1, dtype=V6)[0]
    record['version'], record['input_format'] = 6, 1
    record['probabilities'][:] = -1
    weights = [rng.random() for _ in board.legal_moves]
    for move, weight in zip(board.legal_moves, weights):
        record['probabilities'][_leela_index(move, board)] = weight / sum(weights)
    for square, letter in board_in_frame(board).items():
        bit = (square // 8) * 8 + 7 - square % 8
        record['planes'][('PNBRQKpnbrqk').index(letter)] |= np.uint64(1 << bit)
    record['played_idx'] = _leela_index(played, board)
    record['root_q'], record['visits'] = rng.uniform(-1, 1), 400
    return record


def _game_records(moves, seed=0):
    rng = random.Random(seed)
    board, records = chess.Board(), []
    for move in moves:
        records.append(_record(board, move, rng))
        board.push(move)
    return np.array(records, dtype=V6)


def _random_moves(seed, plies=255):
    rng, board, moves = random.Random(seed), chess.Board(), []
    while len(moves) < plies and not board.is_game_over():
        move = rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
        moves.append(move)
        board.push(move)
    return moves


def _features(moves):
    board, seen = chess.Board(), set()
    for move in moves:
        colour = 'white' if board.turn else 'black'
        if move.promotion:
            seen.add((colour, chess.piece_symbol(move.promotion)))
        if board.is_en_passant(move):
            seen.add((colour, 'en passant'))
        if board.is_castling(move):
            seen.add((colour, 'castling'))
        board.push(move)
    return seen


EN_PASSANT_GAMES = [['e2e4', 'a7a6', 'e4e5', 'd7d5', 'e5d6'],            # White captures en passant
                    ['a2a3', 'd7d5', 'a3a4', 'd5d4', 'e2e4', 'd4e3']]    # Black captures en passant


def test_policy_index_round_trips_every_legal_move_of_random_games():
    covered = set()
    games = [[chess.Move.from_uci(uci) for uci in game] for game in EN_PASSANT_GAMES]
    games += [_random_moves(seed) for seed in range(300)]
    for moves in games:
        board = chess.Board()
        for move in moves:
            for legal in board.legal_moves:
                assert policy_move(_leela_index(legal, board), board) == legal
            board.push(move)
        covered |= _features(moves)
    assert covered == {(c, f) for c in ('white', 'black') for f in ('q', 'r', 'b', 'n', 'en passant', 'castling')}


def test_games_convert_to_move_tokens_with_their_distributions():
    moves = _random_moves(3)
    records = _game_records(moves)
    game = convert_game(records)
    assert game.plies.tolist() == [MOVE_TO_ID[m.uci()] for m in moves]
    board = chess.Board()
    for index, move in enumerate(moves):
        assert game.legal[index].tolist() == legal_ids(board)
        stored = records[index]['probabilities']
        for token, probability in zip(game.legal[index], game.probabilities[index]):
            assert probability == stored[_leela_index(chess.Move.from_uci(
                next(uci for uci, i in MOVE_TO_ID.items() if i == token)), board)]
        board.push(move)
    assert game.probabilities[0].sum() == pytest.approx(1, abs=1e-5)


def _tampered(change):
    records = _game_records(_random_moves(5, plies=20)).copy()
    change(records)
    return records


@pytest.mark.parametrize('change, reason', [
    (lambda r: r.__setitem__(('planes'), np.roll(r['planes'], 1, axis=0)), 'start position'),
    (lambda r: r['planes'][4].__setitem__(0, r['planes'][4][0] ^ np.uint64(1 << 20)), 'stored board'),
    (lambda r: r['probabilities'][2].__setitem__(np.flatnonzero(r['probabilities'][2] < 0)[0], 0.1),
     'exactly the legal moves'),
    (lambda r: r['input_format'].__setitem__(0, 3), 'classical input format'),
    (lambda r: r['probabilities'][2].__setitem__(r['probabilities'][2] >= 0, 0), 'no finite positive mass'),
    (lambda r: r['played_idx'].__setitem__(3, 65535), 'out of range'),
])
def test_games_that_do_not_check_out_are_skipped(change, reason):
    with pytest.raises(SkippedGame, match=reason):
        convert_game(_tampered(change))


def test_a_game_without_records_is_skipped():
    with pytest.raises(SkippedGame, match='no records'):
        convert_game(_game_records(_random_moves(5, plies=4))[:0])


def test_an_illegal_played_move_is_skipped():
    records = _game_records(_random_moves(6, plies=10)).copy()
    legal = np.flatnonzero(records[3]['probabilities'] >= 0)
    records[3]['played_idx'] = next(i for i in range(1858) if i not in set(legal.tolist()))
    with pytest.raises(SkippedGame, match='illegal'):
        convert_game(records)


def write_archive(path, games):
    """A Leela-style archive: one gzip file per game, plus a LICENSE text file."""
    with tarfile.open(path, 'w') as archive:
        for index, records in enumerate(games):
            data = gzip.compress(records.tobytes())
            info = tarfile.TarInfo(f'hour/training.{index}.gz')
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        text = b'This collection of training data for Leela Chess Zero is made available under the ODbL.'
        info = tarfile.TarInfo('hour/LICENSE')
        info.size = len(text)
        archive.addfile(info, io.BytesIO(text))


def test_castling_is_read_from_leelas_king_takes_rook_encoding():
    board = chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    assert policy_move(POLICY_INDEX.index('e1h1'), board) == chess.Move.from_uci('e1g1')
    assert policy_move(POLICY_INDEX.index('e1a1'), board) == chess.Move.from_uci('e1c1')
    board.turn = chess.BLACK   # Black's moves are rank-mirrored: e8h8 is written e1h1
    assert policy_move(POLICY_INDEX.index('e1h1'), board) == chess.Move.from_uci('e8g8')


def test_archives_yield_one_record_array_per_game_file(tmp_path):
    games = [_game_records(_random_moves(seed, plies=12)) for seed in range(3)]
    path = tmp_path / 'training.tar'
    write_archive(path, games)
    read = list(read_archive(path))
    assert [name for name, _ in read] == [f'hour/training.{i}.gz' for i in range(3)]
    for (_, records), original in zip(read, games):
        assert records.tobytes() == original.tobytes()


ARCHIVE = os.environ.get('LEELA_ARCHIVE')


@pytest.mark.skipif(not ARCHIVE, reason='Set LEELA_ARCHIVE to a downloaded Leela training archive')
def test_a_real_leela_archive_converts_with_only_chess960_games_skipped():
    converted, skipped = 0, {}
    for index, (_, records) in enumerate(read_archive(ARCHIVE)):
        if index == 300:
            break
        try:
            game = convert_game(records)
            converted += 1
            assert all(abs(p.sum() - 1) < 1e-3 for p in game.probabilities)
        except SkippedGame as error:
            skipped[str(error)] = skipped.get(str(error), 0) + 1
    assert converted >= 270
    assert set(skipped) <= {'does not start at the standard start position'}
