"""Leela Chess Zero training data (V6): parse archives, rebuild games, convert targets to move tokens.

Leela publishes its self-play training data as hourly tar archives of
gzip files, one game per file, at storage.lczero.org/files/training_data/
(Open Database License). Each record is one position (8,356 bytes, V6 format,
https://lczero.org/dev/wiki/training-data-format-versions/). This module reads
the classical input format (``input_format == 1``): the board is stored from
the side to move's view, rank-mirrored for Black, with each rank's files in
reverse bit order. Castling is written as the king taking its own rook
(``e1h1``); a plain pawn move to the last rank is a knight promotion. Archives
also hold a ``LICENSE`` text file, which is skipped.

A game is kept only if it starts at the standard start position (Chess960 games
are skipped) and every record checks out when the game is replayed with
python-chess:

- the stored board equals the replayed board, in the record's frame;
- the moves with non-negative probability are exactly the legal moves;
- the played move is legal, and leads to the next record's board.

Targets per position are Leela's search distribution over the legal moves
(``probabilities``, from ~250-700 visits in test80). The records' values and
visit counts are not kept; the archives can be re-read if an analysis needs them.
"""

from dataclasses import dataclass
import gzip
import tarfile

import chess
import numpy as np

from moves.leela_policy import POLICY_INDEX
from moves.vocab import MOVE_TO_ID

V6 = np.dtype([
    ('version', '<u4'), ('input_format', '<u4'), ('probabilities', '<f4', 1858), ('planes', '<u8', 104),
    ('castling_us_ooo', 'u1'), ('castling_us_oo', 'u1'), ('castling_them_ooo', 'u1'), ('castling_them_oo', 'u1'),
    ('side_to_move_or_enpassant', 'u1'), ('rule50_count', 'u1'), ('invariance_info', 'u1'), ('dummy', 'u1'),
    ('root_q', '<f4'), ('best_q', '<f4'), ('root_d', '<f4'), ('best_d', '<f4'), ('root_m', '<f4'), ('best_m', '<f4'),
    ('plies_left', '<f4'), ('result_q', '<f4'), ('result_d', '<f4'), ('played_q', '<f4'), ('played_d', '<f4'),
    ('played_m', '<f4'), ('orig_q', '<f4'), ('orig_d', '<f4'), ('orig_m', '<f4'), ('visits', '<u4'),
    ('played_idx', '<u2'), ('best_idx', '<u2'), ('policy_kld', '<f4'), ('reserved', '<u4')])
assert V6.itemsize == 8356

_PIECES = 'PNBRQKpnbrqk'   # planes 0-5: side to move's pieces, 6-11: the opponent's


class SkippedGame(Exception):
    """A game that is not converted, with the reason."""


@dataclass
class LeelaGame:
    plies: np.ndarray            # [n] our move ids
    legal: list                  # per position: sorted legal move ids (uint16)
    probabilities: list          # per position: Leela's search probability of each legal move (float32)


def read_archive(path):
    """(file name, V6 records) for every game file in a Leela training archive."""
    with tarfile.open(path, 'r|*') as archive:
        for member in archive:
            if member.isfile() and member.name.endswith('.gz'):
                data = gzip.decompress(archive.extractfile(member).read())
                if len(data) % V6.itemsize:
                    raise ValueError(f'{member.name}: {len(data)} bytes is not a whole number of V6 records')
                yield member.name, np.frombuffer(data, dtype=V6)


def record_board(record):
    """Pieces of a record's current position, square -> letter, in the record's frame (mover upper case)."""
    pieces = {}
    for plane, letter in enumerate(_PIECES):
        bits = int(record['planes'][plane])
        for bit in range(64):
            if bits >> bit & 1:
                pieces[(bit // 8) * 8 + 7 - bit % 8] = letter
    return pieces


def board_in_frame(board):
    """``board`` as a record stores it: the side to move upper case, rank-mirrored when Black moves."""
    mover = board.turn
    pieces = {}
    for square, piece in board.piece_map().items():
        letter = piece.symbol().upper()
        pieces[square if mover == chess.WHITE else chess.square_mirror(square)] = (
            letter if piece.color == mover else letter.lower())
    return pieces


def policy_move(index, board):
    """The ``chess.Move`` that Leela's policy index ``index`` denotes in ``board``."""
    uci = POLICY_INDEX[index]
    if board.turn == chess.BLACK:
        uci = ''.join(c if c.isalpha() else str(9 - int(c)) for c in uci)
    move = chess.Move.from_uci(uci)
    piece, target = board.piece_at(move.from_square), board.piece_at(move.to_square)
    if piece is None:
        return move
    if (piece.piece_type == chess.KING and target is not None and target.piece_type == chess.ROOK and
            target.color == piece.color):
        # King takes own rook: castling, to the g- or c-file.
        file = 6 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 2
        return chess.Move(move.from_square, chess.square(file, chess.square_rank(move.from_square)))
    if move.promotion is None and piece.piece_type == chess.PAWN and chess.square_rank(move.to_square) in (0, 7):
        return chess.Move(move.from_square, move.to_square, chess.KNIGHT)
    return move


def convert_game(records):
    """A verified ``LeelaGame`` from one game's records; raises ``SkippedGame`` otherwise."""
    if len(records) == 0:
        raise SkippedGame('no records')
    if int(records['input_format'][0]) != 1 or int(records['version'][0]) != 6:
        raise SkippedGame('not V6 with the classical input format')
    board = chess.Board()
    if record_board(records[0]) != board_in_frame(board):
        raise SkippedGame('does not start at the standard start position')
    plies, legal_sets, probabilities = [], [], []
    for index, record in enumerate(records):
        if record_board(record) != board_in_frame(board):
            raise SkippedGame(f'record {index}: stored board differs from the replayed board')
        stored = record['probabilities']
        moves = {policy_move(i, board): float(stored[i]) for i in np.flatnonzero(stored >= 0)}
        if set(moves) != set(board.legal_moves):
            raise SkippedGame(f'record {index}: non-negative probabilities are not exactly the legal moves')
        total = sum(moves.values())
        if not np.isfinite(total) or total <= 0:
            raise SkippedGame(f'record {index}: the probabilities have no finite positive mass')
        if int(record['played_idx']) >= len(POLICY_INDEX):
            raise SkippedGame(f'record {index}: the played move index is out of range')
        order = sorted(moves, key=lambda move: MOVE_TO_ID[move.uci()])
        legal_sets.append(np.array([MOVE_TO_ID[move.uci()] for move in order], dtype=np.uint16))
        probabilities.append(np.array([moves[move] for move in order], dtype=np.float32))
        played = policy_move(int(record['played_idx']), board)
        if played not in moves:
            raise SkippedGame(f'record {index}: the played move is illegal')
        plies.append(MOVE_TO_ID[played.uci()])
        board.push(played)
    return LeelaGame(np.array(plies, dtype=np.uint16), legal_sets, probabilities)
