"""The move vocabulary: ChessBench's 1,968 from/to moves plus two special tokens.

Moves are every from/to pair a queen or knight could make on an empty board
(1,792), plus promotions to queen, rook, bishop and knight for both colours
(176), sorted lexicographically by their UCI string as in ChessBench. Castling
is the king's move (``e1g1``) and en passant is the pawn's move (``e5d6``),
which is how python-chess writes standard chess in UCI.

Token ids: moves are ``0 .. 1967``, so an output layer over moves shares ids
with the input vocabulary; ``GAME_START`` and ``PAD`` follow.
"""

import hashlib

import chess


def _moves():
    moves = set()
    for square in chess.SQUARES:
        for piece in (chess.QUEEN, chess.KNIGHT):
            board = chess.BaseBoard.empty()
            board.set_piece_at(square, chess.Piece(piece, chess.WHITE))
            moves.update(chess.Move(square, target).uci() for target in board.attacks(square))
    for from_rank, to_rank in ((6, 7), (1, 0)):
        for file in range(8):
            for step in (-1, 0, 1):
                if 0 <= file + step < 8:
                    for piece in 'qrbn':
                        moves.add(chess.square_name(chess.square(file, from_rank)) +
                                  chess.square_name(chess.square(file + step, to_rank)) + piece)
    return tuple(sorted(moves))


MOVES = _moves()
MOVE_COUNT = len(MOVES)
GAME_START = MOVE_COUNT
PAD = MOVE_COUNT + 1
VOCAB_SIZE = MOVE_COUNT + 2
MOVE_TO_ID = {uci: index for index, uci in enumerate(MOVES)}
VOCABULARY_SHA256 = hashlib.sha256('\n'.join(MOVES).encode()).hexdigest()

assert MOVE_COUNT == 1968


def move_id(move):
    """Token id of a ``chess.Move``. Raises ``KeyError`` for a move outside the vocabulary."""
    return MOVE_TO_ID[move.uci()]


def id_move(token):
    return chess.Move.from_uci(MOVES[token])


def legal_ids(board):
    """Sorted token ids of the legal moves in ``board``."""
    return sorted(MOVE_TO_ID[move.uci()] for move in board.legal_moves)
