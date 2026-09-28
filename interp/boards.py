"""Board-state labels at the characters just before a move is written.

Rows look like ``;1.e4 e5 2.Nf3 Nc6 ...;1.d4 ...``. A *point* is an input
character whose board is fully determined and whose side to move is known:

``dot``          the ``.`` of a white move number; the next character starts white's SAN
``space_white``  the space after black's move; the next characters are a move number
``space_black``  the space after white's move; the next character starts black's SAN

``dot`` follows Karvonen's probing position. The two space kinds share one
input token, so side to move cannot be read from the current character.

Squares use python-chess order (a1 = 0, h8 = 63) with 13 classes: 0 empty,
1-6 pawn, knight, bishop, rook, queen, king of one colour, 7-12 of the other.
The *absolute* encoding puts white first; the *relative* encoding puts the side
to move first ("mine" / "theirs").
"""

from dataclasses import dataclass
import re

import chess
import numpy as np

KINDS = ('dot', 'space_white', 'space_black')
EMPTY = 0
CLASS_NAMES = ('empty',) + tuple(f'{side}_{piece}' for side in ('A', 'B')
                                 for piece in ('pawn', 'knight', 'bishop', 'rook', 'queen', 'king'))
_MOVE_NUMBER = re.compile(r'\d+\.')


@dataclass
class Point:
    index: int              # row character index (an input position)
    kind: str
    board: chess.Board      # position before the next move
    ply: int                # moves completed in this game
    game: int               # zero-based game ordinal within the row
    next_san: str | None    # the move written next, if complete within the row


def piece_class(piece, first_colour):
    """Class of ``piece`` when ``first_colour`` takes classes 1-6."""
    if piece is None:
        return EMPTY
    return piece.piece_type + (0 if piece.color == first_colour else 6)


def encode(board, relative):
    """[64] int8 square classes, absolute (white first) or relative (side to move first)."""
    first = board.turn if relative else chess.WHITE
    return np.array([piece_class(board.piece_at(square), first) for square in chess.SQUARES], dtype=np.int8)


def class_of(board, square, relative):
    return piece_class(board.piece_at(square), board.turn if relative else chess.WHITE)


def probe_points(text, limit=None):
    """Points of every game in ``text`` with index below ``limit``; a game stops at a replay failure."""
    limit = len(text) if limit is None else limit
    points = []
    starts = [index for index, char in enumerate(text) if char == ';']
    for game, start in enumerate(starts):
        stop = starts[game + 1] if game + 1 < len(starts) else len(text)
        board = chess.Board()
        position = start + 1
        for token in text[start + 1:stop].split(' '):
            token_start, token_stop = position, position + len(token)
            position = token_stop + 1
            if not token or token_start >= limit:
                break
            number = _MOVE_NUMBER.match(token)
            offset = number.end() if number else 0
            san = token[offset:]
            complete = bool(san) and token_stop < len(text)
            next_san = san if complete else None
            if token_start > start + 1:  # a space precedes every token but the first
                kind = 'space_white' if number else 'space_black'
                if (board.turn == chess.WHITE) != (kind == 'space_white'):
                    break
                points.append(Point(token_start - 1, kind, board.copy(stack=False), board.ply(), game, next_san))
            if number:
                dot = token_start + offset - 1
                if board.turn != chess.WHITE:
                    break
                if dot < limit:
                    points.append(Point(dot, 'dot', board.copy(stack=False), board.ply(), game, next_san))
            if not complete:
                break
            try:
                board.push_san(san)
            except ValueError:
                break
    return points
