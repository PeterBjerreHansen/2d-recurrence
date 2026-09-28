"""Per-target annotations for stored PGN rows: character role, ply, game, and board.

Rows look like ``;1.e4 e5 2.Nf3 Nc6 ...;1.d4 ...``: games start after ``;``,
white tokens carry a move number, and there are no result strings. Target
index ``t`` predicts row character ``t + 1`` from inputs ``0..t``.

Character classes (by the target character's role):

``move_number``  digits and ``.`` of a move number
``move_first``   first SAN character (the move choice begins)
``move_body``    remaining SAN characters before any ``+``/``#``
``check_slot``   character right after the SAN core: ``+``/``#`` or the
                 delimiter. Whether it is a check marker is fixed by the board.
``delimiter``    delimiter after a ``+``/``#`` suffix
"""

from dataclasses import dataclass, field
import re

import chess
import numpy as np


CLASSES = ('move_number', 'move_first', 'move_body', 'check_slot', 'delimiter')
CLASS_CODES = {name: code for code, name in enumerate(CLASSES)}
UNKNOWN = -1
_MOVE_NUMBER = re.compile(r'\d+\.')


@dataclass
class Move:
    start: int                 # row index of the first SAN character
    san: str                   # SAN as written, including any + or #
    ply: int                   # zero-based ply within its game
    game: int                  # zero-based game ordinal within the row
    complete: bool             # the SAN is followed by a character inside the row
    legal_sans: list | None = field(default=None, repr=False)  # None after a replay failure


@dataclass
class RowAnnotation:
    klass: np.ndarray          # [len(text) - 1] int8 class code per target, -1 if unclassified
    ply: np.ndarray            # [len(text) - 1] int16 ply of the move the target belongs to
    game: np.ndarray           # [len(text) - 1] int16 game ordinal
    moves: list


def annotate_row(text, *, legal_moves=True):
    """Classify every target of one row and replay its games with python-chess."""
    length = len(text)
    klass = np.full(length, UNKNOWN, dtype=np.int8)     # indexed by row character
    ply_of = np.full(length, UNKNOWN, dtype=np.int16)
    game_of = np.full(length, UNKNOWN, dtype=np.int16)
    moves = []
    starts = [index for index, char in enumerate(text) if char == ';']
    for game, start in enumerate(starts):
        stop = starts[game + 1] if game + 1 < len(starts) else length
        # The next game's ``;`` ends this game, so it belongs to this one.
        game_of[start + 1:min(stop + 1, length)] = game
        board = chess.Board() if legal_moves else None
        ply = 0
        position = start + 1
        for token in text[start + 1:stop].split(' '):
            token_start, token_stop = position, position + len(token)
            position = token_stop + 1
            if not token:
                continue
            number = _MOVE_NUMBER.match(token)
            offset = number.end() if number else 0
            klass[token_start:token_start + offset] = CLASS_CODES['move_number']
            ply_of[token_start:token_stop + 1] = ply
            san = token[offset:]
            if not san:
                continue
            core = san.rstrip('+#')
            san_start = token_start + offset
            klass[san_start] = CLASS_CODES['move_first']
            klass[san_start + 1:san_start + len(core)] = CLASS_CODES['move_body']
            slot = san_start + len(core)
            if slot < length:
                klass[slot] = CLASS_CODES['check_slot']
            if len(core) < len(san) and token_stop < length:
                klass[token_stop] = CLASS_CODES['delimiter']
            complete = token_stop < length
            move = Move(san_start, san, ply, game, complete)
            if board is not None:
                move.legal_sans = sorted(board.san(candidate) for candidate in board.legal_moves)
                if complete:
                    try:
                        parsed = board.parse_san(san)
                    except ValueError:
                        board = None
                    else:
                        board.push(parsed)
            moves.append(move)
            ply += 1
    return RowAnnotation(klass[1:], ply_of[1:], game_of[1:], moves)
