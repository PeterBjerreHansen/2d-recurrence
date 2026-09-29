"""Engine scores to move values Q, always from the perspective of the player making the move.

Non-mate scores use the Lichess conversion ``Q = sigmoid(0.00368208 * cp)``
with centipawns capped at +-1,500, so they lie in [0.004, 0.996]. Mates lie
outside that range and are strictly ordered by distance: delivering mate in
``n`` of the mover's moves is ``0.997 + 0.002 / n``, being mated in ``n`` is
``0.003 - 0.002 / n``.
"""

import math

CENTIPAWN_SCALE = 0.00368208
CENTIPAWN_CAP = 1500


def q_from_centipawns(cp):
    cp = max(-CENTIPAWN_CAP, min(CENTIPAWN_CAP, cp))
    return 1 / (1 + math.exp(-CENTIPAWN_SCALE * cp))


def q_from_mate(n):
    """``n > 0``: the mover mates in ``n`` of its moves; ``n < 0``: the mover is mated in ``-n``."""
    if type(n) is not int or n == 0:
        raise ValueError('Mate distance must be a nonzero integer')
    return 0.997 + 0.002 / n if n > 0 else 0.003 + 0.002 / n


def mate_after_move(score):
    """Mate distance of a move, from the engine's mate score of the position after it.

    ``score`` is that position's score seen from the mover (``PovScore.pov(mover)``);
    the opponent moves first there. A positive ``m`` means the mover mates in ``m``
    more moves, so the move mates in ``m + 1``; a negative ``m`` means the mover is
    mated in ``-m``.
    """
    mate = score.mate()
    if mate is None:
        return None
    if mate == 0:
        raise ValueError('A mated position must be handled before asking the engine')
    return mate + 1 if mate > 0 else mate
