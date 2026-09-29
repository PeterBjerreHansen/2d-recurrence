"""The engine teacher: a value for every legal move, with the game history visible.

Protocol (docs/engine_policy_plan.md, Engine values):

- one search per legal move, of the position after it, with a fixed node budget,
  so every move gets the same work;
- the position is sent with the moves played, so the engine sees repetitions
  and the 50-move count;
- a new game before every search (clearing the hash), one thread, a fixed hash
  size and no tablebases, so labels don't depend on annotation order;
- scores are read from the mover's perspective (``PovScore.pov(mover)``), and so
  are score bounds: the engine's lower bound for the opponent is an upper bound
  for the mover.

Rule outcomes are labelled without searching: a move that mates is mate in 1,
and a move that ends the game in a draw by rule gets 0.5. If the opponent could
claim a draw after the move (threefold repetition or the 50-move rule), the
move is worth at most 0.5, since the opponent would claim when winning is out
of reach.
"""

import hashlib
import importlib.metadata

import chess
import chess.engine
import numpy as np

from moves.vocab import MOVE_TO_ID
from moves.values import mate_after_move, q_from_centipawns, q_from_mate

SEARCH, CHECKMATE, RULE_DRAW = 0, 1, 2
EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2
LABEL_DTYPE = np.dtype([('q', '<f4'), ('cp', '<i4'), ('mate', '<i2'), ('kind', 'u1'), ('bound', 'u1'),
                        ('claimable', 'u1'), ('depth', '<u2'), ('nodes', '<u4')])


def history_key(board):
    """Cache key for a position's labels: its FEN and every position since the last irreversible ply.

    Earlier history cannot affect repetitions or the 50-move count, so two
    boards with the same key get the same labels from a deterministic engine.
    """
    board = board.copy()
    key = [board.fen()]
    while board.move_stack:
        move = board.pop()
        if board.is_irreversible(move):
            break
        key.append(board.epd())
    return tuple(key)


class Teacher:
    """Labels positions with one engine at one node budget."""

    def __init__(self, engine, nodes):
        if type(nodes) is not int or nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.engine = engine
        self.nodes = nodes

    @classmethod
    def open(cls, path, nodes, hash_mb=16):
        engine = chess.engine.SimpleEngine.popen_uci(path)
        engine.configure({'Threads': 1, 'Hash': hash_mb})
        return cls(engine, nodes)

    def close(self):
        self.engine.quit()

    def label(self, board):
        """Sorted legal move ids of ``board`` and one ``LABEL_DTYPE`` record per move.

        ``board`` must carry the game's move stack; it is returned unchanged.
        """
        mover = board.turn
        moves = sorted(board.legal_moves, key=lambda move: MOVE_TO_ID[move.uci()])
        labels = np.zeros(len(moves), LABEL_DTYPE)
        for index, move in enumerate(moves):
            board.push(move)
            try:
                labels[index] = self._label_after(board, mover)
            finally:
                board.pop()
        return [MOVE_TO_ID[move.uci()] for move in moves], labels

    def _label_after(self, board, mover):
        if board.is_checkmate():
            return q_from_mate(1), 0, 1, CHECKMATE, EXACT, 0, 0, 0
        if board.is_game_over():
            return 0.5, 0, 0, RULE_DRAW, EXACT, 0, 0, 0
        # A new game object makes python-chess send ucinewgame, which clears the hash.
        info = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), game=object())
        score = info['score'].pov(mover)
        mate = mate_after_move(score)
        if mate is None:
            cp = score.score()
            q = q_from_centipawns(cp)
        else:
            cp, q = 0, q_from_mate(mate)
        # The engine reports bounds for the side to move there, the opponent; flip them.
        bound = (UPPER_BOUND if info.get('lowerbound') else LOWER_BOUND if info.get('upperbound') else EXACT)
        claimable = board.can_claim_draw()
        if claimable:
            q = min(q, 0.5)
        return q, cp, mate or 0, SEARCH, bound, claimable, info.get('depth', 0), info.get('nodes', 0)


def engine_metadata(path, teacher, hash_mb):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(name=teacher.engine.id.get('name'), path=str(path), sha256=digest.hexdigest(),
                threads=1, hash_mb=hash_mb, tablebases=None, nodes=teacher.nodes,
                search='one search per legal move, new game before each',
                python_chess=importlib.metadata.version('chess'))
