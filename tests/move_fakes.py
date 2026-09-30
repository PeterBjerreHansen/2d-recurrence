"""A deterministic stand-in for Stockfish: material balance for the side to move."""

import chess
import chess.engine


PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}


class MaterialEngine:
    id = {'name': 'material'}

    def __init__(self, bound=None):
        self.calls = []
        self.bound = bound   # 'lowerbound' or 'upperbound' to report every score as a bound

    def analyse(self, board, limit, game=None):
        self.calls.append((board.fen(), len(board.move_stack), game))
        cp = sum(PIECE_VALUES.get(piece.piece_type, 0) * (1 if piece.color == board.turn else -1)
                 for piece in board.piece_map().values())
        info = {'score': chess.engine.PovScore(chess.engine.Cp(cp), board.turn), 'depth': 1, 'nodes': limit.nodes}
        if self.bound:
            info[self.bound] = True
        return info

    def quit(self):
        pass

