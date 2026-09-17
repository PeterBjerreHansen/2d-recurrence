"""Incremental SAN validation: stop at the first completed invalid move."""
import io
import re

import chess
import chess.pgn

RESULTS = {'1-0', '0-1', '1/2-1/2', '*'}


class QuietGameBuilder(chess.pgn.GameBuilder):
    def handle_error(self, error):
        self.game.errors.append(error)


class MoveValidator:
    """Accept characters without inspecting or filtering candidate model logits."""
    def __init__(self):
        self.board = chess.Board()
        self.pending = ''
        self.text = ''
        self.reason = None
        self.offending_move = None
        self.failure_fen = None
        self.legal_moves = 0
        self.attempted_moves = 0
        self.prompt_moves = 0
        self.declared_result = None

    def _fail(self, reason, candidate):
        self.reason = reason
        self.offending_move = candidate
        self.failure_fen = self.board.fen()

    def _complete(self, prompt=False):
        token, self.pending = self.pending, ''
        if not token:
            return
        # Karvonen transcripts attach the white move number to SAN: 1.e4.
        token = re.sub(r'^\d+\.(?:\.\.)?', '', token)
        if not token:
            return
        if token in RESULTS:
            self.declared_result = token
            actual = self.board.outcome(claim_draw=False)
            if token == '*':
                self.reason = 'unfinished_result'
            elif actual is not None and token != actual.result():
                self._fail('invalid_result', token)
            else:
                # Resignation/agreed draw need not be a terminal board position.
                self.reason = 'declared_result'
            return
        if not prompt:
            self.attempted_moves += 1
        try:
            move = self.board.parse_san(token)
        except chess.IllegalMoveError:
            self._fail('illegal_move', token)
            return
        except (chess.InvalidMoveError, chess.AmbiguousMoveError):
            self._fail('malformed_move', token)
            return
        if not self.board.is_legal(move):  # parse_san also accepts null moves; these are not legal chess moves.
            self._fail('illegal_move', token)
            return
        self.board.push(move)
        if prompt:
            self.prompt_moves += 1
        else:
            self.legal_moves += 1
        if self.board.is_game_over(claim_draw=False):
            self.reason = 'board_terminal'

    def feed(self, character, *, prompt=False):
        if self.reason:
            return
        at_start = not self.text
        self.text += character
        if character.isspace() or character == ';':
            self._complete(prompt=prompt)
            if character == ';' and not at_start and not self.reason:
                # Upstream omits results and separates games with ;. Report that distinctly.
                self.reason = 'game_boundary'
        else:
            self.pending += character

    def read_prompt(self, text):
        if not text.startswith(';'):
            raise ValueError('Evaluation prompts must start at a game boundary (;)')
        for char in text:
            self.feed(char, prompt=True)
            if self.reason:
                raise ValueError(f'Prompt ended or failed: {self.reason}')
        # Require prompts to end at whitespace or a bare move number, so completed
        # prompt moves never enter generated-move counts after the first sample.
        if self.pending and not re.fullmatch(r'\d+\.(?:\.\.)?', self.pending):
            raise ValueError('Prompt must end at a move boundary or bare move number, e.g. ;1.')

    def finish(self, reason='generation_limit'):
        if self.reason is None:
            self.reason = reason
        # Never turn a final incomplete character prefix into a completed illegal move.
        return self.report()

    def report(self):
        pgn_text = self.text[1:] if self.text.startswith(';') else self.text
        pgn_text = pgn_text.rstrip(';')
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text), Visitor=QuietGameBuilder)
            parse_success = game is not None and not game.errors
        except ValueError:
            parse_success = False
        # The PGN reader can ignore garbage; a stopped malformed move is not a parse success.
        if self.reason in {'illegal_move', 'malformed_move', 'invalid_result'}:
            parse_success = False
        return dict(text=self.text, termination_reason=self.reason,
                    offending_move=self.offending_move, board_before_failure=self.failure_fen,
                    final_board=self.board.fen(), legal_moves=self.legal_moves,
                    attempted_moves=self.attempted_moves, prompt_moves=self.prompt_moves,
                    incomplete_move=self.pending or None, declared_result=self.declared_result,
                    pgn_parse_success=parse_success,
                    valid_termination=self.reason in {'board_terminal', 'declared_result'})
