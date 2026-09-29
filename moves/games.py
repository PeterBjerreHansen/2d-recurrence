"""Games as ply-token sequences: parsing character rows, replay and random games."""

import re

import chess

from moves.vocab import id_move, legal_ids, move_id

MAX_PLIES = 255
_MOVE_NUMBER = re.compile(r'^\d+\.')


def parse_row(text):
    """Games in one character row, as lists of ply ids, plus the number of parse failures.

    Rows start at a game start (``;``) and may hold several games. A SAN token
    counts only when a delimiter follows it, so the ply cut off by the row end
    is dropped. A token that fails to parse ends its game there and is counted.
    """
    if not text.startswith(';'):
        raise ValueError('A row must start at a game start (;)')
    segments = text.split(';')[1:]
    games, failures = [], 0
    for index, segment in enumerate(segments):
        tokens = segment.split(' ')
        if index == len(segments) - 1:
            tokens = tokens[:-1]
        board, game = chess.Board(), []
        for token in tokens:
            san = _MOVE_NUMBER.sub('', token)
            if not san:
                continue
            try:
                move = board.parse_san(san)
            except ValueError:
                failures += 1
                break
            game.append(move_id(move))
            board.push(move)
        if game:
            games.append(game)
    return games, failures


def san_plies(text):
    """The complete SAN plies of each game in a row, as written (for round-trip checks)."""
    segments = text.split(';')[1:]
    games = []
    for index, segment in enumerate(segments):
        tokens = segment.split(' ')
        if index == len(segments) - 1:
            tokens = tokens[:-1]
        plies = [_MOVE_NUMBER.sub('', token) for token in tokens]
        games.append([ply for ply in plies if ply])
    return [game for game in games if game]


def to_san(game):
    """SAN of each ply of a game given as ply ids."""
    board, plies = chess.Board(), []
    for token in game:
        move = id_move(token)
        plies.append(board.san(move))
        board.push(move)
    return plies


def positions(game):
    """The board before each ply, then the final board (``len(game) + 1`` boards).

    Each yielded board is the same object, advanced in place; copy it to keep it.
    """
    board = chess.Board()
    yield board
    for token in game:
        board.push(id_move(token))
        yield board


def legal_sets(game):
    """Legal ply ids before each ply of ``game`` (one list per ply)."""
    sets = []
    for index, board in enumerate(positions(game)):
        if index == len(game):
            break
        sets.append(legal_ids(board))
    return sets


def random_game(rng, max_plies=MAX_PLIES):
    """Uniformly random legal plies until the game ends by rule or ``max_plies`` is reached.

    Returns the ply ids and the legal set before each ply, which the random
    choice needed anyway.
    """
    board, game, sets = chess.Board(), [], []
    while len(game) < max_plies and not board.is_game_over():
        legal = legal_ids(board)
        token = rng.choice(legal)
        sets.append(legal)
        game.append(token)
        board.push(id_move(token))
    return game, sets

