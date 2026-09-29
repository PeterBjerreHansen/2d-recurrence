import random

import chess
from datasets import Dataset
import pytest

from data.chess_v1.prepare import prepare


@pytest.fixture
def prepared_data(tmp_path):
    rng = random.Random(7)
    rows = []
    for _ in range(30):
        text = ''
        while len(text) < 1024:
            board = chess.Board()
            text += ';'
            for _ in range(30):
                if board.is_game_over():
                    break
                move = rng.choice(list(board.legal_moves))
                if board.turn:
                    text += f'{board.fullmove_number}.'
                text += board.san(move) + ' '
                board.push(move)
        rows.append(text[:1024])
    directory = tmp_path / 'data'
    prepare(Dataset.from_dict({'transcript': rows}), directory,
            source={'fixture': 'synthetic legal random chess'}, val_fraction=0.2)
    return directory


@pytest.fixture
def move_data(prepared_data, tmp_path):
    """The character fixture converted to move tokens, with large dev and test splits."""
    from moves.prepare import prepare as prepare_moves
    directory = tmp_path / 'moves'
    prepare_moves(prepared_data, directory, test_per_mille=200, dev_per_mille=200)
    return directory


@pytest.fixture
def engine_data(move_data, tmp_path):
    """Games of ``move_data`` labelled by the material stand-in engine at two budgets."""
    from moves.annotate import annotate
    from tests.move_fakes import MaterialTeachers
    directory = tmp_path / 'engine'
    annotate(move_data, directory, MaterialTeachers(), {'train': 20, 'dev': 6}, seed=3)
    return directory
