from evaluation.pgn_annotations import CLASS_CODES, CLASSES, annotate_row


def _classes(annotation):
    return [CLASSES[code] if code >= 0 else None for code in annotation.klass]


def test_classes_plies_and_games_follow_the_row_text():
    text = ';1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6 4.Qxf7#;1.d4 d5'
    annotation = annotate_row(text)
    classes = _classes(annotation)
    target = lambda index: index - 1  # target t predicts character t + 1
    assert classes[target(1)] == 'move_number' and classes[target(2)] == 'move_number'
    assert classes[target(3)] == 'move_first'           # e of e4
    assert classes[target(4)] == 'move_body'            # 4
    assert classes[target(5)] == 'check_slot'           # space after e4
    mate = text.index('#')
    assert classes[target(mate)] == 'check_slot'
    assert classes[target(mate + 1)] == 'delimiter'     # ; after Qxf7#
    assert annotation.game[target(mate + 1)] == 0       # that ; still belongs to game 0
    assert annotation.game[target(mate + 2)] == 1
    assert annotation.ply[target(mate)] == 6
    assert [move.san for move in annotation.moves] == [
        'e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6', 'Qxf7#', 'd4', 'd5']
    assert [move.ply for move in annotation.moves][-3:] == [6, 0, 1]
    assert annotation.moves[-1].complete is False
    assert all(move.legal_sans is not None for move in annotation.moves)
    first = annotation.moves[0]
    assert len(first.legal_sans) == 20 and 'e4' in first.legal_sans
    assert 'Qxf7#' in annotation.moves[6].legal_sans


def test_replay_failure_stops_legal_annotations_for_that_game_only():
    annotation = annotate_row(';1.e4 e5 2.Ke3 Nc6;1.e4')
    assert annotation.moves[2].legal_sans is not None   # board before the illegal move is known
    assert annotation.moves[3].legal_sans is None
    assert annotation.moves[4].legal_sans is not None
    assert CLASS_CODES['move_first'] in set(annotation.klass.tolist())
