"""Move datasets: packing, reading, both builders, the policy loss, and training on them."""

import json
import os
import random

import chess
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from moves.build_leela import build as build_leela, convert_archive
from moves.build_stage1 import build as build_stage1
from moves.games import random_game
from moves.leela import convert_game
from moves.objectives import PolicyTargets
from moves.rows import GameRecord, RowData, game_hash, pack, write_dataset_manifest
from moves.vocab import GAME_START, MOVE_COUNT, PAD, id_move, legal_ids
from tests.test_leela import _game_records, _random_moves, write_archive

CONTEXT = 64


def _legal_record(source, seed, plies):
    game, sets = random_game(random.Random(seed), plies)
    return GameRecord(source, np.asarray(game, dtype=np.uint16), np.array([len(s) for s in sets], dtype=np.uint8),
                      np.concatenate(sets).astype(np.uint16))


def _policy_record(seed, plies):
    record = _legal_record(0, seed, plies)
    rng = np.random.default_rng(seed)
    weights = []
    for count in record.counts:
        w = rng.random(count) ** 4
        weights.append(w / w.sum())
    record.weights = np.concatenate(weights).astype(np.float16)
    record.values = rng.uniform(-1, 1, (len(record.plies), 2)).astype(np.float16)
    record.visits = rng.integers(50, 800, len(record.plies)).astype(np.uint16)
    return record


def _write(directory, split_games, *, sources=('human', 'random'), policy=False):
    splits = {split: pack(directory, split, games, context=CONTEXT, sources=sources, policy=policy,
                          one_game_per_row=split != 'train') for split, games in split_games.items()}
    write_dataset_manifest(directory, context=CONTEXT, sources=sources, targets='policy' if policy else 'legal',
                           splits=splits, eval_splits=[s for s in split_games if s != 'train'])
    return RowData(directory, CONTEXT)


def _games_in_row(tokens):
    games = []
    for index, token in enumerate(tokens):
        if token == GAME_START:
            games.append((index, []))
        elif token != PAD:
            games[-1][1].append(int(token))
    return games


# --- packing and reading ---

def test_rows_hold_whole_games_and_their_move_lists(tmp_path):
    records = [_legal_record(i % 2, seed=i, plies=random.Random(i).randint(5, 40)) for i in range(12)]
    data = _write(tmp_path, {'train': records, 'dev': records[:3]})
    total = data.rows('train')
    batch = data.build('train', np.arange(total), objective='legal')
    tokens = np.asarray(data._split('train')['tokens'])
    placed = [plies for row in tokens for _, plies in _games_in_row(row)]
    assert placed == [r.plies.tolist() for r in records]           # stored order, no game split or lost
    boards, time = {}, CONTEXT
    for row_index, row in enumerate(tokens):
        board = None
        for t in range(CONTEXT):
            if row[t] == GAME_START:
                board = chess.Board()
            elif row[t] != PAD:
                board.push(id_move(int(row[t])))
            if row[t + 1] < MOVE_COUNT:
                boards[row_index * time + t] = board.copy()
    targets = batch.targets
    assert sorted(boards) == targets.positions.tolist()
    for owner, position in enumerate(targets.positions.tolist()):
        assert targets.moves[targets.owner == owner].tolist() == legal_ids(boards[position])
    assert batch.counts['human_plies'] == sum(len(r.plies) for r in records if r.source == 0)
    assert batch.counts['random_plies'] == sum(len(r.plies) for r in records if r.source == 1)
    assert batch.counts['supervised_positions'] == sum(len(r.plies) for r in records)
    human = data.build('train', np.arange(total), objective='human')
    assert int((human.targets >= 0).sum()) == targets.count
    dev = np.asarray(data._split('dev')['tokens'])
    assert all((row == GAME_START).sum() == 1 and row[0] == GAME_START for row in dev)


def test_policy_rows_keep_renormalised_weights(tmp_path):
    records = [_policy_record(seed, plies=20) for seed in range(4)]
    data = _write(tmp_path, {'train': records, 'dev': records[:1]}, sources=('leela',), policy=True)
    rows = np.arange(data.rows('train'))
    raw = data.build('train', rows, objective='engine').targets
    expected = []
    for record in records:
        weights, start = record.weights.astype(np.float32), 0
        for count in record.counts.astype(int):
            expected.append(weights[start:start + count] / weights[start:start + count].sum())
            start += count
    np.testing.assert_allclose(raw.weights.numpy(), np.concatenate(expected), rtol=1e-6)   # stored, renormalised
    with pytest.raises(ValueError, match="can't train"):
        data.build('train', rows, objective='legal')


def test_training_rows_read_stored_order_then_fixed_permutations(tmp_path):
    records = [_legal_record(0, seed=i, plies=60) for i in range(10)]
    data = _write(tmp_path, {'train': records, 'dev': records[:1]})
    total = data.rows('train')
    assert data.training_rows(0, total).tolist() == list(range(total))
    second = data.training_rows(total, total)
    assert sorted(second.tolist()) == list(range(total)) and second.tolist() != list(range(total))
    assert data.training_rows(total - 1, 2).tolist() == [total - 1, second[0]]


def test_the_policy_loss_is_cross_entropy_of_a_softmax_over_legal_moves():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 9, requires_grad=True)
    weights = torch.tensor([0.7, 0.2, 0.1, 0.5, 0.5])
    targets = PolicyTargets(positions=torch.tensor([0, 2]), owner=torch.tensor([0, 0, 0, 1, 1]),
                            moves=torch.tensor([1, 4, 7, 0, 2]), weights=weights)
    z = logits.reshape(-1, 9)
    expected = (-(weights[:3] * F.log_softmax(z[0, [1, 4, 7]], -1)).sum()
                - (weights[3:] * F.log_softmax(z[2, [0, 2]], -1)).sum()) / 2
    loss = targets.loss(logits)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    grad = logits.grad.reshape(-1, 9)
    assert grad[1].abs().sum() == 0 and grad[0, [0, 2, 3, 5, 6, 8]].abs().sum() == 0
    with torch.no_grad():
        logits.zero_()
        logits[0, 0, 1] = 3.0          # agrees with the teacher's top move (1)
        logits[0, 2, 2] = 3.0          # teacher tie 0/2: the lowest id (0) is its top move, so no agreement
    assert targets.metrics(logits)['top_move'] == 1


# --- the stage-1 builder ---

@pytest.fixture
def stage1(prepared_data, tmp_path):
    return build_stage1(prepared_data, tmp_path / 'stage1', positions=2000, random_fraction=0.25, eval_games=4,
                        context=CONTEXT, seed=3, test_per_mille=200, dev_per_mille=200), tmp_path / 'stage1'


def test_stage1_mixes_sources_and_lists_every_legal_move(stage1):
    manifest, directory = stage1
    data = RowData(directory, CONTEXT)
    build = manifest['build']
    assert build['human_training_positions'] >= 1500 and build['random_training_positions'] >= 500
    batch = data.build('train', np.arange(data.rows('train')), objective='legal')
    assert batch.counts['human_plies'] == build['human_training_positions']
    assert batch.counts['random_plies'] == build['random_training_positions']
    tokens = np.asarray(data._split('train')['tokens'])
    for owner, position in enumerate(batch.targets.positions.tolist()[:400]):
        row, t = divmod(position, CONTEXT)
        start = max(i for i in range(t + 1) if tokens[row, i] == GAME_START)
        board = chess.Board()
        for token in tokens[row, start + 1:t + 1]:
            board.push(id_move(int(token)))
        assert batch.targets.moves[batch.targets.owner == owner].tolist() == legal_ids(board)
    assert set(manifest['eval_splits']) == {'human_dev', 'random_dev'}
    for split in ('human_dev', 'human_test', 'random_dev', 'random_test'):
        assert 0 < manifest['splits'][split]['games'] <= 4


def test_stage1_evaluation_games_never_occur_in_training(stage1):
    _, directory = stage1
    data = RowData(directory, CONTEXT)

    def games(split):
        return {game_hash(p) for row in np.asarray(data._split(split)['tokens']) for _, p in _games_in_row(row)}

    train = games('train')
    for split in ('human_dev', 'human_test', 'random_dev', 'random_test'):
        assert not games(split) & train


def test_stage1_builds_identically_with_any_number_of_workers(prepared_data, tmp_path, stage1):
    manifest, _ = stage1
    again = build_stage1(prepared_data, tmp_path / 'again', positions=2000, random_fraction=0.25, eval_games=4,
                         context=CONTEXT, seed=3, test_per_mille=200, dev_per_mille=200, workers=2)
    for split, info in manifest['splits'].items():
        assert again['splits'][split]['sha256'] == info['sha256']


# --- the Leela builder ---

def _long_moves():
    for seed in range(200):
        moves = _random_moves(seed, plies=400)
        if len(moves) > 255:
            return moves
    raise AssertionError('no random game longer than 255 plies')


@pytest.fixture
def leela(tmp_path):
    games = [_game_records(_random_moves(seed, plies=random.Random(seed).randint(20, 60)), seed) for seed in range(24)]
    games.append(_game_records(_long_moves(), 99))
    chess960 = _game_records(_random_moves(500, plies=12), 500).copy()
    chess960['planes'] = np.roll(chess960['planes'], 1, axis=0)   # not the standard start position
    archives = [tmp_path / 'a.tar', tmp_path / 'b.tar']
    write_archive(archives[0], games[:13] + [chess960])
    write_archive(archives[1], games[13:])
    return games, archives


def test_leela_rows_carry_the_converted_games_and_their_probabilities(leela, tmp_path):
    games, archives = leela
    manifest = build_leela(archives, tmp_path / 'leela', positions=500, eval_games=3, context=256, seed=1,
                           test_per_mille=200, dev_per_mille=200)
    build = manifest['build']
    assert build['games_skipped'] == {'does not start at the standard start position': 1}
    assert build['games_truncated'] == 1 and build['truncated_at_plies'] == 255
    assert not (tmp_path / 'leela' / 'stores').exists()
    data = RowData(tmp_path / 'leela', 256)
    converted = {game_hash(g.plies[:255]): g for g in (convert_game(r) for r in games)}
    rows = np.arange(data.rows('train'))
    targets = data.build('train', rows, objective='engine').targets
    tokens = np.asarray(data._split('train')['tokens'])
    owner = 0
    for row in tokens:
        for start, plies in _games_in_row(row):
            game = converted[game_hash(np.asarray(plies, dtype=np.uint16))]
            for ply in range(len(plies)):
                select = targets.owner == owner
                assert targets.moves[select].tolist() == game.legal[ply].tolist()
                np.testing.assert_allclose(targets.weights[select].numpy(), game.probabilities[ply], atol=2e-3)
                owner += 1
    assert owner == targets.count >= 500
    assert manifest['splits']['train']['max_plies'] <= 255


def test_leela_archives_convert_in_parallel_workers_identically(leela, tmp_path):
    _, archives = leela
    one = build_leela(archives, tmp_path / 'one', positions=300, eval_games=2, seed=1, test_per_mille=200,
                      dev_per_mille=200)
    two = build_leela(archives, tmp_path / 'two', positions=300, eval_games=2, seed=1, test_per_mille=200,
                      dev_per_mille=200, workers=2)
    for split, info in one['splits'].items():
        assert two['splits'][split]['sha256'] == info['sha256']


def test_an_archive_with_empty_splits_still_builds(leela, tmp_path):
    games, archives = leela
    skipped = _game_records(_random_moves(500, plies=12), 500).copy()
    skipped['planes'] = np.roll(skipped['planes'], 1, axis=0)
    write_archive(tmp_path / 'c.tar', [skipped])
    manifest = build_leela(archives + [tmp_path / 'c.tar'], tmp_path / 'leela', positions=300, eval_games=2,
                           seed=1, test_per_mille=200, dev_per_mille=200)
    assert manifest['build']['games_skipped'] == {'does not start at the standard start position': 2}


ARCHIVE = os.environ.get('LEELA_ARCHIVE')


@pytest.mark.skipif(not ARCHIVE, reason='Set LEELA_ARCHIVE to a downloaded Leela training archive')
def test_a_real_archive_converts_with_its_statistics(tmp_path):
    stats = convert_archive((ARCHIVE, str(tmp_path / 'stores'), (5, 5), 255))
    games = sum(stats['games'].values())
    assert games > 40000
    assert set(stats['skipped']) <= {'does not start at the standard start position'}
    assert sum(stats['skipped'].values()) < 0.06 * (games + sum(stats['skipped'].values()))


# --- training on rows ---

def _config(dataset, output, objective, **overrides):
    config = dict(data_format='moves', objective=objective, dataset=str(dataset), n_layer=2, n_head=2, n_embd=16,
                  block_size=CONTEXT, batch_size=2, gradient_accumulation_steps=2, max_iters=4, eval_interval=2,
                  eval_iters=1, log_interval=1, warmup_iters=0, lr_decay_iters=4, compile=False, device='cpu',
                  dtype='float32', num_threads=1, out_dir=str(output))
    config.update(overrides)
    return config


def _recurrent(mode, **overrides):
    from importlib import import_module
    base = import_module('tests.test_recurrence_modes').mode_training_config(mode, 'unused')
    keys = ('architecture', 'recurrence_mode', 'update_support', 'update_probabilities', 'recurrence_seed',
            'n_layer', 'n_prelude', 'n_buffer', 'n_core', 'n_source', 'n_coda', 'eval_u_t', 'eval_u_d')
    return {**{key: base[key] for key in keys}, **overrides}


def _records(output):
    return [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('objective, architecture', [('legal', {}), ('human', _recurrent('temporal')),
                                                     ('legal', _recurrent('hybrid'))])
def test_stage1_rows_train_with_counters_and_live_evaluation(stage1, tmp_path, objective, architecture):
    from train import train
    _, directory = stage1
    output = tmp_path / 'run'
    train(_config(directory, output, objective, **architecture))
    records = _records(output)
    evaluation = [r for r in records if r['event'] == 'evaluation'][-1]
    assert {'train_loss', 'human_dev_loss', 'random_dev_loss'} <= set(evaluation)
    mode = architecture.get('recurrence_mode')
    for depth in ([] if mode is None else [1] if mode == 'temporal' else [1, 2, 4]):
        assert f'live_J{depth}_human_dev_loss' in evaluation
    last = [r for r in records if r['event'] == 'train'][-1]
    assert last['row_tokens'] == 4 * 2 * 2 * CONTEXT
    assert last['human_plies'] + last['random_plies'] == last['supervised_positions']


def test_row_training_resumes_exactly(stage1, tmp_path):
    from train import train
    _, directory = stage1
    config = _config(directory, tmp_path / 'full', 'legal', **_recurrent('temporal'))
    full = train(config)
    part = {**config, 'out_dir': str(tmp_path / 'part')}
    train({**part, 'max_iters': 2})
    resumed = train({**part, 'init_from': 'resume'})
    a, b = (torch.load(path, weights_only=False) for path in (full, resumed))
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['move_counts'] == b['move_counts']


def test_stage2_continues_a_stage1_trunk_on_leela_rows(stage1, leela, tmp_path):
    from train import train
    _, directory = stage1
    _, archives = leela
    build_leela(archives, tmp_path / 'leela', positions=300, eval_games=2, context=CONTEXT,
                seed=1, test_per_mille=200, dev_per_mille=200)
    legal = train(_config(directory, tmp_path / 'legal', 'legal', max_iters=2, **_recurrent('hybrid')))
    engine = train(_config(tmp_path / 'leela', tmp_path / 'engine', 'engine', max_iters=2, 
                           init_from='continue', continue_from=str(legal),
                           **_recurrent('hybrid')))
    evaluation = [r for r in _records(tmp_path / 'engine') if r['event'] == 'evaluation'][-1]
    assert {'leela_dev_kl', 'leela_dev_top_move', 'live_J4_leela_dev_top_move'} <= set(evaluation)
    last = [r for r in _records(tmp_path / 'engine') if r['event'] == 'train'][-1]
    assert last['leela_plies'] == last['supervised_positions'] > 0
    assert torch.load(engine, weights_only=False)['continued_from']['objective'] == 'legal'


def test_a_continued_run_resumes_exactly(stage1, leela, tmp_path):
    from train import train
    _, directory = stage1
    _, archives = leela
    build_leela(archives, tmp_path / 'leela', positions=300, eval_games=2, context=CONTEXT, seed=1,
                test_per_mille=200, dev_per_mille=200)
    legal = train(_config(directory, tmp_path / 'legal', 'legal', max_iters=2, **_recurrent('temporal')))
    config = _config(tmp_path / 'leela', tmp_path / 'full', 'engine', init_from='continue',
                     continue_from=str(legal), **_recurrent('temporal'))
    full = train(config)
    part = {**config, 'out_dir': str(tmp_path / 'part')}
    train({**part, 'max_iters': 2})
    resumed = train({**part, 'init_from': 'resume'})
    a, b = (torch.load(path, weights_only=False) for path in (full, resumed))
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert b['continued_from'] == a['continued_from']
    with pytest.raises(ValueError, match='continue_from'):
        train({**part, 'init_from': 'resume', 'continue_from': str(tmp_path / 'other.pt')})


def test_continuing_copies_the_trunk_and_reinitialises_the_readout(stage1, leela, tmp_path):
    from train import train
    _, directory = stage1
    _, archives = leela
    build_leela(archives, tmp_path / 'leela', positions=300, eval_games=2, context=CONTEXT, seed=1,
                test_per_mille=200, dev_per_mille=200)
    legal = train(_config(directory, tmp_path / 'legal', 'legal', max_iters=2))
    engine = train(_config(tmp_path / 'leela', tmp_path / 'engine', 'engine', max_iters=0,
                           init_from='continue', continue_from=str(legal)))
    source, result = (torch.load(path, weights_only=False) for path in (legal, engine))
    for key, value in source['model'].items():
        if key == 'lm_head.weight':
            assert not torch.equal(value, result['model'][key])
        else:
            torch.testing.assert_close(value, result['model'][key], rtol=0, atol=0)


@pytest.mark.parametrize('objective', ['human', 'legal'])
def test_live_evaluation_is_paired_with_the_training_graph(stage1, tmp_path, objective):
    # A depth-only model with depth_specialized caches runs live exactly as its training cell (0, J-1).
    from train import train
    _, directory = stage1
    output = tmp_path / 'run'
    train(_config(directory, output, objective, max_iters=2, live_eval_depths=[2],
                  **_recurrent('depth', eval_u_d=1)))
    evaluation = [r for r in _records(output) if r['event'] == 'evaluation'][-1]
    assert evaluation['live_J2_human_dev_loss'] == pytest.approx(evaluation['human_dev_loss'], rel=1e-4)
    key = 'accuracy' if objective == 'human' else 'exact_set'
    assert evaluation[f'live_J2_human_dev_{key}'] == pytest.approx(evaluation[f'human_dev_{key}'])


@pytest.mark.parametrize('overrides, message', [
    (dict(objective='engine'), "can't train"),
    (dict(objective='cheating'), 'objective must be one of'),
    (dict(init_from='continue'), 'continue_from'),
    (dict(continue_from='elsewhere.pt'), 'continue_from'),
    (dict(data_format='characters'), 'need move data'),
    (dict(live_eval_depths=[1]), 'transformer runs live exactly'),
])
def test_invalid_row_configurations_are_refused(stage1, tmp_path, overrides, message):
    from train import train
    _, directory = stage1
    with pytest.raises(ValueError, match=message):
        train({**_config(directory, tmp_path / 'run', 'legal'), **overrides})
