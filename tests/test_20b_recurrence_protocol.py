from importlib import import_module
import json
import sys

import pytest

from recurrence.schedule import update_probabilities_at_step


study = import_module('experiments.long_runs.20B_recurrence.study')
runner = import_module('experiments.long_runs.20B_recurrence.run')


def _max_update_marginals(matrix):
    support = study.UPDATE_SUPPORT
    return [sum(matrix[i][j] for i, temporal in enumerate(support)
                for j, depth in enumerate(support)
                if max(temporal, depth) == count)
            for count in support]


def test_20b_configs_pin_horizon_wsd_and_all_retained_checkpoints():
    assert study.CHARACTERS == 20_000_000_000
    assert study.UPDATES == 195_504
    assert study.ACTUAL_CHARACTERS == 20_000_059_200
    assert study.WARMUP_UPDATES == 2_000
    assert study.WSD_DECAY_START == 175_954
    assert study.CHECKPOINT_STEPS == [
        0, 9_776, 19_551, 39_101, 78_202, 97_752,
        136_853, 156_403, 175_954, 195_504,
    ]
    assert study.MAJOR_CHECKPOINT_STEPS == (9_776, 39_101, 97_752, 175_954, 195_504)
    assert study.EVALUATION_CHECKPOINT_STEPS == tuple(study.CHECKPOINT_STEPS[1:])


def test_study_command_evaluates_every_retained_checkpoint_after_each_arm(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'train_arm', lambda name: calls.append(('train', name)))
    monkeypatch.setattr(
        runner, 'evaluate_arm', lambda name, step: calls.append(('evaluate', name, step)))
    monkeypatch.setattr(sys, 'argv', [
        'run.py', 'study', '--evaluate-checkpoints'])

    runner.main()

    expected = []
    for name in study.ARM_ORDER:
        expected.append(('train', name))
        expected.extend(('evaluate', name, step) for step in study.EVALUATION_CHECKPOINT_STEPS)
    assert calls == expected


def test_frozen_source_includes_data_test_dependencies():
    paths = set(runner.source_snapshot()['files'])
    assert {
        'data/chess_v1/prepare.py',
        'data/chess_v1/meta.pkl',
        'docs/upstream.json',
    } <= paths


def test_source_snapshot_verifies_bundle_without_git_metadata(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('VALUE = 1\n')
    digest = runner.file_hash(source)
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps({'source': {
        'files': {'source.py': digest},
        'head_commit': 'base-commit',
        'branch': 'feature/test',
        'working_tree_patch_sha256': 'patch-hash',
    }}))
    transfer_path = tmp_path / 'TRANSFER_MANIFEST.json'
    transfer_path.write_text(json.dumps({
        'files': {'source.py': digest},
        'base_commit': 'base-commit',
        'branch': 'feature/test',
        'working_tree_patch_sha256': 'patch-hash',
    }))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'PROTOCOL', protocol_path)
    monkeypatch.setattr(runner, 'TRANSFER_MANIFEST', transfer_path)

    def no_git(*args, **kwargs):
        raise runner.subprocess.CalledProcessError(128, ['git'])

    monkeypatch.setattr(runner, '_git', no_git)

    snapshot = runner.source_snapshot()

    assert snapshot['head_commit'] == 'base-commit'
    assert snapshot['branch'] == 'feature/test'
    assert snapshot['working_tree_status'] == 'verified transfer bundle without Git metadata'
    assert snapshot['files'] == {'source.py': digest}


def test_four_arm_curriculum_matches_update_marginals_and_hybrid_diagonal():
    expected = [(.10, .80, .10), (.05, .60, .35), (0., .40, .60), (0., .20, .80)]
    names = ('temporal', 'depth', 'hybrid')
    for name in names:
        config = study.run_config(name)
        assert config['lr_schedule'] == 'wsd'
        assert config['lr_decay_start'] == 175_954
        assert config['warmup_iters'] == 2_000
        assert config['max_iters'] == 195_504
        assert config['eval_interval'] == 10_000
        assert config['temporal_memory_gate_init'] == .10
        assert config['checkpoint_interval'] == 1_000
        schedule = config['update_probability_schedule']
        assert [phase['start_step'] for phase in schedule['phases']] == [0, 9_776, 39_101, 97_752]
        for index, phase in enumerate(schedule['phases']):
            matrix = phase['update_probabilities']
            assert _max_update_marginals(matrix) == pytest.approx(expected[index])
            if name == 'hybrid':
                for count_index, count in enumerate(study.UPDATE_SUPPORT[1:], start=1):
                    bucket = expected[index][count_index]
                    if bucket:
                        diagonal = matrix[count_index][count_index]
                        assert diagonal / bucket == pytest.approx(.8)

        for phase_index, start in enumerate((0, 9_776, 39_101, 97_752)):
            assert update_probabilities_at_step(config, start) == tuple(
                tuple(row) for row in schedule['phases'][phase_index]['update_probabilities'])
            if start:
                assert update_probabilities_at_step(config, start - 1) == tuple(
                    tuple(row) for row in schedule['phases'][phase_index - 1]['update_probabilities'])


@pytest.mark.parametrize('name, expected_primary, expected_extended', [
    ('transformer', [(0, 0)], [(0, 0)]),
    ('temporal', [(0, 0), (3, 0)], [(0, 0), (1, 0), (3, 0), (7, 0), (15, 0)]),
    ('depth', [(0, 0), (0, 3)], [(0, 0), (0, 1), (0, 3), (0, 7), (0, 15)]),
    ('hybrid', [(0, 0), (3, 3)], [(0, 0), (1, 1), (3, 3), (7, 7), (15, 15)]),
])
def test_evaluation_cells_follow_plan_and_only_expand_at_major_checkpoints(
        name, expected_primary, expected_extended):
    assert study.evaluation_cells(name, 19_551) == expected_primary
    assert study.evaluation_cells(name, 175_954) == expected_extended


def test_runner_dry_run_checks_settled_protocol_without_writing_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path / 'results')
    monkeypatch.setattr(runner, 'PROTOCOL', tmp_path / 'results' / 'protocol.json')
    monkeypatch.setattr(runner, 'PANEL', tmp_path / 'results' / 'panel.json')

    report = runner.dry_run()

    assert report['status'] == 'passed'
    assert report['protocol_frozen'] is False
    assert report['temporal_memory_gate_init'] == .10
    assert [arm['name'] for arm in report['arms']] == list(study.ARM_ORDER)
    assert report['training']['actual_characters'] == study.ACTUAL_CHARACTERS
    assert report['training']['lr_trace']['175954'] == study.LEARNING_RATE
    assert report['training']['lr_trace']['195503'] == study.MIN_LEARNING_RATE
    assert report['training']['lr_trace']['195504'] == study.MIN_LEARNING_RATE
    assert not runner.PROTOCOL.exists()


def test_freeze_records_fixed_gate_and_is_repeatable(tmp_path, monkeypatch):
    results = tmp_path / 'results'
    monkeypatch.setattr(runner, 'ROOT', results)
    monkeypatch.setattr(runner, 'PROTOCOL', results / 'protocol.json')
    monkeypatch.setattr(runner, 'PANEL', results / 'panel.json')
    monkeypatch.setattr(runner, 'ENVIRONMENT', results / 'environment.json')

    receipt = runner.freeze()

    assert receipt['gate_initialization']['selected_value'] == .10
    assert receipt['training']['lr_schedule'] == 'wsd'
    assert receipt['evaluation']['extended_cells']['hybrid'][-1] == [15, 15]
    assert receipt['evaluation']['live']['settings']['temporal'] == [
        {'depth_steps': 1, 'kv_strategy': 'final_depth'}]
    assert runner.freeze() == receipt


@pytest.fixture
def frozen_checkpoint(monkeypatch):
    protocol = {
        'configurations': runner.configurations(),
        'dataset': {'manifest_hash': 'dataset'},
        'panel': {'sha256': 'panel'},
    }
    monkeypatch.setattr(runner, 'recorded_protocol', lambda: protocol)
    return dict(config=study.run_config('temporal'), iter_num=study.UPDATES,
                manifest_hash='dataset', eval_panel_sha256='panel')


def test_checkpoint_identity_rejects_wrong_arm_and_step(frozen_checkpoint):
    runner._validate_checkpoint(frozen_checkpoint, 'temporal', study.UPDATES)
    with pytest.raises(ValueError, match='configuration'):
        runner._validate_checkpoint(frozen_checkpoint, 'hybrid')
    with pytest.raises(ValueError, match='step'):
        runner._validate_checkpoint(frozen_checkpoint, 'temporal', 9_776)
    frozen_checkpoint['config']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='configuration'):
        runner._validate_checkpoint(frozen_checkpoint, 'temporal')


def test_train_rejects_misplaced_completed_checkpoint_and_unverified_resume(
        frozen_checkpoint, tmp_path, monkeypatch):
    import torch
    config = study.run_config('temporal')
    config['out_dir'] = str(tmp_path)
    frozen_checkpoint['config']['out_dir'] = str(tmp_path)
    protocol = runner.recorded_protocol()
    protocol['configurations']['temporal']['out_dir'] = str(tmp_path)
    protocol['configurations']['hybrid']['out_dir'] = str(tmp_path)
    monkeypatch.setattr(study, 'run_config', lambda name: config)
    monkeypatch.setattr(runner, 'preflight', lambda: None)
    monkeypatch.setattr(runner, '_write_arm_plan', lambda name: {})
    monkeypatch.setattr(runner, 'train', lambda config: pytest.fail('Training must not start'))
    checkpoint = tmp_path / 'ckpt.pt'
    torch.save(frozen_checkpoint, checkpoint)
    with pytest.raises(ValueError, match='configuration'):
        runner.train_arm('hybrid')
    assert runner.train_arm('temporal') == checkpoint
    frozen_checkpoint['iter_num'] = 500
    torch.save(frozen_checkpoint, checkpoint)
    with pytest.raises(RuntimeError, match='Automatic resume is disabled'):
        runner.train_arm('temporal')


def test_live_settings_run_only_at_major_checkpoints():
    assert study.live_evaluation_settings('hybrid', 19_551) == []
    assert study.live_evaluation_settings('transformer', study.UPDATES) == []
    assert study.live_evaluation_settings('temporal', 9_776) == [(1, 'final_depth')]
    assert study.live_evaluation_settings('hybrid', study.UPDATES) == [
        (1, 'depth_specialized'), (4, 'depth_specialized')]


def _tiny_model(mode):
    import torch
    from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
    torch.manual_seed(0)
    return Recurrent2DGPT(RecurrentGPTConfig(
        block_size=12, vocab_size=32, n_layer=8, n_head=2, n_embd=16,
        recurrence_mode=mode)).eval()


def _tiny_rows(count=2, length=12):
    import torch
    generator = torch.Generator().manual_seed(1)
    rows = torch.randint(32, (count, length + 1), generator=generator)
    return [(row[:-1], row[1:]) for row in rows]


def test_depth_live_nll_matches_its_training_graph_cell():
    import torch
    from recurrence.schedule import RecurrenceSchedule
    model = _tiny_model('depth')
    rows = _tiny_rows()
    [report] = runner.evaluate_live(model, 'depth', study.UPDATES, rows)
    inputs = torch.stack([inputs for inputs, _ in rows])
    targets = torch.stack([targets for _, targets in rows])
    schedule = RecurrenceSchedule((False,) * 3, (True,) * 3)
    with torch.no_grad():
        _, graph_nll = model(inputs, targets, schedule=schedule)
    assert report['execution'] == 'live'
    assert report['depth_steps'] == 4
    assert report['nll'] == pytest.approx(graph_nll.item(), abs=1e-5)


def test_live_evaluation_reports_each_declared_setting():
    rows = _tiny_rows(count=1)
    assert runner.evaluate_live(_tiny_model('hybrid'), 'hybrid', 19_551, rows) == []
    reports = runner.evaluate_live(_tiny_model('hybrid'), 'hybrid', study.UPDATES, rows)
    assert [(item['depth_steps'], item['kv_strategy']) for item in reports] == [
        (1, 'depth_specialized'), (4, 'depth_specialized')]
    [temporal] = runner.evaluate_live(_tiny_model('temporal'), 'temporal', study.UPDATES, rows)
    assert temporal['temporal_feedback_enabled'] is True
    assert temporal['target_count'] == 12
