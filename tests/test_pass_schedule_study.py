from importlib import import_module
from pathlib import Path
import sys

import pytest

from recurrence.schedule import probabilities_at_step
from train import DEFAULTS


study = import_module('experiments.ablations.1B_pass_schedule.study')
runner = import_module('experiments.ablations.1B_pass_schedule.run')
common = import_module('experiments.ablations.1B_pass_schedule.configs.common')
SUPPORT = common.SUPPORT


def pass_distribution(matrix):
    return tuple(sum(matrix[i][j] for i, temporal in enumerate(SUPPORT)
                     for j, depth in enumerate(SUPPORT)
                     if max(temporal, depth) == count)
                 for count in SUPPORT)


def test_one_billion_study_horizon_and_crossover_are_frozen():
    assert study.UPDATES == 9776
    assert study.ACTUAL_CHARACTERS == 1_000_084_800
    assert study.WARMUP_UPDATES == 196
    assert study.CROSSOVER_STEP == 8310
    assert study.CHECKPOINT_STEPS == [0, 100, 250, 500, 1000, 2500, 5000, 7500,
                                     8310, 8311, 9000, 9776]


@pytest.mark.parametrize('name, mode, scheduled', [
    ('depth_fixed', 'depth', False),
    ('depth_hard_2to4', 'depth', True),
    ('temporal_fixed', 'temporal', False),
    ('temporal_hard_2to4', 'temporal', True),
    ('hybrid_fixed', 'hybrid', False),
    ('hybrid_hard_2to4', 'hybrid', True),
])
def test_one_billion_study_arms_resolve_expected_schedule(name, mode, scheduled):
    config = study.run_config(name)
    assert config['recurrence_mode'] == mode
    assert (config['recurrence_probability_schedule'] is not None) is scheduled
    if scheduled:
        assert probabilities_at_step(config, 8309) != probabilities_at_step(config, 8310)
        assert config['recurrence_probability_schedule']['phases'][1]['start_step'] == 8310
    else:
        assert probabilities_at_step(config, 0) == probabilities_at_step(config, 8310)


def test_temporal_only_control_uses_temporal_axis_and_expected_evaluation():
    config = study.run_config('temporal_fixed')
    assert config['recurrence_mode'] == 'temporal'
    assert (config['eval_u_t'], config['eval_u_d']) == (3, 0)
    assert pass_distribution(config['recurrence_probabilities']) == pytest.approx(
        study.FIXED_PASS_PROBABILITIES)
    assert all(row[1:] == (0.0, 0.0) for row in config['recurrence_probabilities'])


@pytest.mark.parametrize('name', ['depth_fixed', 'temporal_fixed', 'hybrid_fixed'])
def test_fixed_arms_match_pass_count_distribution(name):
    matrix = study.run_config(name)['recurrence_probabilities']
    assert pass_distribution(matrix) == pytest.approx(study.FIXED_PASS_PROBABILITIES)


@pytest.mark.parametrize('name', ['depth_hard_2to4', 'temporal_hard_2to4', 'hybrid_hard_2to4'])
def test_hard_arms_match_fixed_pass_count_distribution_over_horizon(name):
    schedule = study.run_config(name)['recurrence_probability_schedule']['phases']
    weights = (study.CROSSOVER_STEP / study.UPDATES,
               (study.UPDATES - study.CROSSOVER_STEP) / study.UPDATES)
    phase_distributions = [pass_distribution(phase['probabilities']) for phase in schedule]
    actual = tuple(sum(weight * distribution[index]
                       for weight, distribution in zip(weights, phase_distributions))
                   for index in range(len(SUPPORT)))
    assert actual == pytest.approx(study.FIXED_PASS_PROBABILITIES)


def test_source_snapshot_excludes_unrelated_experiment_receipts():
    paths = {str(path) for path in runner._source_paths()}
    assert 'experiments/long_runs/5B_axis/runpod_batch_sweep_20260920/batch_sweep/summary.json' not in paths
    assert 'train.py' in paths
    assert 'experiments/ablations/1B_pass_schedule/study.py' in paths
    assert 'docs/RECURRENCE_CONTRACT.md' in paths


def test_source_snapshot_status_is_scoped_to_study_runtime():
    status = runner.source_snapshot()['working_tree_status']
    assert 'runpod_batch_sweep_20260920' not in status


@pytest.mark.parametrize('name', sorted(study.RUNS))
def test_one_billion_config_entry_points_work_with_configurator(name, monkeypatch):
    namespace = dict(DEFAULTS)
    config = (Path('experiments/ablations/1B_pass_schedule/configs') / f'{name}.py')
    monkeypatch.setattr(sys, 'argv', ['configurator.py', str(config)])
    exec(Path('configurator.py').read_text(), namespace)
    assert namespace['architecture'] == 'recurrent'
    assert namespace['out_dir'].startswith('experiments/ablations/1B_pass_schedule/')
