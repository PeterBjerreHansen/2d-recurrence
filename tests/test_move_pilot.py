import json

import pytest
import torch

from experiments.move_pilot import pilot
from recurrence.schedule import update_probability_map_at_step
from tests.test_rows import CONTEXT, stage1  # noqa: F401 (fixture)


@pytest.fixture
def datasets(tmp_path, monkeypatch):
    for name, rows, positions in (('stage1', 2_400_000, 500_000_000), ('leela', 500_000, 100_000_000)):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'dataset.json').write_text(
            json.dumps(dict(splits=dict(train=dict(rows=rows, positions=positions)))))
    monkeypatch.setattr(pilot, 'STAGE1', tmp_path / 'stage1')
    monkeypatch.setattr(pilot, 'LEELA', tmp_path / 'leela')
    monkeypatch.setattr(pilot, 'ROOT', tmp_path / 'results')
    return tmp_path


def _max_count_mass(config, step):
    mass = {}
    for (u_t, u_d), probability in update_probability_map_at_step(config, step).items():
        mass[max(u_t, u_d)] = mass.get(max(u_t, u_d), 0) + probability
    return mass


def test_every_pilot_run_resolves(datasets):
    for name in pilot.run_names():
        config = pilot.run_config(name)
        assert config['data_format'] == 'moves' and config['block_size'] == 256
        assert config['batch_size'] * config['gradient_accumulation_steps'] == 400
        assert config['lr_decay_start'] < config['max_iters'] == config['lr_decay_iters']
    assert pilot.run_config('legal_hybrid')['max_iters'] == 6000
    assert pilot.run_config('engine_hybrid')['max_iters'] == 625
    assert pilot.run_config('lr_legal_1e-3')['max_iters'] == 1200
    check = pilot.run_config('lr_legal_depth_3e-4')
    assert check['max_iters'] == 1200 and check['recurrence_mode'] == 'depth' and check['learning_rate'] == 3e-4
    assert check['min_lr'] == pytest.approx(3e-5) and check['update_probability_schedule'] is not None
    seeds = {pilot.run_config(name)['seed'] for name in ('legal_hybrid', 'legal_hybrid_seed2')}
    assert len(seeds) == 2


def test_the_update_mixture_is_gap_free_broad_and_deepens(datasets):
    for mode in ('temporal', 'depth', 'hybrid'):
        config = pilot.run_config(f'legal_{mode}')
        assert config['update_support'] == [0, 1, 2, 3]
        centres = []
        for step in (0, 1500, 3000, config['max_iters']):
            mass = _max_count_mass(config, step)
            assert min(mass.values()) >= 0.10 and max(mass.values()) <= 0.35 + 1e-9
            centres.append(sum(count * p for count, p in mass.items()))
        assert centres == sorted(centres) and centres[0] < centres[-1]
        # Stage 2 keeps stage 1's final mixture throughout.
        engine = pilot.run_config(f'engine_{mode}')
        assert engine['update_probability_schedule'] is None
        assert _max_count_mass(engine, 0) == pytest.approx(_max_count_mass(config, config['max_iters']))


def test_stage2_continues_from_the_matching_stage1_run(datasets):
    config = pilot.run_config('engine_depth')
    assert config['init_from'] == 'continue' and config['objective'] == 'engine'
    assert config['continue_from'] == str(datasets / 'results' / 'legal_depth' / 'ckpt.pt')


def test_temporal_warm_starts_only_in_the_decay(datasets, monkeypatch):
    calls = []
    monkeypatch.setattr(pilot, 'train', lambda config: calls.append(config))
    pilot.run('legal_temporal')
    first, second = calls
    assert first['max_iters'] == first['lr_decay_start'] and first['warm_start_fraction'] == 0
    assert second['init_from'] == 'resume' and second['warm_start_fraction'] == 0.25
    # Resumed inside the decay, only the warm-started segment remains.
    calls.clear()
    checkpoint = datasets / 'results' / 'legal_temporal' / 'ckpt.pt'
    checkpoint.parent.mkdir(parents=True)
    torch.save(dict(iter_num=5900), checkpoint)
    pilot.run('legal_temporal')
    assert len(calls) == 1 and calls[0]['warm_start_fraction'] == 0.25
    calls.clear()
    pilot.run('legal_hybrid')
    assert len(calls) == 1 and calls[0]['warm_start_fraction'] == 0


@pytest.mark.parametrize('arm', ['temporal', 'hybrid'])
def test_pilot_configs_train_with_two_updates_in_the_support(stage1, tmp_path, arm):  # noqa: F811
    _, directory = stage1
    config = pilot._config(arm, 'legal', directory, 20, tmp_path / arm)
    config.update(block_size=CONTEXT, n_head=2, n_embd=16, batch_size=2, gradient_accumulation_steps=2,
                  eval_interval=10, eval_iters=1, log_interval=1, live_eval_batches=1, checkpoint_interval=0,
                  device='cpu', dtype='float32', num_threads=1)
    pilot.train(config)
    records = [json.loads(line) for line in (tmp_path / arm / 'metrics.jsonl').read_text().splitlines()]
    counts = {max(s['u_t'], s['u_d']) for r in records if r['event'] == 'train' for s in r['schedules']}
    assert counts == {0, 1, 2, 3}


def test_the_micro_batch_changes_speed_not_the_update_or_the_evaluation(datasets):
    small, large = pilot.run_config('legal_hybrid'), pilot.run_config('legal_hybrid', micro_batch=100)
    for config in (small, large):
        assert config['batch_size'] * config['gradient_accumulation_steps'] == 400
        assert config['eval_iters'] * config['batch_size'] == 1000
        assert config['live_eval_batches'] * config['batch_size'] == 100
    assert {key: value for key, value in small.items() if key not in ('batch_size', 'gradient_accumulation_steps',
                                                                     'eval_iters', 'live_eval_batches')} == \
        {key: value for key, value in large.items() if key not in ('batch_size', 'gradient_accumulation_steps',
                                                                   'eval_iters', 'live_eval_batches')}
    assert pilot.run_config('legal_temporal', micro_batch=50)['gradient_accumulation_steps'] == 8
    with pytest.raises(ValueError, match='quarter'):
        pilot.run_config('legal_temporal', micro_batch=80)
    with pytest.raises(ValueError, match='divide'):
        pilot.run_config('legal_depth', micro_batch=30)


def test_bench_times_the_deepest_mixture_without_live_evaluation(datasets, tmp_path):
    config = pilot.bench_config('legal_depth', 30, 50, tmp_path / 'bench')
    assert config['max_iters'] == 30 and config['live_eval_batches'] == 0 and config['batch_size'] == 50
    assert config['update_probability_schedule'] is None
    assert _max_count_mass(config, 0) == pytest.approx(dict(enumerate(pilot.CURRICULUM[-1][1])))
    directory = tmp_path / 'summary'
    directory.mkdir()
    (directory / 'metrics.jsonl').write_text('\n'.join(json.dumps(dict(event='train', step=step, seconds=seconds))
                                                       for step, seconds in enumerate([9, 9, 9, 9, 1, 3, 2], 1)))
    (directory / 'peak_memory.json').write_text(json.dumps(dict(peak_memory_gb=5.5)))
    assert pilot.bench_summary(directory) == dict(seconds_per_update=3, peak_memory_gb=5.5)


def test_the_queue_keeps_at_most_its_slots_running(datasets, monkeypatch):
    running, peak, commands = set(), [0], []

    class Process:
        def __init__(self, command, stdout, stderr):
            self.name, self.polls = command[4], 0
            commands.append(command)
            running.add(self.name)
            peak[0] = max(peak[0], len(running))

        def poll(self):
            self.polls += 1
            if self.polls < 2:
                return None
            running.discard(self.name)
            self.returncode = 0
            return 0

    monkeypatch.setattr(pilot.subprocess, 'Popen', Process)
    codes = pilot.queue(['legal_transformer', 'legal_temporal', 'legal_depth'], slots=2, threads=3, poll_seconds=0)
    assert codes == dict.fromkeys(['legal_transformer', 'legal_temporal', 'legal_depth'], 0) and peak[0] == 2
    assert commands[0][3:] == ['run', 'legal_transformer', '--micro-batch', '50', '--threads', '3']
    assert (datasets / 'results' / 'legal_depth' / 'console.log').exists()


def test_each_arm_trains_at_its_own_rate_in_both_stages(datasets):
    for arm, rate in pilot.LEARNING_RATES.items():
        for name in (f'legal_{arm}', f'engine_{arm}'):
            config = pilot.run_config(name)
            assert config['learning_rate'] == rate and config['min_lr'] == pytest.approx(rate / 10)
    assert pilot.run_config('lr_legal_depth_1e-3')['learning_rate'] == 1e-3
