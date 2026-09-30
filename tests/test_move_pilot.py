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
