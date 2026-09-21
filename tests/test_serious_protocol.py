"""Budget stopping/resume and comparability invariants for expensive experiments."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

import train as trainer
from experiments import serious
from experiments.run_serious import write_once


def tiny_config(prepared_data, output):
    return dict(dataset=str(prepared_data), out_dir=str(output), architecture='baseline',
                n_layer=1, n_head=1, n_embd=8, block_size=8, batch_size=1,
                gradient_accumulation_steps=1, max_iters=100, warmup_iters=2,
                lr_decay_iters=10, training_budget_seconds=3.0,
                eval_interval=100, eval_iters=1, log_interval=1,
                device='cpu', dtype='float32', compile=False, num_threads=1)


def test_time_budget_stops_saves_and_resumes_with_same_clock(prepared_data, tmp_path, monkeypatch):
    # Each update costs exactly one second; evaluation/save calls are outside this clock.
    ticks = iter(range(1000))
    monkeypatch.setattr(trainer, 'time', SimpleNamespace(perf_counter=lambda: next(ticks)))
    config = tiny_config(prepared_data, tmp_path / 'split')
    first = trainer.train({**config, 'max_iters': 1})
    saved = torch.load(first, weights_only=False)
    assert saved['training_seconds'] == 1.0
    final = trainer.train({**config, 'init_from': 'resume'})
    actual = torch.load(final, weights_only=False)
    assert actual['training_seconds'] == 3.0
    assert actual['iter_num'] == actual['last_eval_step'] == 3
    rows = [json.loads(s) for s in (final.parent / 'metrics.jsonl').read_text().splitlines()]
    lrs = [r['lr'] for r in rows if r['event'] == 'train']
    assert lrs == pytest.approx([trainer.get_lr(10 * i / 3, {**trainer.DEFAULTS, **config}) for i in range(3)])
    full = trainer.train({**config, 'out_dir': str(tmp_path / 'full')})
    expected = torch.load(full, weights_only=False)
    torch.testing.assert_close(actual['model'], expected['model'], rtol=0, atol=0)
    torch.testing.assert_close(actual['optimizer'], expected['optimizer'], rtol=0, atol=0)
    # A completed time budget cannot silently restart on resume.
    trainer.train({**config, 'init_from': 'resume'})
    assert torch.load(final, weights_only=False)['iter_num'] == 3


@pytest.mark.parametrize('budget', [-1, float('nan'), float('inf'), True])
def test_invalid_time_budget_rejected(budget):
    with pytest.raises(ValueError, match='training_budget_seconds'):
        trainer.train({'training_budget_seconds': budget})


def test_serious_pair_requires_choice_and_matches_data_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(serious, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError):
        serious.long_run('transformer', 10**9)
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/decision.json').write_text(json.dumps(dict(mode='deep', reason='reviewed')))
    for tokens in (10**9, 64 * 10**9):
        a, b = [serious.long_run(m, tokens) for m in ('transformer', 'recurrent_a')]
        for key in ('dataset', 'batch_size', 'gradient_accumulation_steps', 'dtype', 'compile',
                    'block_size', 'max_iters', 'warmup_iters', 'lr_decay_iters', 'seed'):
            assert a[key] == b[key]
        assert a['batch_size'] * a['gradient_accumulation_steps'] == 100
        assert a['max_iters'] == math.ceil(tokens / 102300)
        assert tokens <= a['max_iters'] * 102300 < tokens + 102300
        assert b['deep_supervision'] and not a['deep_supervision']
        assert (b['n_prelude'], b['n_buffer'], b['n_core'], b['n_source'], b['n_coda']) == (1,1,4,1,1)


def test_ablation_changes_only_supervision_and_output():
    a, b = serious.ablation('final'), serious.ablation('deep')
    assert {k for k in a if a[k] != b[k]} == {'deep_supervision', 'out_dir'}
    assert sum(map(sum, serious.ablation('deep_more')['update_probabilities'])) == pytest.approx(1)


def test_frozen_receipt_refuses_changes(tmp_path):
    path = tmp_path / 'frozen.json'
    write_once(path, {'a': 1})
    write_once(path, {'a': 1})
    with pytest.raises(ValueError, match='different contents'):
        write_once(path, {'a': 2})
