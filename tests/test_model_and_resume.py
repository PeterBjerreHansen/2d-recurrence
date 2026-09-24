from pathlib import Path

import torch

from model import GPT, GPTConfig
from train import train


def test_model_ties_weights_and_preserves_causality():
    torch.manual_seed(1)
    model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16)).eval()
    assert model.lm_head.weight is model.transformer.wte.weight
    a = torch.randint(32, (1, 16))
    b = a.clone()
    b[:, 8:] = (b[:, 8:] + 1) % 32
    with torch.no_grad():
        first, _ = model(a, a)
        second, _ = model(b, b)
    torch.testing.assert_close(first[:, :8], second[:, :8], rtol=0, atol=0)


def test_resume_matches_uninterrupted_training(prepared_data, tmp_path):
    config = dict(dataset=str(prepared_data), n_layer=2, n_head=2, n_embd=32,
                  block_size=24, batch_size=2, gradient_accumulation_steps=2,
                  max_iters=4, eval_interval=2, eval_iters=2, log_interval=2,
                  warmup_iters=0, lr_decay_iters=4, compile=False, device='cpu',
                  dtype='float32', dropout=0.2, num_threads=2)
    full = train({**config, 'out_dir': str(tmp_path / 'full')})
    resumed_dir = str(tmp_path / 'resumed')
    train({**config, 'max_iters': 2, 'out_dir': resumed_dir})
    resumed = train({**config, 'init_from': 'resume', 'out_dir': resumed_dir})
    a = torch.load(full, weights_only=False)
    b = torch.load(resumed, weights_only=False)
    assert a['iter_num'] == b['iter_num'] == 4
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['best_val_loss'] == b['best_val_loss']
    assert torch.equal(a['rng_by_rank'][0]['batches'], b['rng_by_rank'][0]['batches'])


def test_generation_defaults_beside_checkpoint_and_preserves_existing_output(prepared_data, tmp_path, monkeypatch):
    import json
    import pickle
    import sys
    from dataclasses import asdict
    import pytest
    from model import GPT, GPTConfig
    import sample

    with (prepared_data / 'meta.pkl').open('rb') as stream:
        meta = pickle.load(stream)
    config = GPTConfig(n_layer=2, n_head=2, n_embd=8, block_size=12, vocab_size=meta['vocab_size'])
    model = GPT(config)
    checkpoint = tmp_path / 'ckpt.pt'
    torch.save(dict(config={'architecture': 'baseline'}, model_args=asdict(config),
                    model=model.state_dict(), meta=meta, iter_num=0, manifest_hash='fixture'), checkpoint)
    monkeypatch.setattr(sys, 'argv', ['sample.py', '--checkpoint', str(checkpoint),
                                    '--num-samples', '1', '--max-new-tokens', '1'])
    sample.main()
    output = tmp_path / 'generation-ckpt.json'
    assert json.loads(output.read_text())['checkpoint'] == str(checkpoint)
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        sample.main()
    assert output.read_bytes() == before


def test_checkpoint_only_recovery_resumes_exactly_without_extra_evaluation(
        prepared_data, tmp_path, monkeypatch):
    import importlib
    import json
    import pytest

    trainer = importlib.import_module('train')
    config = dict(
        dataset=str(prepared_data), architecture='recurrent', recurrence_mode='hybrid',
        n_layer=3, n_head=1, n_embd=8, n_prelude=1, n_buffer=0, n_core=1,
        n_source=0, n_coda=1, block_size=12, batch_size=1,
        gradient_accumulation_steps=2, dropout=0.2, max_iters=5,
        eval_interval=5, eval_iters=1, log_interval=1,
        checkpoint_interval=2, keep_checkpoints=True, checkpoint_steps=[3],
        update_support=[0, 1], update_probabilities=[],
        update_probability_schedule={'type': 'piecewise_constant', 'phases': [
            {'start_step': 0, 'update_probabilities': [[.5, 0], [0, .5]]},
            {'start_step': 2, 'update_probabilities': [[0, 0], [0, 1.]]},
        ]},
        lr_schedule='wsd', warmup_iters=1, lr_decay_start=3, lr_decay_iters=5,
        compile=False, device='cpu', dtype='float32', num_threads=1,
    )
    # Exercise the trainer from an extracted bundle directory without Git metadata.
    monkeypatch.chdir(tmp_path)
    full_path = train({**config, 'out_dir': str(tmp_path / 'full')})
    save = trainer.atomic_save
    recovery_dir = tmp_path / 'recovered'

    def interrupt_after_recovery_save(payload, path):
        save(payload, path)
        if payload['iter_num'] == 2 and path.name == 'ckpt.pt':
            raise RuntimeError('simulated interruption')

    monkeypatch.setattr(trainer, 'atomic_save', interrupt_after_recovery_save)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train({**config, 'out_dir': str(recovery_dir)})
    intermediate = torch.load(recovery_dir / 'ckpt.pt', weights_only=False)
    assert intermediate['iter_num'] == 2
    assert intermediate['last_eval_step'] == 0
    assert not list(recovery_dir.glob('ckpt-step*.pt'))

    monkeypatch.setattr(trainer, 'atomic_save', save)
    resumed_path = train({**config, 'out_dir': str(recovery_dir), 'init_from': 'resume'})
    full, resumed = [torch.load(path, weights_only=False) for path in (full_path, resumed_path)]
    for key in ('model', 'optimizer', 'scaler'):
        torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
    assert full['recurrence_sampler'] == resumed['recurrence_sampler']
    for key in ('torch', 'batches'):
        assert torch.equal(full['rng_by_rank'][0][key], resumed['rng_by_rank'][0][key])
    assert resumed['provenance']['commit'] is None
    assert [path.name for path in recovery_dir.glob('ckpt-step*.pt')] == ['ckpt-step000003.pt']

    def events(directory):
        return [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]

    assert [e['step'] for e in events(recovery_dir) if e['event'] == 'evaluation'] == [0, 5]
    for expected, actual in zip(
            [e for e in events(tmp_path / 'full') if e['event'] == 'train' and e['step'] > 2],
            [e for e in events(recovery_dir) if e['event'] == 'train' and e['step'] > 2], strict=True):
        for key in ('nll', 'lr', 'grad_norm_pre_clip', 'schedules'):
            assert expected[key] == actual[key]
