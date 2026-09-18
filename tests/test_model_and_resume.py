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
