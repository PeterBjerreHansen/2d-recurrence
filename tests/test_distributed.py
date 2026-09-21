import os
from pathlib import Path
import socket
import subprocess
import sys


def test_distributed_accumulation_matches_global_batch():
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    result = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=2',
         '--master_addr=127.0.0.1', f'--master_port={port}', 'tests/ddp_recurrence_worker.py'],
        cwd=root, env={**os.environ, 'PYTHONPATH': str(root)}, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_distributed_training_resume(prepared_data, tmp_path):
    import torch

    root = Path(__file__).resolve().parents[1]
    config = dict(architecture='recurrent', update_support=[0, 1, 3],
                  update_probabilities=[[.1, .12, .04], [.12, .26, .08], [.04, .08, .16]],
                  recurrence_seed=19, n_layer=4, n_prelude=1, n_buffer=0, n_core=1, n_coda=1,
                  n_head=2, n_embd=16, dataset=str(prepared_data), block_size=12,
                  batch_size=1, gradient_accumulation_steps=4, max_iters=4,
                  eval_interval=2, eval_iters=1, eval_u_t=3, eval_u_d=1,
                  log_interval=1, warmup_iters=0, lr_decay_iters=4, compile=False,
                  device='cpu', dtype='float32', dropout=.2, num_threads=1, backend='gloo')

    def run(directory, limit, resume=False):
        settings = {**config, 'out_dir': str(directory), 'max_iters': limit,
                    'init_from': 'resume' if resume else 'scratch'}
        path = tmp_path / 'config.py'
        path.write_text('\n'.join(f'{key} = {value!r}' for key, value in settings.items()))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        result = subprocess.run(
            [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=2',
             '--master_addr=127.0.0.1', f'--master_port={port}', 'train.py', str(path)],
            cwd=root, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return torch.load(directory / 'ckpt.pt', weights_only=False)

    full = run(tmp_path / 'full', 4)
    run(tmp_path / 'resumed', 2)
    resumed = run(tmp_path / 'resumed', 4, resume=True)
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
    torch.testing.assert_close(full['optimizer'], resumed['optimizer'], rtol=0, atol=0)
    assert full['recurrence_sampler'] == resumed['recurrence_sampler']
    assert full['recurrence_sampler']['draw_count'] == 8  # Two synchronized microbatches per step.
    assert full['best_val_loss'] == resumed['best_val_loss']
    for rank in range(2):
        assert torch.equal(full['rng_by_rank'][rank]['torch'], resumed['rng_by_rank'][rank]['torch'])
        assert torch.equal(full['rng_by_rank'][rank]['batches'], resumed['rng_by_rank'][rank]['batches'])
