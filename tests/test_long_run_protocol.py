import json

import pytest
import torch

from data_loader import ChessData
from train import aggregate_gradient_stats, train


def write_panel(path, data, selection):
    confirmation = [index for index in range(len(data.rows['val'])) if index not in selection]
    path.write_text(json.dumps({
        'dataset_manifest_hash': data.manifest_hash,
        'validation_row_count': len(data.rows['val']),
        'selection_seed': 2027,
        'selection_indices': selection,
        'confirmation_indices': confirmation,
    }))


def test_aggregate_gradient_stats_logs_actual_clipping_and_rejects_nonfinite():
    model = torch.nn.Linear(2, 1, bias=False)
    parameter = model.weight
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    parameter.grad = torch.tensor([[3., 4.]])
    stats = aggregate_gradient_stats(model, scaler, optimizer, 1.0, 0)
    assert stats['grad_norm_pre_clip'] == pytest.approx(5.)
    assert stats['applied_clip_coefficient'] == pytest.approx(1.0 / (5.0 + 1e-6))
    assert stats['gradients_clipped'] is True
    assert torch.linalg.vector_norm(parameter.grad) == pytest.approx(1.0)
    parameter.grad = torch.tensor([[float('nan'), 1.]])
    with pytest.raises(FloatingPointError, match='Non-finite gradient'):
        aggregate_gradient_stats(model, scaler, optimizer, 1.0, 1)


@pytest.mark.parametrize('weight', [True, False])
def test_train_rejects_boolean_deep_supervision_weight(weight):
    with pytest.raises(ValueError, match='deep_supervision_lambda'):
        train({'deep_supervision_lambda': weight})


def test_panel_restricts_routine_validation_and_resume_rejects_changed_panel(prepared_data, tmp_path,
                                                                               monkeypatch):
    data = ChessData(prepared_data, 8)
    panel_path = tmp_path / 'panels.json'
    write_panel(panel_path, data, [0, 2, 4])
    validation_calls = []
    original_batch = ChessData.batch

    def recording_batch(self, split, batch_size, device, generator, allowed_indices=None):
        if split == 'val':
            validation_calls.append(tuple(allowed_indices or []))
        return original_batch(self, split, batch_size, device, generator, allowed_indices)

    monkeypatch.setattr(ChessData, 'batch', recording_batch)
    config = dict(dataset=str(prepared_data), eval_panel_path=str(panel_path), architecture='baseline',
                  n_layer=2, n_head=2, n_embd=32, block_size=8, batch_size=1,
                  gradient_accumulation_steps=1, max_iters=0, eval_interval=1, eval_iters=2,
                  log_interval=1, warmup_iters=0, lr_decay_iters=2, compile=False,
                  device='cpu', dtype='float32', num_threads=1, out_dir=str(tmp_path / 'run'))
    latest = train(config)
    assert validation_calls and all(call == (0, 2, 4) for call in validation_calls)
    checkpoint = torch.load(latest, weights_only=False)
    assert checkpoint['eval_panel_sha256']
    write_panel(panel_path, data, [0, 1, 4])
    with pytest.raises(ValueError, match='panel differs'):
        train({**config, 'init_from': 'resume'})


def test_legacy_checkpoint_resumes_after_panel_and_directory_move(prepared_data, tmp_path):
    """Content identity, not an old filesystem path, controls panel compatibility."""
    import shutil
    data = ChessData(prepared_data, 8)
    old_panel, new_panel = tmp_path / 'old-panel.json', tmp_path / 'new-panel.json'
    write_panel(old_panel, data, [0, 2, 4])
    config = dict(architecture='recurrent', dataset=str(prepared_data), block_size=8,
                  n_layer=4, n_prelude=1, n_buffer=0, n_core=1, n_source=1, n_coda=1,
                  n_head=2, n_embd=8, batch_size=1, gradient_accumulation_steps=1,
                  update_support=[0, 1], update_probabilities=[[.1, .2], [.2, .5]],
                  eval_u_t=1, eval_u_d=1, device='cpu', dtype='float32', compile=False,
                  max_iters=4, eval_interval=2, eval_iters=1, warmup_iters=0, lr_decay_iters=4,
                  eval_panel_path=str(old_panel), num_threads=1)
    expected_path = train({**config, 'out_dir': str(tmp_path / 'full')})
    original = tmp_path / 'old-run'
    checkpoint_path = train({**config, 'out_dir': str(original), 'max_iters': 2})
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    for key in ('n_buffer', 'n_source'):
        checkpoint['model_args'].pop(key)
        checkpoint['config'].pop(key)
    torch.save(checkpoint, checkpoint_path)
    relocated = tmp_path / 'relocated' / 'results'
    relocated.parent.mkdir()
    original.rename(relocated)
    shutil.move(old_panel, new_panel)
    actual_path = train({**config, 'out_dir': str(relocated), 'eval_panel_path': str(new_panel),
                         'init_from': 'resume'})
    a, b = [torch.load(path, weights_only=False) for path in (expected_path, actual_path)]
    torch.testing.assert_close(a['model'], b['model'], rtol=0, atol=0)
    torch.testing.assert_close(a['optimizer'], b['optimizer'], rtol=0, atol=0)
    assert a['recurrence_sampler'] == b['recurrence_sampler']
    # Merely relocating is allowed; changing the panel remains forbidden.
    write_panel(new_panel, data, [1, 3, 5])
    with pytest.raises(ValueError, match='panel differs'):
        train({**config, 'out_dir': str(relocated), 'eval_panel_path': str(new_panel),
               'init_from': 'resume'})
