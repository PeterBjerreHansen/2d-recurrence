import json

import numpy as np
import pytest

from data_loader import ChessData
from evaluation.compare_losses import load_run, paired_difference, row_sums, strata
from evaluation.row_selection import select_validation_rows


def test_row_sums_and_paired_difference_match_direct_means():
    rng = np.random.default_rng(3)
    left, right = rng.random((6, 5)), rng.random((6, 5))
    codes = rng.integers(-1, 3, size=(6, 5))
    left_sums, counts = row_sums(left, codes, 3)
    right_sums, _ = row_sums(right, codes, 3)
    weights = [np.ones(6)] * 4
    result = paired_difference(left_sums, right_sums, counts, weights)
    for category in range(3):
        mask = codes == category
        expected = left[mask].mean() - right[mask].mean()
        assert result[category]['difference'] == pytest.approx(expected)
        assert result[category]['ci_low'] == pytest.approx(expected)   # identical unit weights
        assert result[category]['count'] == mask.sum()


def test_load_run_merges_shards_in_row_order(tmp_path):
    for shard, rows in enumerate(([5, 1], [3])):
        losses = np.array([[row, row] for row in rows], dtype=np.float32)
        meta = dict(manifest_hash='m', rows=dict(shard=shard, shards=2, shard_row_count=len(rows)))
        np.savez(tmp_path / f'run.shard{shard}.npz', row_indices=np.array(rows), losses=losses,
                 correct=np.ones_like(losses), meta=json.dumps(meta))
    run = load_run(str(tmp_path / 'run.shard*.npz'))
    assert run['rows'].tolist() == [1, 3, 5]
    assert run['losses'][:, 0].tolist() == [1, 3, 5]


def test_seeded_subsets_are_nested_and_strata_cover_targets(prepared_data):
    data = ChessData(prepared_data, 1023)
    small, _ = select_validation_rows(data, limit=2, seed=4)
    large, _ = select_validation_rows(data, limit=5, seed=4)
    assert set(small) <= set(large)
    families = strata(data, large)
    classes, names = families['character_class']
    assert classes.shape == (len(large), 1023) and len(names) == 5
    assert (classes >= 0).mean() > 0.99


def _shard(path, row_indices, **meta):
    losses = np.ones((len(row_indices), 2), dtype=np.float32)
    np.savez(path, row_indices=np.array(row_indices), losses=losses, correct=losses,
             meta=json.dumps(dict(manifest_hash='m', checkpoint_sha256='a', depth_steps=1, **meta)))


def test_load_run_rejects_mixed_checkpoints_and_incomplete_shards(tmp_path):
    _shard(tmp_path / 'x.shard0of2.npz', [1], rows=dict(shard=0, shards=2, limit=4))
    _shard(tmp_path / 'x.shard1of2.npz', [2], rows=dict(shard=1, shards=2, limit=4), execution='other')
    with pytest.raises(ValueError, match='different'):
        load_run(str(tmp_path / 'x.shard*.npz'))
    _shard(tmp_path / 'y.shard0of2.npz', [1], rows=dict(shard=0, shards=2, limit=4))
    with pytest.raises(ValueError, match='expected shards'):
        load_run(str(tmp_path / 'y.shard*.npz'))


def test_atomic_save_publishes_only_complete_files(tmp_path):
    from evaluation.position_losses import save_npz_atomic
    save_npz_atomic(tmp_path / 'cell.npz', values=np.arange(3))
    assert np.load(tmp_path / 'cell.npz')['values'].tolist() == [0, 1, 2]
    assert [path.name for path in tmp_path.iterdir()] == ['cell.npz']


def test_battery_refuses_final_comparisons_from_incomplete_stages(tmp_path):
    from argparse import Namespace
    from experiments.evaluation_battery import run as battery
    args = Namespace(max_passes=4, max_live_depth=1, live_workers=2, allow_partial=False, bootstrap=10)
    paths = {arm: 'unused' for arm in ('transformer', 'temporal', 'depth', 'hybrid')}
    with pytest.raises(SystemExit, match='Refusing final comparisons'):
        battery.compare(args, paths, tmp_path)
    assert not list(tmp_path.glob('compare_*'))
