"""Deterministic validation-row sets shared by the evaluation battery."""

import numpy as np

from evaluation.panels import load_panel


def select_validation_rows(data, *, panel_file=None, panel_split='confirmation', limit=None, seed=0):
    """Return sorted validation row indices and a description of how they were chosen.

    Without a panel, every validation row is used. ``limit`` takes the first
    rows of one seeded permutation, so subsets with the same seed are nested:
    a small live-inference subset stays inside a larger training-graph subset.
    """
    if panel_file:
        panel = load_panel(panel_file, data, panel_split)
        candidates = list(panel['row_indices'])
        source = dict(panel_file=panel['path'], panel_sha256=panel['sha256'], panel_split=panel_split)
    else:
        candidates = list(range(len(data.rows['val'])))
        source = dict(panel_file=None, panel_split='all_validation_rows')
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError('limit must be a positive integer')
        if limit < len(candidates):
            order = np.random.default_rng(seed).permutation(len(candidates))
            candidates = sorted(candidates[index] for index in order[:limit])
    return candidates, dict(**source, limit=limit, subset_seed=seed if limit else None,
                            row_count=len(candidates), manifest_hash=data.manifest_hash)


def row_arrays(data, indices):
    """Inputs and targets for the given validation rows as int64 arrays [rows, context]."""
    block = np.array(data.rows['val'][list(indices), :data.context_length + 1], dtype=np.int64)
    return block[:, :-1], block[:, 1:]
