"""Validation-panel loading and exact row-aligned batch construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_loader import file_hash


def load_panel(path, data, split='selection'):
    """Validate a frozen panel against the loaded dataset and return its rows."""
    if split not in {'selection', 'confirmation'}:
        raise ValueError("panel split must be 'selection' or 'confirmation'")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Validation panel does not exist: {path}')
    try:
        panel = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'{path}: invalid JSON panel') from error
    validation_count = len(data.rows['val'])
    if panel.get('dataset_manifest_hash') != data.manifest_hash:
        raise ValueError(f'{path}: panel dataset manifest differs from loaded data')
    if panel.get('validation_row_count') != validation_count:
        raise ValueError(f'{path}: panel validation row count differs from loaded data')
    selection = panel.get('selection_indices')
    confirmation = panel.get('confirmation_indices')
    if not isinstance(selection, list) or not isinstance(confirmation, list):
        raise ValueError(f'{path}: panel must contain selection_indices and confirmation_indices')
    if selection != sorted(selection) or confirmation != sorted(confirmation):
        raise ValueError(f'{path}: panel indices must be sorted')
    if any(type(index) is not int or index < 0 or index >= validation_count
           for index in selection + confirmation):
        raise ValueError(f'{path}: panel index is outside the validation split')
    if len(set(selection)) != len(selection) or len(set(confirmation)) != len(confirmation):
        raise ValueError(f'{path}: panel indices must be distinct')
    if set(selection).intersection(confirmation):
        raise ValueError(f'{path}: selection and confirmation panels overlap')
    if sorted(selection + confirmation) != list(range(validation_count)):
        raise ValueError(f'{path}: panels must partition the validation split')
    indices = selection if split == 'selection' else confirmation
    if not indices:
        raise ValueError(f'{path}: {split} panel is empty')
    return dict(path=str(path.resolve()), sha256=file_hash(path), split=split,
                dataset_manifest_hash=data.manifest_hash,
                validation_row_count=validation_count,
                selection_indices=selection, confirmation_indices=confirmation,
                row_indices=indices, row_count=len(indices))


def fixed_panel_batches(data, panel, batch_size):
    """Build canonical, exact-once CPU batches, including a final partial batch."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    indices = list(panel['row_indices'])
    batches = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        block = np.array(data.rows['val'][selected, :data.context_length + 1], dtype=np.int64)
        batches.append((torch.from_numpy(block[:, :-1].copy()),
                        torch.from_numpy(block[:, 1:].copy())))
    digest = hashlib.sha256()
    targets = 0
    for x, y in batches:
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
        targets += y.numel()
    return batches, dict(row_indices=indices, row_count=len(indices), batch_size=batch_size,
                         batch_count=len(batches), batch_sha256=digest.hexdigest(),
                         target_count=targets,
                         sampling='every specified panel row exactly once in canonical row order')
