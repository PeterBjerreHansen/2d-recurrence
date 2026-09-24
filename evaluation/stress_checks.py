"""Run bounded eight- and sixteen-pass numerical stress checks."""

import argparse
import json
import math
from pathlib import Path

import torch

from data_loader import file_hash
from evaluation.all_rows import load_model
from evaluation.panels import fixed_panel_batches, load_panel
from recurrence.schedule import RecurrenceSchedule
from training_utils import provenance


def stress_schedules(recurrence_mode):
    """Return axis-valid eight- and sixteen-pass numerical stress schedules."""
    if recurrence_mode == 'hybrid':
        cells = [(7, 0), (7, 7), (1, 7), (15, 0), (0, 15), (15, 15)]
    elif recurrence_mode == 'temporal':
        cells = [(7, 0), (15, 0)]
    elif recurrence_mode == 'depth':
        cells = [(0, 7), (0, 15)]
    else:
        raise ValueError(f'Unsupported recurrence_mode: {recurrence_mode}')
    schedules = []
    for u_t, u_d in cells:
        slots = max(u_t, u_d)
        temporal = tuple(index < u_t for index in range(slots))
        depth = tuple(index < u_d for index in range(slots))
        schedules.append((u_t, u_d, temporal, depth))
    return schedules


def check(model, data, panel, device):
    fixed, metadata = fixed_panel_batches(data, panel, batch_size=2)
    x, y = fixed[0]
    x, y = x.to(device), y.to(device)
    results = []
    core_end = model.config.core_end
    schedules = stress_schedules(model.config.recurrence_mode)
    for u_t, u_d, temporal, depth in schedules:
        states = []
        logits = loss = None
        out_of_memory = None

        def record(module, inputs, output):
            states.append(output.detach().float().square().mean().sqrt().item())

        handle = model.transformer.h[core_end].register_forward_hook(record)
        try:
            with torch.no_grad():
                logits, loss = model(x, y, schedule=RecurrenceSchedule(temporal, depth))
        except torch.cuda.OutOfMemoryError as error:
            out_of_memory = error
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
        finally:
            handle.remove()
        expected_passes = max(u_t, u_d) + 1
        finite = (out_of_memory is None and logits is not None and loss is not None and
                  bool(torch.isfinite(logits).all() and torch.isfinite(loss)) and
                  len(states) == expected_passes and all(math.isfinite(value) for value in states))
        loss_value = loss.item() if loss is not None else math.nan
        safe_states = [value if math.isfinite(value) else None for value in states]
        diagnostic_failure = (
            f'{type(out_of_memory).__name__}: {out_of_memory}' if out_of_memory else
            None if finite else 'non-finite values or an incomplete activation trajectory')
        results.append(dict(u_t=u_t, u_d=u_d, temporal_write_mask=temporal,
                            depth_write_mask=depth, finite=finite,
                            nll=loss_value if math.isfinite(loss_value) else None,
                            state_rms_by_pass=safe_states, expected_passes=expected_passes,
                            diagnostic_failure=diagnostic_failure))
    return metadata, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--panel-file', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output exists; choose a new report path')
    reports = []
    panel_hash = file_hash(args.panel_file)
    for raw in args.checkpoint:
        label, path = raw.split('=', 1)
        checkpoint, checkpoint_hash, data, model, recurrent = load_model(path, args.device)
        if not recurrent:
            raise ValueError('Stress checks require a recurrent checkpoint')
        panel = load_panel(args.panel_file, data, split='selection')
        metadata, checks = check(model, data, panel, args.device)
        reports.append(dict(label=label, checkpoint=str(Path(path).resolve()),
                            checkpoint_sha256=checkpoint_hash, checkpoint_step=checkpoint['iter_num'],
                            training_seed=checkpoint['config']['seed'], panel=metadata, checks=checks))
        if file_hash(path) != checkpoint_hash:
            raise ValueError(f'{path}: checkpoint changed during stress check')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(execution='training_graph', device=args.device, dtype='float32',
                                      panel_file_sha256=panel_hash, provenance=provenance(),
                                      checkpoints=reports), indent=2, allow_nan=False) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
