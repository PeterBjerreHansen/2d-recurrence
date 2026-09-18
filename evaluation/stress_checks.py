"""Run the bounded eight-pass numerical stress checks from the long-run handoff."""

import argparse
import json
from pathlib import Path

import torch

from data_loader import file_hash
from evaluation.all_rows import load_model
from evaluation.panels import fixed_panel_batches, load_panel
from recurrence.schedule import RecurrenceSchedule
from training_utils import provenance


def check(model, data, panel, device):
    fixed, metadata = fixed_panel_batches(data, panel, batch_size=2)
    x, y = fixed[0]
    x, y = x.to(device), y.to(device)
    results = []
    core_end = model.config.core_end
    # RecurrenceSchedule requires the write count to equal rounds - 1. Thus
    # the eight-pass (7,0) and (7,7) checks write on all seven nonfinal
    # passes; the requested single temporal write mask is used for (1,7).
    all_writes = (True, True, True, True, True, True, True)
    no_writes = (False, False, False, False, False, False, False)
    one_temporal_write = (True, False, False, False, False, False, False)
    schedules = [
        (7, 0, all_writes, no_writes),
        (7, 7, all_writes, all_writes),
        (1, 7, one_temporal_write, all_writes),
    ]
    for u_t, u_d, temporal, depth in schedules:
        states = []

        def record(module, inputs, output):
            states.append(output.detach().float().square().mean().sqrt().item())

        handle = model.transformer.h[core_end].register_forward_hook(record)
        try:
            with torch.no_grad():
                logits, loss = model(x, y, schedule=RecurrenceSchedule(temporal, depth))
        finally:
            handle.remove()
        results.append(dict(u_t=u_t, u_d=u_d, temporal_write_mask=temporal,
                            depth_write_mask=depth, finite=bool(torch.isfinite(logits).all() and
                                                               torch.isfinite(loss)),
                            nll=loss.item(), state_rms_by_pass=states,
                            expected_passes=max(u_t, u_d) + 1))
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
