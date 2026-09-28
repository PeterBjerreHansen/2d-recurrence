"""How much of the board survives the temporal mixer, and how much each mixer input contributes.

1. At ``dot`` points in Karvonen's setting (first game, first 365 characters),
   board probes are fit on the memory the dot reads (L7 of the previous
   character), on the mixer output ``Tmix`` and on ``L2``.
2. At every position of the first rows, the size of the temporal mixer's
   memory term (gate times projected memory) is compared with its
   current-character term (gate times projected prelude).
3. For looped arms, the depth mixer's previous-iteration term is compared with
   its fresh-start (L2) term entering each iteration.

Writes ``results/mixer/<arm>.json``.

Example:
    python -m experiments.interp.board_state.mixer --arms temporal hybrid depth
"""

import argparse
import json

import numpy as np
import torch

from interp.boards import KINDS
from interp.probes import fit
from models.recurrent_2d import shift_right
from .common import CONTEXT, RESULTS, device, load_arm, load_rows
from .probe import KARVONEN_CHARACTERS, TRAINING, initial_boards

CONTRIBUTION_ROWS = 40


@torch.no_grad()
def all_positions(runner, rows, sites, device_name):
    x = torch.from_numpy(rows[:, :CONTEXT].astype(np.int64)).to(device_name)
    b, t = x.shape
    index = (torch.arange(b, device=device_name).repeat_interleave(t), torch.arange(t, device=device_name).repeat(b))
    run = runner.run(x, capture=sites, index=index)
    return {site: run.captures[site].float().to(device_name).view(b, t, -1) for site in sites}


@torch.no_grad()
def memory_at_dots(runner, rows, labels, points, device_name):
    """L7 of the character before each point, from its own fixed-point run."""
    memory = np.zeros((len(points), runner.model.config.n_embd), np.float32)
    for start in range(0, len(rows), 8):
        chosen = np.flatnonzero((labels['row'][points] >= start) & (labels['row'][points] < start + 8))
        x = torch.from_numpy(rows[start:start + 8, :CONTEXT].astype(np.int64)).to(device_name)
        index = (torch.from_numpy(labels['row'][points[chosen]] - start).to(device_name),
                 torch.from_numpy(labels['index'][points[chosen]] - 1).to(device_name))
        memory[chosen] = runner.run(x, capture=['L7'], index=index).captures['L7'].float().cpu().numpy()
    return memory


def board_through_mixer(name, runner, rows, labels, device_name):
    points = np.flatnonzero((labels['kind'] == KINDS.index('dot')) & (labels['game'] == 0) &
                            (labels['index'] < KARVONEN_CHARACTERS))
    train = labels['train'][points]
    y = labels['board_rel'][points].astype(np.int64)
    changed = y != initial_boards(labels)[points]
    readers = {'memory': memory_at_dots(runner, rows, labels, points, device_name)}
    for site in ('Tmix', 'L2'):
        readers[site] = np.asarray(np.load(RESULTS / 'activations' / name / f'{site}.npy', mmap_mode='r')[points],
                                   dtype=np.float32)
    scores = {}
    for reader, x in readers.items():
        probe = fit(x[train], y[train], 13, device=device_name, **TRAINING['board_karvonen'])
        correct = probe.predict(torch.from_numpy(x[~train])).numpy() == y[~train]
        scores[reader] = dict(accuracy=float(correct.mean()), changed_accuracy=float(correct[changed[~train]].mean()))
        print(f'{name} {reader:>6}: board {scores[reader]["accuracy"]:.3f}, '
              f'changed {scores[reader]["changed_accuracy"]:.3f}', flush=True)
    return scores


@torch.no_grad()
def contributions(name, runner, rows, device_name):
    model, result = runner.model, {}
    sites = ['L1', 'L7'] + (['L2'] + [f'L6@{i}' for i in range(1, runner.depth_steps)] if runner.depth_steps > 1 else [])
    states = all_positions(runner, rows[:CONTRIBUTION_ROWS], sites, device_name)

    def quantiles(ratio):
        ratio = ratio.flatten().cpu()
        return dict(median=float(ratio.median()), q10=float(ratio.quantile(0.1)), q90=float(ratio.quantile(0.9)))

    if model.temporal_mixer is not None:
        mixer = model.temporal_mixer
        memory = mixer.memory_norm(shift_right(states['L7']))
        current = mixer.prelude_norm(states['L1'])
        alpha, beta = mixer.gates(torch.cat((memory, current), -1)).sigmoid().chunk(2, -1)
        ratio = ((alpha * mixer.memory_value(memory)).norm(dim=-1) /
                 (beta * mixer.prelude_value(current)).norm(dim=-1))[:, 1:]
        result['temporal_memory_over_current'] = quantiles(ratio)
        result['temporal_gates'] = dict(memory=float(alpha[:, 1:].mean()), current=float(beta[:, 1:].mean()))
        print(f'{name}: temporal mixer |memory term| / |current-character term| {result["temporal_memory_over_current"]}, '
              f'mean gates {result["temporal_gates"]}', flush=True)
    if runner.depth_steps > 1:
        mixer = model.depth_mixer
        fresh = mixer.anchor_value(mixer.anchor_norm(states['L2'])).norm(dim=-1)
        result['depth_previous_over_fresh'] = {}
        for iteration in range(1, runner.depth_steps):
            previous = mixer.state_value(mixer.state_norm(states[f'L6@{iteration}'])).norm(dim=-1)
            result['depth_previous_over_fresh'][f'into_{iteration + 1}'] = quantiles(previous / fresh)
        print(f'{name}: depth mixer |previous iteration| / |fresh start| {result["depth_previous_over_fresh"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=['temporal', 'hybrid', 'depth'])
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    rows, _, _ = load_rows()
    labels = dict(np.load(RESULTS / 'labels.npz'))
    out = RESULTS / 'mixer'
    out.mkdir(parents=True, exist_ok=True)
    for name in args.arms:
        runner, _ = load_arm(name, args.device)
        result = dict(arm=name)
        if runner.model.config.uses_temporal_recurrence:
            result['board_through_mixer'] = board_through_mixer(name, runner, rows, labels, args.device)
        result.update(contributions(name, runner, rows, args.device))
        (out / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
