"""What happens if the temporal mixer passes more of the memory?

The mixer's memory term is multiplied by ``scale`` on every settling pass (the
gates themselves are unchanged), and the model is read at its new fixed point.
For each scale this records test-row NLL, whether settling converged, and
board probes per character role at the mixer output, L2 and L5.

If the board at L2 improves while NLL gets worse, the layers after the mixer
are tuned to expect a weak memory. If both improve, the mixer is simply set
too low.

Writes ``results/memory_scale/<arm>.json``.

Example:
    python -m experiments.interp.board_state.memory_scale --arms temporal temporal_decay --scales 1 2
"""

import argparse
from contextlib import contextmanager
import json

import chess
import numpy as np
import torch
import torch.nn.functional as F

from interp.boards import encode
from interp.probes import fit
from .common import CONTEXT, RESULTS, TRAIN_ROWS, decode, device, load_arm, load_rows
from .cycle import BATCH, MIN_EPOCHS, MIN_STEPS, ROLES, label_row

FOCUS = ('space_number', 'digit_last', 'dot', 'first_white', 'space_black')
SITES = ('Tmix', 'L2', 'L5')


@contextmanager
def scaled_memory(mixer, scale):
    """Temporarily multiply the mixer's memory term by ``scale``."""
    original = mixer._mix

    def mix(prelude, memory):
        m, c = mixer.memory_norm(memory), mixer.prelude_norm(prelude)
        gate_m, gate_c = mixer.gates(torch.cat((m, c), -1)).sigmoid().chunk(2, -1)
        return scale * gate_m * mixer.memory_value(m) + gate_c * mixer.prelude_value(c)

    mixer._mix = mix
    try:
        yield
    finally:
        mixer._mix = original


@torch.no_grad()
def run_rows(runner, rows, test_from, device_name):
    """Captured sites at every position, test-row NLL, and settling statistics."""
    out = {site: [] for site in SITES}
    nll, targets, passes, change = 0.0, 0, [], []
    for start in range(0, len(rows), 8):
        block = torch.from_numpy(rows[start:start + 8, :CONTEXT + 1].astype(np.int64)).to(device_name)
        x, y = block[:, :-1], block[:, 1:]
        b, t = x.shape
        index = (torch.arange(b, device=device_name).repeat_interleave(t), torch.arange(t, device=device_name).repeat(b))
        run = runner.run(x, capture=SITES, index=index)
        for site in SITES:
            out[site].append(run.captures[site].view(b, t, -1))
        test = torch.arange(start, start + b, device=device_name) >= test_from
        if test.any():
            nll += float(F.cross_entropy(run.logits[test].transpose(1, 2), y[test], reduction='sum'))
            targets += int(y[test].numel())
        passes.append(run.passes)
        change.append(run.memory_change)
    return {site: torch.cat(v).numpy() for site, v in out.items()}, nll / targets, passes, change


def analyse(name, scales, rows, meta, train_rows, device_name):
    runner, _ = load_arm(name, device_name)
    labels = [label_row(decode(row, meta), CONTEXT) for row in rows]
    role = np.stack([r for r, _, _, _ in labels])
    current = np.stack([c for _, c, _, _ in labels])
    previous = np.stack([p for _, _, p, _ in labels])
    valid = np.stack([v for _, _, _, v in labels])
    valid[:, 0] = False
    is_train = np.zeros(role.shape, bool)
    is_train[:train_rows] = True
    initial = encode(chess.Board(), False)
    result = dict(arm=name, scales={})
    for scale in scales:
        with scaled_memory(runner.model.temporal_mixer, scale):
            states, nll, passes, change = run_rows(runner, rows, train_rows, device_name)
        entry = dict(nll=nll, max_passes=max(passes), max_memory_change=max(change), roles={})
        for role_name in FOCUS:
            mask = valid & (role == ROLES.index(role_name))
            train, test = mask & is_train, mask & ~is_train
            latest = current[test] != previous[test]
            earlier = (current[test] != initial) & ~latest
            epochs = max(MIN_EPOCHS, -(-MIN_STEPS * BATCH // int(train.sum())))
            entry['roles'][role_name] = {}
            for site in SITES:
                probe = fit(states[site][train].astype(np.float32), current[train].astype(np.int64), 13,
                            device=device_name, epochs=epochs, batch=BATCH, lr=3e-3)
                correct = probe.predict(torch.from_numpy(states[site][test].astype(np.float32))).numpy() == current[test]
                entry['roles'][role_name][site] = dict(latest=float(correct[latest].mean()),
                                                       earlier=float(correct[earlier].mean()))
        result['scales'][str(scale)] = entry
        print(f'{name} scale {scale}: NLL {nll:.4f}, passes up to {max(passes)}, '
              f'final memory change {max(change):.1e}', flush=True)
        for role_name, sites in entry['roles'].items():
            print(f'    {role_name:>12}: ' + ' | '.join(f'{site} {s["latest"]:.2f}/{s["earlier"]:.2f}'
                                                     for site, s in sites.items()), flush=True)
    out = RESULTS / 'memory_scale'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=['temporal', 'temporal_decay'])
    parser.add_argument('--scales', nargs='+', type=float, default=[1.0, 2.0])
    parser.add_argument('--train-rows', type=int, default=600)
    parser.add_argument('--test-rows', type=int, default=100)
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    rows, _, meta = load_rows()
    rows = np.concatenate((rows[:args.train_rows], rows[TRAIN_ROWS:TRAIN_ROWS + args.test_rows]))
    for name in args.arms:
        analyse(name, args.scales, rows, meta, args.train_rows, args.device)


if __name__ == '__main__':
    main()
