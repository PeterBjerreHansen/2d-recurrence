"""Is the board lost in the temporal mixer, or only hard to read linearly after it?

The mixer output is the sum of two terms,
``gate_m * W_m norm(memory) + gate_c * W_c norm(current)``. At a few character
roles this fits board probes on:

``memory``        the memory the mixer reads (L7 of the previous character)
``memory_term``   the memory term alone
``current_term``  the current-character term alone
``mixer``         their sum, the mixer output

each with a linear probe, plus a one-hidden-layer MLP probe on ``memory`` and
``mixer``. If ``memory_term`` reads as well as ``memory``, the mixer keeps the
board and the drop in ``mixer`` comes from the current-character term
overlapping it; if the MLP recovers what the linear probe misses at
``mixer``, the board is there but not linearly readable.

Writes ``results/mixer_terms/<arm>.json``.

Example:
    python -m experiments.interp.board_state.mixer_terms --arms temporal temporal_decay
"""

import argparse
import json

import chess
import numpy as np
import torch
import torch.nn.functional as F

from interp.boards import encode
from interp.probes import fit
from .common import RESULTS, TRAIN_ROWS, decode, device, load_arm, load_rows
from .cycle import BATCH, MIN_EPOCHS, MIN_STEPS, ROLES, capture, label_row

FOCUS = ('space_number', 'digit_last', 'dot', 'first_white', 'space_black')
MLP_WIDTH = 1024


def fit_mlp(x, y, device_name, epochs, seed=0):
    """One-hidden-layer MLP board probe; returns a predict function."""
    torch.manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float32, device=device_name)
    y = torch.as_tensor(y, dtype=torch.long, device=device_name)
    mean, scale = x.mean(0), x.std(0) + 1e-3
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], MLP_WIDTH), torch.nn.GELU(),
                              torch.nn.Linear(MLP_WIDTH, 64 * 13)).to(device_name)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        for rows in torch.randperm(len(x), device=device_name).split(BATCH):
            loss = F.cross_entropy(net((x[rows] - mean) / scale).view(-1, 13), y[rows].view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict(z):
        z = torch.as_tensor(z, dtype=torch.float32, device=device_name)
        return net((z - mean) / scale).view(len(z), 64, 13).argmax(-1).cpu().numpy()
    return predict


@torch.no_grad()
def mixer_terms(mixer, memory, current, device_name, batch=32768):
    """Memory and current-character terms of the temporal mixer for paired inputs."""
    out_memory, out_current = [], []
    for start in range(0, len(memory), batch):
        m = mixer.memory_norm(torch.from_numpy(memory[start:start + batch]).to(device_name).float())
        c = mixer.prelude_norm(torch.from_numpy(current[start:start + batch]).to(device_name).float())
        gate_m, gate_c = mixer.gates(torch.cat((m, c), -1)).sigmoid().chunk(2, -1)
        out_memory.append((gate_m * mixer.memory_value(m)).cpu().numpy())
        out_current.append((gate_c * mixer.prelude_value(c)).cpu().numpy())
    return np.concatenate(out_memory), np.concatenate(out_current)


def analyse(name, rows, meta, train_rows, device_name):
    runner, _ = load_arm(name, device_name)
    states = capture(runner, rows, device_name, ['L1', 'L7', 'Tmix'])
    labels = [label_row(decode(row, meta), states['L7'].shape[1]) for row in rows]
    role = np.stack([r for r, _, _, _ in labels])
    current = np.stack([c for _, c, _, _ in labels])
    previous = np.stack([p for _, _, p, _ in labels])
    valid = np.stack([v for _, _, _, v in labels])
    valid[:, 0] = False
    is_train = np.zeros(role.shape, bool)
    is_train[:train_rows] = True
    initial = encode(chess.Board(), False)
    result = dict(arm=name, roles={})
    for role_name in FOCUS:
        mask = valid & (role == ROLES.index(role_name))
        rows_i, positions = np.nonzero(mask)
        memory = states['L7'][rows_i, positions - 1].astype(np.float32)
        mixer_out = states['Tmix'][rows_i, positions].astype(np.float32)
        memory_term, current_term = mixer_terms(runner.model.temporal_mixer, memory,
                                                states['L1'][rows_i, positions].astype(np.float32), device_name)
        gap = np.abs(memory_term + current_term - mixer_out).max() / np.abs(mixer_out).max()
        readers = dict(memory=memory, memory_term=memory_term, current_term=current_term, mixer=mixer_out)
        train, y = is_train[rows_i, positions], current[rows_i, positions].astype(np.int64)
        latest = current[rows_i, positions][~train] != previous[rows_i, positions][~train]
        earlier = (y[~train] != initial) & ~latest
        epochs = max(MIN_EPOCHS, -(-MIN_STEPS * BATCH // int(train.sum())))
        entry = dict(test=int((~train).sum()), sum_check=float(gap), readers={})

        def scores(correct):
            return dict(latest=float(correct[latest].mean()), earlier=float(correct[earlier].mean()))

        for reader, x in readers.items():
            probe = fit(x[train], y[train], 13, device=device_name, epochs=epochs, batch=BATCH, lr=3e-3)
            entry['readers'][reader] = scores(probe.predict(torch.from_numpy(x[~train])).numpy() == y[~train])
        for reader in ('memory', 'mixer'):
            predict = fit_mlp(readers[reader][train], y[train], device_name, epochs)
            entry['readers'][f'{reader}_mlp'] = scores(predict(readers[reader][~train]) == y[~train])
        result['roles'][role_name] = entry
        print(f'{name} {role_name:>12} (sum check {gap:.1e}): ' + ' | '.join(
            f'{reader} {s["latest"]:.2f}/{s["earlier"]:.2f}' for reader, s in entry['readers'].items()), flush=True)
    out = RESULTS / 'mixer_terms'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=['temporal', 'temporal_decay'])
    parser.add_argument('--train-rows', type=int, default=600)
    parser.add_argument('--test-rows', type=int, default=100)
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    rows, _, meta = load_rows()
    rows = np.concatenate((rows[:args.train_rows], rows[TRAIN_ROWS:TRAIN_ROWS + args.test_rows]))
    for name in args.arms:
        analyse(name, rows, meta, args.train_rows, args.device)


if __name__ == '__main__':
    main()
