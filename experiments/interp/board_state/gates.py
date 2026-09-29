"""What does the temporal mixer's gating select?

The mixer computes ``gate_m * W_m norm(memory) + gate_c * W_c norm(current)``
with per-dimension gates that depend on both inputs. Three tests:

``constant``  Replace the gates by their per-dimension means (no input
              dependence), settle, and compare test-row NLL and board probes
              per character role with the normal model. If NLL barely moves,
              the input-dependent selection does not matter for prediction.
``removed``   What the memory term drops: at roles where the current character
              varies, probe the memory and the memory term for the board and
              for the current character's identity, which is what the previous
              character was predicting. If identity drops much more than the
              board, the gate mainly strips stale prediction features.
``quality``   Do the gates react to memory quality? Gates for the same
              positions when the memory has had 1, 2 or 3 updates (as in
              training) and when it is settled.

Writes ``results/gates/<test>-<arm>.json``.

Examples:
    python -m experiments.interp.board_state.gates constant --arms temporal temporal_decay
    python -m experiments.interp.board_state.gates removed --arms temporal
    python -m experiments.interp.board_state.gates quality --arms temporal_trained temporal
"""

import argparse
from contextlib import contextmanager
import json

import chess
import numpy as np
import torch

from experiments.ablations.live_warm_start.align import prelude as run_prelude, settle
from interp.boards import encode
from interp.probes import fit
from models.recurrent_2d import shift_right
from .common import CONTEXT, RESULTS, TRAIN_ROWS, decode, device, load_arm, load_rows
from .cycle import BATCH, MIN_EPOCHS, MIN_STEPS, ROLES, capture, label_row
from .memory_scale import FOCUS as SCALE_FOCUS, run_rows

GATE_ROWS = 50
IDENTITY_ROLES = ('digit', 'first_white', 'body_white', 'last_white', 'first_black', 'body_black')
QUALITY_ROLES = ('space_number', 'digit_last', 'dot', 'first_white', 'body_white', 'space_black')


def cycle_labels(rows, meta):
    labels = [label_row(decode(row, meta), CONTEXT) for row in rows]
    role = np.stack([r for r, _, _, _ in labels])
    current = np.stack([c for _, c, _, _ in labels])
    previous = np.stack([p for _, _, p, _ in labels])
    valid = np.stack([v for _, _, _, v in labels])
    valid[:, 0] = False
    return role, current, previous, valid


def gate_values(mixer, memory, current):
    m, c = mixer.memory_norm(memory), mixer.prelude_norm(current)
    return mixer.gates(torch.cat((m, c), -1)).sigmoid().chunk(2, -1)


@contextmanager
def constant_gates(mixer, gate_m, gate_c):
    """Temporarily replace the mixer's input-dependent gates by fixed vectors."""
    original = mixer._mix

    def mix(prelude, memory):
        return gate_m * mixer.memory_value(mixer.memory_norm(memory)) + \
            gate_c * mixer.prelude_value(mixer.prelude_norm(prelude))

    mixer._mix = mix
    try:
        yield
    finally:
        mixer._mix = original


@torch.no_grad()
def mean_gates(runner, rows, device_name):
    """Per-dimension mean gates over every position of ``rows`` at the fixed point."""
    states = capture(runner, rows, device_name, ['L1', 'L7'])
    memory = torch.from_numpy(states['L7'][:, :-1].reshape(-1, states['L7'].shape[-1]))
    current = torch.from_numpy(states['L1'][:, 1:].reshape(-1, states['L1'].shape[-1]))
    sums = [0, 0]
    for start in range(0, len(memory), 32768):
        gm, gc = gate_values(runner.model.temporal_mixer, memory[start:start + 32768].to(device_name).float(),
                             current[start:start + 32768].to(device_name).float())
        sums[0] = sums[0] + gm.sum(0)
        sums[1] = sums[1] + gc.sum(0)
    return sums[0] / len(memory), sums[1] / len(memory)


def board_scores(states, labels, train_rows, device_name, roles):
    role, current, previous, valid = labels
    is_train = np.zeros(role.shape, bool)
    is_train[:train_rows] = True
    initial = encode(chess.Board(), False)
    out = {}
    for role_name in roles:
        mask = valid & (role == ROLES.index(role_name))
        train, test = mask & is_train, mask & ~is_train
        latest = current[test] != previous[test]
        earlier = (current[test] != initial) & ~latest
        epochs = max(MIN_EPOCHS, -(-MIN_STEPS * BATCH // int(train.sum())))
        out[role_name] = {}
        for site, values in states.items():
            probe = fit(values[train].astype(np.float32), current[train].astype(np.int64), 13,
                        device=device_name, epochs=epochs, batch=BATCH, lr=3e-3)
            correct = probe.predict(torch.from_numpy(values[test].astype(np.float32))).numpy() == current[test]
            out[role_name][site] = dict(latest=float(correct[latest].mean()), earlier=float(correct[earlier].mean()))
    return out


def test_constant(name, rows, meta, train_rows, device_name):
    runner, _ = load_arm(name, device_name)
    labels = cycle_labels(rows, meta)
    gate_m, gate_c = mean_gates(runner, rows[:GATE_ROWS], device_name)
    result = dict(arm=name, conditions={})
    for condition in ('normal', 'constant'):
        if condition == 'normal':
            states, nll, passes, change = run_rows(runner, rows, train_rows, device_name)
        else:
            with constant_gates(runner.model.temporal_mixer, gate_m, gate_c):
                states, nll, passes, change = run_rows(runner, rows, train_rows, device_name)
        roles = board_scores(states, labels, train_rows, device_name, SCALE_FOCUS)
        result['conditions'][condition] = dict(nll=nll, max_passes=max(passes), max_memory_change=max(change),
                                               roles=roles)
        print(f'{name} {condition}: NLL {nll:.4f}, passes up to {max(passes)}', flush=True)
        for role_name, sites in roles.items():
            print(f'    {role_name:>12}: ' + ' | '.join(f'{site} {s["latest"]:.2f}/{s["earlier"]:.2f}'
                                                     for site, s in sites.items()), flush=True)
    return result


def test_removed(name, rows, meta, train_rows, device_name):
    runner, _ = load_arm(name, device_name)
    states = capture(runner, rows, device_name, ['L1', 'L7'])
    role, current, previous, valid = cycle_labels(rows, meta)
    tokens = rows[:, :CONTEXT].astype(np.int64)
    is_train = np.zeros(role.shape, bool)
    is_train[:train_rows] = True
    initial = encode(chess.Board(), False)
    mixer = runner.model.temporal_mixer
    result = dict(arm=name, roles={})
    for role_name in IDENTITY_ROLES:
        rows_i, positions = np.nonzero(valid & (role == ROLES.index(role_name)))
        memory = torch.from_numpy(states['L7'][rows_i, positions - 1].astype(np.float32)).to(device_name)
        cur = torch.from_numpy(states['L1'][rows_i, positions].astype(np.float32)).to(device_name)
        with torch.no_grad():
            gm, _ = gate_values(mixer, memory, cur)
            memory_term = (gm * mixer.memory_value(mixer.memory_norm(memory))).cpu().numpy()
        readers = dict(memory=memory.cpu().numpy(), memory_term=memory_term)
        train = is_train[rows_i, positions]
        board, token = current[rows_i, positions].astype(np.int64), tokens[rows_i, positions]
        latest = board[~train] != previous[rows_i, positions][~train]
        earlier = (board[~train] != initial) & ~latest
        epochs = max(MIN_EPOCHS, -(-MIN_STEPS * BATCH // int(train.sum())))
        entry = dict(test=int((~train).sum()), distinct_characters=int(len(np.unique(token))), readers={})
        for reader, x in readers.items():
            board_probe = fit(x[train], board[train], 13, device=device_name, epochs=epochs, batch=BATCH, lr=3e-3)
            ok = board_probe.predict(torch.from_numpy(x[~train])).numpy() == board[~train]
            char_probe = fit(x[train], token[train, None], 32, device=device_name, epochs=epochs, batch=BATCH, lr=3e-3)
            char_ok = char_probe.predict(torch.from_numpy(x[~train]))[:, 0].numpy() == token[~train]
            entry['readers'][reader] = dict(board_latest=float(ok[latest].mean()), board_earlier=float(ok[earlier].mean()),
                                            character=float(char_ok.mean()))
        result['roles'][role_name] = entry
        print(f'{name} {role_name:>12} ({entry["distinct_characters"]} characters): ' + ' | '.join(
            f'{r}: board {v["board_latest"]:.2f}/{v["board_earlier"]:.2f}, character {v["character"]:.2f}'
            for r, v in entry['readers'].items()), flush=True)
    return result


@torch.no_grad()
def test_quality(name, rows, meta, device_name, updates=(1, 2, 3)):
    runner, _ = load_arm(name, device_name)
    model = runner.model
    role, _, _, valid = cycle_labels(rows, meta)
    sums = {u: {r: np.zeros(3) for r in QUALITY_ROLES} for u in list(updates) + ['settled']}
    for start in range(0, len(rows), 8):
        x = torch.from_numpy(rows[start:start + 8, :CONTEXT].astype(np.int64)).to(device_name)
        p = run_prelude(model, x)
        memories = {u: settle(model, p, u) for u in updates}
        # Settled memory: the final-pass T-source output at the runner's fixed point.
        b, t = x.shape
        index = (torch.arange(b, device=device_name).repeat_interleave(t), torch.arange(t, device=device_name).repeat(b))
        memories['settled'] = runner.run(x, capture=['L7'], index=index).captures['L7'].float().to(device_name).view(b, t, -1)
        for key, memory in memories.items():
            gm, gc = gate_values(model.temporal_mixer, shift_right(memory), p)
            for role_name in QUALITY_ROLES:
                mask = torch.from_numpy(valid[start:start + b] & (role[start:start + b] == ROLES.index(role_name))).to(device_name)
                sums[key][role_name] += [float(gm.mean(-1)[mask].sum()), float(gc.mean(-1)[mask].sum()), int(mask.sum())]
    result = dict(arm=name, updates={})
    for key, per_role in sums.items():
        result['updates'][str(key)] = {r: dict(memory_gate=v[0] / v[2], current_gate=v[1] / v[2]) for r, v in per_role.items()}
        print(f'{name} memory after {key} update(s): ' + ' | '.join(
            f'{r} {v[0] / v[2]:.3f}/{v[1] / v[2]:.3f}' for r, v in per_role.items()), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('test', choices=['constant', 'removed', 'quality'])
    parser.add_argument('--arms', nargs='+', default=['temporal'])
    parser.add_argument('--train-rows', type=int, default=600)
    parser.add_argument('--test-rows', type=int, default=100)
    parser.add_argument('--quality-rows', type=int, default=100)
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    rows, _, meta = load_rows()
    probe_rows = np.concatenate((rows[:args.train_rows], rows[TRAIN_ROWS:TRAIN_ROWS + args.test_rows]))
    out = RESULTS / 'gates'
    out.mkdir(parents=True, exist_ok=True)
    for name in args.arms:
        if args.test == 'constant':
            result = test_constant(name, probe_rows, meta, args.train_rows, args.device)
        elif args.test == 'removed':
            result = test_removed(name, probe_rows, meta, args.train_rows, args.device)
        else:
            result = test_quality(name, rows[TRAIN_ROWS:TRAIN_ROWS + args.quality_rows], meta, args.device)
        (out / f'{args.test}-{name}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
