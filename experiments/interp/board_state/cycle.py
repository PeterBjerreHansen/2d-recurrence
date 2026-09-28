"""The board and the temporal gates across the move cycle, at every character.

Every input character gets a role in the cycle (``ROLES``) and the board after
the latest move whose final character has been read. Characters of a move are
split by the side moving: the board is represented relative to the side to move
(Karvonen's mine/theirs), so one absolute-encoding probe cannot read both. A move counts as made at
its last SAN character, before any ``+``/``#``: that is where the model must
know the new position to predict a check marker.

For each role, linear probes read that board (absolute encoding) from:

``memory``  L7 of the previous character, the state the temporal mixer reads
``Tmix``    the mixer output, then ``L2``, ``L5`` and ``L7`` of the character itself

scored on three square groups: squares the latest move changed, squares
changed earlier in the game, and never-changed squares. For temporal-memory
arms, the mean memory and prelude gate values of the mixer are recorded per role.

Probes train on held-in rows and test on the held-out rows of the main
selection. Writes ``results/cycle/<arm>.json``.

Example:
    python -m experiments.interp.board_state.cycle --arms temporal transformer
"""

import argparse
import json
import re
import time

import chess
import numpy as np
import torch

from interp.boards import encode
from interp.probes import fit
from .common import CONTEXT, RESULTS, TRAIN_ROWS, decode, device, load_arm, load_rows

_MOVE_ROLES = {'first': "first character of {}'s move", 'body': "inside {}'s move",
               'last': "last character of {}'s move (move applied)", 'check': "+ or # after {}'s move"}
ROLE_NAMES = {
    'space_number': "space after Black's move (next: move number)",
    'digit': 'move-number digit',
    'dot': 'dot (next: White chooses)',
    **{f'{role}_white': text.format('White') for role, text in _MOVE_ROLES.items()},
    'space_black': "space after White's move (next: Black chooses)",
    **{f'{role}_black': text.format('Black') for role, text in _MOVE_ROLES.items()},
}
ROLES = tuple(ROLE_NAMES)
# Roles differ tenfold in size, so every probe gets at least MIN_STEPS optimizer steps.
BATCH, MIN_STEPS, MIN_EPOCHS = 512, 4000, 8
_MOVE_NUMBER = re.compile(r'\d+\.')


def label_row(text, limit):
    """Per-character role code, current board [64] and board before the latest move [64]."""
    n = min(len(text), limit)
    role = np.full(n, -1, np.int8)
    current = np.zeros((n, 64), np.int8)
    previous = np.zeros((n, 64), np.int8)
    valid = np.zeros(n, bool)
    starts = [i for i, c in enumerate(text) if c == ';']
    for game, start in enumerate(starts):
        stop = starts[game + 1] if game + 1 < len(starts) else len(text)
        board = chess.Board()
        now, before = encode(board, False), encode(board, False)
        position = start + 1

        def mark(index, code):
            if index < n:
                role[index], current[index], previous[index], valid[index] = code, now, before, True

        for token in text[start + 1:stop].split(' '):
            token_start = position
            position += len(token) + 1
            if not token:
                break
            number = _MOVE_NUMBER.match(token)
            if token_start > start + 1:
                mark(token_start - 1, ROLES.index('space_number' if number else 'space_black'))
            offset = number.end() if number else 0
            for i in range(token_start, token_start + offset - 1):
                mark(i, ROLES.index('digit'))
            if number:
                mark(token_start + offset - 1, ROLES.index('dot'))
            san = token[offset:]
            core = san.rstrip('+#')
            side = 'white' if board.turn == chess.WHITE else 'black'
            if not core or token_start + len(token) >= len(text):
                break  # incomplete final move
            for j, _ in enumerate(core[:-1]):
                mark(token_start + offset + j, ROLES.index(f'{"first" if j == 0 else "body"}_{side}'))
            try:
                board.push_san(san)
            except ValueError:
                break
            before, now = now, encode(board, False)
            last = token_start + offset + len(core) - 1
            mark(last, ROLES.index(f'last_{side}'))
            for j in range(len(core), len(san)):
                mark(token_start + offset + j, ROLES.index(f'check_{side}'))
    return role, current, previous, valid


@torch.no_grad()
def capture(runner, rows, device_name, sites):
    """Float16 [rows, positions, width] per site, over the full context."""
    out = {site: [] for site in sites}
    for start in range(0, len(rows), 8):
        x = torch.from_numpy(rows[start:start + 8, :CONTEXT].astype(np.int64)).to(device_name)
        b, t = x.shape
        index = (torch.arange(b, device=device_name).repeat_interleave(t), torch.arange(t, device=device_name).repeat(b))
        run = runner.run(x, capture=sites, index=index)
        for site in sites:
            out[site].append(run.captures[site].view(b, t, -1))
    return {site: torch.cat(values).numpy() for site, values in out.items()}


@torch.no_grad()
def gate_means(model, memory, prelude, device_name, batch=65536):
    """Mean memory and prelude gate of the temporal mixer for (memory, prelude) pairs."""
    mixer, sums = model.temporal_mixer, np.zeros(2)
    for start in range(0, len(memory), batch):
        m = mixer.memory_norm(torch.from_numpy(memory[start:start + batch]).to(device_name).float())
        p = mixer.prelude_norm(torch.from_numpy(prelude[start:start + batch]).to(device_name).float())
        alpha, beta = mixer.gates(torch.cat((m, p), -1)).sigmoid().chunk(2, -1)
        sums += [float(alpha.mean(-1).sum()), float(beta.mean(-1).sum())]
    return sums / len(memory)


def analyse(name, rows, meta, train_rows, device_name):
    started = time.monotonic()
    runner, _ = load_arm(name, device_name)
    temporal = 'Tmix' in runner.site_names
    later = [s for s in ('L2', 'L5', 'L7') if s in runner.site_names] if runner.depth_steps == 1 else \
        ['L2', f'L5@{runner.depth_steps}', 'L7']
    sites = sorted(set(['L1', 'L7'] + (['Tmix'] if temporal else []) + later))
    states = capture(runner, rows, device_name, sites)
    labels = [label_row(decode(row, meta), CONTEXT) for row in rows]
    role = np.stack([r for r, _, _, _ in labels])
    current = np.stack([c for _, c, _, _ in labels])
    previous = np.stack([p for _, _, p, _ in labels])
    valid = np.stack([v for _, _, _, v in labels])
    valid[:, 0] = False  # position 0 has no memory to read
    memory = np.concatenate((np.zeros_like(states['L7'][:, :1]), states['L7'][:, :-1]), axis=1)
    readers = {'memory': memory, **({'Tmix': states['Tmix']} if temporal else {}),
               **{site: states[site] for site in later}}
    is_train = np.zeros(role.shape, bool)
    is_train[:train_rows] = True
    initial = encode(chess.Board(), False)
    result = dict(arm=name, depth_steps=runner.depth_steps, roles={})
    for code, role_name in enumerate(ROLES):
        mask = valid & (role == code)
        train, test = mask & is_train, mask & ~is_train
        y = current[mask].astype(np.int64)
        latest = current[test] != previous[test]
        earlier = (current[test] != initial) & ~latest
        never = (current[test] == initial) & ~latest
        entry = dict(description=ROLE_NAMES[role_name], train=int(train.sum()), test=int(test.sum()), readers={})
        if temporal:
            entry['memory_gate'], entry['prelude_gate'] = gate_means(
                runner.model, memory[mask], states['L1'][mask], device_name)
        for reader, values in readers.items():
            epochs = max(MIN_EPOCHS, -(-MIN_STEPS * BATCH // int(train.sum())))
            probe = fit(values[train].astype(np.float32), current[train].astype(np.int64), 13,
                        device=device_name, epochs=epochs, batch=BATCH, lr=3e-3)
            correct = probe.predict(torch.from_numpy(values[test].astype(np.float32))).numpy() == current[test]
            entry['readers'][reader] = dict(latest=float(correct[latest].mean()) if latest.any() else None,
                                            earlier=float(correct[earlier].mean()),
                                            never=float(correct[never].mean()))
        result['roles'][role_name] = entry
        gates = (f' | gates memory {entry["memory_gate"]:.3f} prelude {entry["prelude_gate"]:.3f}'
                 if temporal else '')
        print(f'{name} {role_name:>12} (test {entry["test"]}){gates}', flush=True)
        for reader, scores in entry['readers'].items():
            latest_text = 'n/a' if scores['latest'] is None else f'{scores["latest"]:.3f}'
            print(f'    {reader:>7}: latest move {latest_text}, earlier {scores["earlier"]:.3f}, '
                  f'never {scores["never"]:.3f}', flush=True)
    result['seconds'] = time.monotonic() - started
    out = RESULTS / 'cycle'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=['temporal', 'transformer'])
    parser.add_argument('--train-rows', type=int, default=600)
    parser.add_argument('--test-rows', type=int, default=100)
    parser.add_argument('--device', default=device())
    args = parser.parse_args()
    rows, _, meta = load_rows()
    # Held-in rows come from the probe-training part of the selection, held-out rows from the test part.
    rows = np.concatenate((rows[:args.train_rows], rows[TRAIN_ROWS:TRAIN_ROWS + args.test_rows]))
    for name in args.arms:
        analyse(name, rows, meta, args.train_rows, args.device)


if __name__ == '__main__':
    main()
