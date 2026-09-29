"""Does a character get the board from the previous character by attention?

At a move's first character the previous character is the decision character,
where the full board sits; the temporal memory hands over only part of it. For
the blocks after the mixer (L2–L8) this measures:

1. the attention each query at a focus role pays to the previous character
   (mean over heads, and the most attentive head), on held-out rows;
2. board probes, fit on unablated activations, read at L5 and L7 after
   attention from the focus characters to the previous character is blocked
   in every block after the mixer. Blocking the character two back instead is
   the control.

If blocking the previous character costs much more board than blocking the one
before it, the board comes from attention to the decision character, not only
from the memory.

Writes ``results/attention/<arm>.json``.

Example:
    python -m experiments.interp.board_state.attention --arms temporal transformer temporal_decay
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
from .cycle import BATCH, MIN_EPOCHS, MIN_STEPS, ROLES, capture, label_row

FOCUS = ('dot', 'first_white', 'first_black')
READOUTS = ('L5', 'L7')


def block_inputs(runner):
    """Site names whose state enters each block after the mixer, keyed by block index."""
    if runner.depth_steps != 1:
        raise ValueError('attention.py covers single-pass arms only')
    first = 'Tmix' if 'Tmix' in runner.site_names else 'L1'
    names = [first] + [f'L{i}' for i in range(2, 8)]
    return dict(zip(range(1, 8), names))


@torch.no_grad()
def attention_to_previous(block, x, queries):
    """Attention of each query position to the position before it: [heads, queries]."""
    attn = block.attn
    q, k, _ = attn.c_attn(block.ln_1(x)).split(attn.n_embd, dim=-1)
    heads, width = attn.n_head, attn.n_embd // attn.n_head
    q = q[queries].view(len(queries), heads, width).transpose(0, 1)
    k = k.view(len(x), heads, width).transpose(0, 1)
    scores = q @ k.transpose(1, 2) / width ** 0.5
    causal = torch.arange(len(x), device=x.device)[None, :] <= queries[:, None]
    weights = scores.masked_fill(~causal, float('-inf')).softmax(-1)
    return weights[:, torch.arange(len(queries)), queries - 1]


@contextmanager
def blocked_attention(model, blocks, forbid):
    """Blocks in ``blocks`` skip query→key pairs where ``forbid()`` [batch, time, time] is True."""
    originals = {}
    for index in blocks:
        attn = model.transformer.h[index].attn
        originals[index] = attn.forward

        def forward(x, attn=attn):
            b, t, c = x.size()
            q, k, v = attn.c_attn(x).split(attn.n_embd, dim=2)
            q, k, v = (z.view(b, t, attn.n_head, c // attn.n_head).transpose(1, 2) for z in (q, k, v))
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            allowed = (causal[None] & ~forbid()[:, :t, :t])[:, None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            return attn.resid_dropout(attn.c_proj(y.transpose(1, 2).contiguous().view(b, t, c)))

        attn.forward = forward
    try:
        yield
    finally:
        for index, forward in originals.items():
            model.transformer.h[index].attn.forward = forward


def analyse(name, rows, meta, train_rows, device_name):
    runner, _ = load_arm(name, device_name)
    model = runner.model
    labels = [label_row(decode(row, meta), CONTEXT) for row in rows]
    role = np.stack([r for r, _, _, _ in labels])
    current = np.stack([c for _, c, _, _ in labels])
    previous = np.stack([p for _, _, p, _ in labels])
    valid = np.stack([v for _, _, _, v in labels])
    valid[:, :2] = False
    focus = np.isin(role, [ROLES.index(r) for r in FOCUS]) & valid
    initial = encode(chess.Board(), False)
    test_rows = rows[train_rows:]
    result = dict(arm=name, attention={}, ablation={})

    # 1. Attention to the previous character, block by block.
    inputs = block_inputs(runner)
    states = capture(runner, test_rows, device_name, sorted(set(inputs.values())))
    for role_name in FOCUS:
        per_block = {}
        for index, site in inputs.items():
            weights = []
            for r in range(len(test_rows)):
                queries = np.flatnonzero(valid[train_rows + r] & (role[train_rows + r] == ROLES.index(role_name)))
                if len(queries):
                    x = torch.from_numpy(states[site][r].astype(np.float32)).to(device_name)
                    weights.append(attention_to_previous(model.transformer.h[index], x,
                                                         torch.from_numpy(queries).to(device_name)).cpu())
            weights = torch.cat(weights, dim=1)
            per_block[f'L{index + 1}'] = dict(mean=float(weights.mean()), top_head=float(weights.mean(1).max()))
        result['attention'][role_name] = per_block
        print(f'{name} {role_name:>12} attention to previous character (mean / top head): ' + ' | '.join(
            f'{b} {v["mean"]:.2f}/{v["top_head"]:.2f}' for b, v in per_block.items()), flush=True)

    # 2. Board after blocking attention to the previous character (or, as control, the one before it).
    clean = capture(runner, rows, device_name, list(READOUTS))
    probes = {}
    for role_name in FOCUS:
        train = (valid & (role == ROLES.index(role_name)))[:train_rows]
        epochs = max(MIN_EPOCHS, -(-MIN_STEPS * BATCH // int(train.sum())))
        for site in READOUTS:
            probes[role_name, site] = fit(clean[site][:train_rows][train].astype(np.float32),
                                          current[:train_rows][train].astype(np.int64), 13,
                                          device=device_name, epochs=epochs, batch=BATCH, lr=3e-3)
    blocks = list(inputs)
    for condition, offset in (('clean', None), ('block_previous', 1), ('block_two_back', 2)):
        if offset is None:
            ablated = {site: clean[site][train_rows:] for site in READOUTS}
        else:
            batch_forbid = {}

            def forbid():
                return batch_forbid['mask']

            ablated = {site: [] for site in READOUTS}
            with blocked_attention(model, blocks, forbid):
                for start in range(0, len(test_rows), 8):
                    chunk = test_rows[start:start + 8]
                    mask = torch.zeros(len(chunk), CONTEXT, CONTEXT, dtype=torch.bool)
                    for r in range(len(chunk)):
                        queries = torch.from_numpy(np.flatnonzero(focus[train_rows + start + r]))
                        mask[r, queries, queries - offset] = True
                    batch_forbid['mask'] = mask.to(device_name)
                    got = capture(runner, chunk, device_name, list(READOUTS))
                    for site in READOUTS:
                        ablated[site].append(got[site])
            ablated = {site: np.concatenate(v) for site, v in ablated.items()}
        result['ablation'][condition] = {}
        for role_name in FOCUS:
            test = (valid & (role == ROLES.index(role_name)))[train_rows:]
            latest = current[train_rows:][test] != previous[train_rows:][test]
            earlier = (current[train_rows:][test] != initial) & ~latest
            scores = {}
            for site in READOUTS:
                correct = probes[role_name, site].predict(torch.from_numpy(ablated[site][test].astype(np.float32))).numpy() \
                    == current[train_rows:][test]
                scores[site] = dict(latest=float(correct[latest].mean()), earlier=float(correct[earlier].mean()))
            result['ablation'][condition][role_name] = scores
            print(f'{name} {condition:>15} {role_name:>12}: ' + ' | '.join(
                f'{site} {s["latest"]:.2f}/{s["earlier"]:.2f}' for site, s in scores.items()), flush=True)
    out = RESULTS / 'attention'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=['temporal', 'transformer', 'temporal_decay'])
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
