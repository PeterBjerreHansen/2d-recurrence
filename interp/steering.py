"""Additive residual-stream edits along probe directions, and greedy move decoding.

An edit adds one vector per sequence at every position from that sequence's
``start`` onward: the probed character and every character generated after it.
The board is unchanged until the move being written is complete, so the same
edit is valid at all of them. Recurrent models apply edits on every settling
pass (see ``interp.sites``).

Scales follow Karvonen's dynamic-scale idea: for an edit along ``d`` that
should move a probe from ``source`` to ``target``, the coefficient makes the
probe's (target - source) logit margin reach ``margin`` at the edited point,
and ``multiplier`` then scales that coefficient.
"""

import numpy as np
import torch
import torch.nn.functional as F


def pad(sequences, device):
    """Right-pad token sequences; returns idx [batch, time] and lengths."""
    lengths = torch.tensor([len(s) for s in sequences])
    idx = torch.zeros(len(sequences), int(lengths.max()), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        idx[row, :len(sequence)] = torch.as_tensor(np.asarray(sequence, dtype=np.int64))
    return idx.to(device), lengths.to(device)


def valid_positions(lengths, width):
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]


def additive_edit(vectors, starts):
    """Edit adding ``vectors`` [batch, width] at positions >= ``starts`` [batch]."""
    def edit(h):
        mask = torch.arange(h.shape[1], device=h.device)[None, :] >= starts[:, None]
        return h + mask[..., None].to(h.dtype) * vectors[:, None, :].to(h.dtype)
    return edit


def probe_coefficients(probe, activations, head, source, target, margin):
    """Per-row coefficients along (w_target - w_source) that set the logit margin to ``margin``.

    ``head``, ``source`` and ``target`` are [batch] integer tensors; returns
    (coefficients [batch], directions [batch, width]).
    """
    logits = probe.logits(activations)
    rows = torch.arange(len(activations), device=logits.device)
    current = logits[rows, head, target] - logits[rows, head, source]
    weight = probe.weight.view(-1, probe.heads, probe.classes)
    directions = (weight[:, head, target] - weight[:, head, source]).T          # [batch, width]
    coefficients = (margin - current) / directions.pow(2).sum(-1)
    return coefficients.clamp_min(0.0), directions


def random_like(directions, generator):
    """Random directions with the norms of ``directions``."""
    noise = torch.randn(directions.shape, generator=generator).to(directions.device)
    return noise * (directions.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True))


@torch.no_grad()
def next_char_probabilities(runner, sequences, device, edits=None):
    """Next-character distribution after each sequence's last character.

    ``edits`` is ``None`` or a function of the row indices in the batch that
    returns the edit dictionary for those rows.
    """
    idx, lengths = pad(sequences, device)
    rows = list(range(len(sequences)))
    run = runner.run(idx, edits=edits(rows) if edits else None, valid=valid_positions(lengths, idx.shape[1]))
    last = torch.arange(len(sequences), device=device)
    return F.softmax(run.logits[last, lengths - 1].float(), -1).cpu()


@torch.no_grad()
def greedy_moves(runner, sequences, itos, device, edits=None, max_chars=8, stop=' ;'):
    """Greedily continue each sequence until a delimiter; return the written moves.

    Finished rows leave the batch; ``edits`` receives the remaining row indices.
    """
    sequences = [list(map(int, s)) for s in sequences]
    written = [''] * len(sequences)
    active = list(range(len(sequences)))
    for _ in range(max_chars):
        if not active:
            break
        idx, lengths = pad([sequences[row] for row in active], device)
        run = runner.run(idx, edits=edits(active) if edits else None,
                         valid=valid_positions(lengths, idx.shape[1]))
        last = torch.arange(len(active), device=device)
        tokens = run.logits[last, lengths - 1].argmax(-1).tolist()
        still = []
        for row, token in zip(active, tokens):
            if itos[token] in stop:
                continue
            written[row] += itos[token]
            sequences[row].append(token)
            still.append(row)
        active = still
    return written
