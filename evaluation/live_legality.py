"""Live teacher-forced NLL with legal-move probability at every move start.

Each row is decoded token by token with real temporal feedback (the deployed
execution). Per-position losses are saved as in ``evaluation.position_losses``.
At every move start the live state is branched over all legal SAN strings
(python-chess replay of the true game) and the model's probability of each
``SAN + delimiter`` is computed exactly by walking the SAN trie:

``legal_mass``            total probability of writing a legal move, with its
                          correct ``+``/``#`` suffix, followed by `` `` or ``;``
``actual_logprob``        log probability of the move actually played
``top_legal_match``       the most probable legal move is the move played
``greedy_legal``          greedy character decoding writes a legal move
``greedy_match``          greedy decoding writes the move played

Moves too close to the context limit to branch every legal move are skipped
and counted. Progress is saved every ``--save-every`` rows and resumed.
``--shard K/N`` evaluates every N-th selected row starting at K, so shards can
run as parallel processes; the analysis merges them (``move_row`` holds the
validation row index).

Example:
    python -m evaluation.live_legality --checkpoint .../ckpt-step195504.pt \
        --depth-steps 1 --output .../battery/hybrid/live_J1.npz \
        --panel-file .../panel.json --panel-split confirmation --limit 1000 --device cuda
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from evaluation.pgn_annotations import annotate_row
from evaluation.position_losses import (checkpoint_identity, load_evaluation_target, save_npz_atomic,
                                        write_text_atomic)
from evaluation.row_selection import row_arrays, select_validation_rows
from inference.branch import branch_from_live, branch_step, select_branches
from inference.live import create_live_state, decode_live_step, validate_live_inference_spec
from training_utils import provenance


DELIMITERS = (' ', ';')
MAX_GREEDY_CHARACTERS = 8
MOVE_FIELDS = dict(move_row=np.int64, move_start=np.int64, move_ply=np.int64, move_game=np.int64,
                   n_legal=np.int64, legal_mass=np.float64, actual_logprob=np.float64,
                   top_legal_match=bool, greedy_legal=bool, greedy_match=bool, greedy_delimited=bool)


def move_legality(model, root_state, root_logits, legal_sans, actual_san, stoi, itos):
    """Score every legal SAN from one live state; ``None`` if the context is too short."""
    block_size = model.config.block_size
    start = root_state.position
    if start + max(len(san) for san in legal_sans) > block_size:
        return None
    delimiters = [stoi[char] for char in DELIMITERS]
    logprobs = {'': root_logits[0].float().log_softmax(-1).cpu()}  # tiny CPU lookups avoid device syncs
    greedy, greedy_running, greedy_delimited = '', True, False
    level_state, level_index = None, {}
    depth = 0
    while True:
        depth += 1
        nodes = sorted({san[:depth] for san in legal_sans if len(san) >= depth})
        if greedy_running:
            character = itos[int(logprobs[greedy].argmax())]
            if character in DELIMITERS:
                greedy_running, greedy_delimited = False, True
            elif len(greedy) >= MAX_GREEDY_CHARACTERS or start + len(greedy) >= block_size:
                greedy_running = False
            else:
                greedy += character
                if greedy not in nodes:
                    nodes.append(greedy)
        if not nodes:
            break
        if depth == 1:
            state = branch_from_live(root_state, len(nodes))
        else:
            state = select_branches(level_state, [level_index[node[:-1]] for node in nodes])
        logits = branch_step(model, [stoi[node[-1]] for node in nodes], state)
        for node, row in zip(nodes, logits.float().log_softmax(-1).cpu()):
            logprobs[node] = row
        level_state, level_index = state, {node: index for index, node in enumerate(nodes)}

    def sequence_logprob(san):
        total = sum(logprobs[san[:index]][stoi[san[index]]] for index in range(len(san)))
        return float(total + torch.logsumexp(logprobs[san][delimiters], dim=0))

    scores = {san: sequence_logprob(san) for san in legal_sans}
    legal = set(legal_sans)
    top = max(scores, key=scores.get)
    return dict(n_legal=len(legal_sans), legal_mass=math.fsum(math.exp(value) for value in scores.values()),
                actual_logprob=scores[actual_san], top_legal_match=top == actual_san,
                greedy_legal=greedy_delimited and greedy in legal,
                greedy_match=greedy_delimited and greedy == actual_san,
                greedy_delimited=greedy_delimited)


@torch.no_grad()
def evaluate_row(model, inputs, targets, text, spec, stoi, itos, *, legality=True):
    """Return per-position losses/correctness and per-move legality records for one row."""
    device = model.lm_head.weight.device
    annotation = annotate_row(text, legal_moves=legality)
    complete = [move for move in annotation.moves if legality and move.complete]
    # A move the replay cannot score (after an illegal move, or written in noncanonical SAN
    # such as a mate without ``#``) is counted and skipped, never scored against the wrong set.
    moves = {move.start - 1: move for move in complete
             if move.legal_sans is not None and move.san in move.legal_sans}
    state = create_live_state(model, spec.depth_steps, spec.kv_strategy)
    tokens = torch.from_numpy(inputs).to(device)
    losses = np.zeros(len(targets), dtype=np.float32)
    correct = np.zeros(len(targets), dtype=np.float32)
    records = []
    skipped = dict(context_limit=0, replay=sum(move.legal_sans is None for move in complete),
                   noncanonical=sum(move.legal_sans is not None and move.san not in move.legal_sans
                                    for move in complete))
    for position in range(len(targets)):
        logits = decode_live_step(model, tokens[position:position + 1], state)
        logprobs = logits[0].float().log_softmax(-1)
        losses[position] = -float(logprobs[targets[position]])
        correct[position] = float(int(logprobs.argmax()) == targets[position])
        move = moves.get(position)
        if move is None:
            continue
        record = move_legality(model, state, logits, move.legal_sans, move.san, stoi, itos)
        if record is None:
            skipped['context_limit'] += 1
            continue
        records.append(dict(move_start=move.start, move_ply=move.ply, move_game=move.game, **record))
    if not np.isfinite(losses).all() or any(
            not math.isfinite(record[name]) for record in records for name in ('legal_mass', 'actual_logprob')):
        raise FloatingPointError('Non-finite live loss or legality score')
    return losses, correct, records, skipped


def _signature(args, identity, spec, indices):
    payload = dict(checkpoint_sha256=identity['checkpoint_sha256'], depth_steps=spec.depth_steps,
                   kv_strategy=spec.kv_strategy, legality=not args.no_legality,
                   rows=hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _arrays(indices, losses, correct, records):
    arrays = dict(row_indices=np.asarray(indices[:len(losses)], dtype=np.int64),
                  losses=np.asarray(losses, dtype=np.float32).reshape(len(losses), -1),
                  correct=np.asarray(correct, dtype=np.float32).reshape(len(correct), -1))
    for name, dtype in MOVE_FIELDS.items():
        arrays[name] = np.asarray([record[name] for record in records], dtype=dtype)
    return arrays


def _parse_shard(value):
    try:
        shard, shards = (int(part) for part in value.split('/'))
    except ValueError:
        raise argparse.ArgumentTypeError('--shard must look like K/N') from None
    if not 0 <= shard < shards:
        raise argparse.ArgumentTypeError('--shard requires 0 <= K < N')
    return shard, shards


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New .npz path; a .json summary is written alongside')
    parser.add_argument('--depth-steps', type=int)
    parser.add_argument('--kv-strategy', choices=['final_depth', 'depth_specialized', 'ordinary'])
    parser.add_argument('--no-legality', action='store_true', help='Live NLL only')
    parser.add_argument('--panel-file')
    parser.add_argument('--panel-split', choices=['selection', 'confirmation'], default='confirmation')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--subset-seed', type=int, default=0)
    parser.add_argument('--shard', type=_parse_shard, default=(0, 1), help='K/N: every N-th row from K')
    parser.add_argument('--data-dir')
    parser.add_argument('--verify-data', action='store_true')
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    output = Path(args.output)
    if output.suffix != '.npz' or output.exists():
        parser.error('--output must be a new .npz path')
    partial = output.with_suffix('.partial.npz')

    model, checkpoint, data = load_evaluation_target(args.checkpoint, args.device, data_dir=args.data_dir,
                                                     verify_data=args.verify_data)
    spec = validate_live_inference_spec(model.config, args.depth_steps, args.kv_strategy)
    indices, selection = select_validation_rows(data, panel_file=args.panel_file, panel_split=args.panel_split,
                                                limit=args.limit, seed=args.subset_seed)
    shard, shards = args.shard
    indices = indices[shard::shards]
    selection.update(shard=shard, shards=shards, shard_row_count=len(indices))
    inputs, targets = row_arrays(data, indices)
    stoi, itos = data.meta['stoi'], data.meta['itos']
    identity = checkpoint_identity(args.checkpoint, checkpoint, model)
    signature = _signature(args, identity, spec, indices)

    losses, correct, records = [], [], []
    skipped = dict(context_limit=0, replay=0, noncanonical=0)
    elapsed = 0.0
    if partial.exists():
        saved = np.load(partial)
        state = json.loads(str(saved['state']))
        if state['signature'] != signature:
            parser.error(f'{partial} belongs to different arguments; move it away to start over')
        losses, correct = list(saved['losses']), list(saved['correct'])
        records = [{name: saved[name][index].item() for name in MOVE_FIELDS}
                   for index in range(len(saved['move_row']))]
        skipped, elapsed = state['skipped'], state['seconds']
        print(f'Resuming after {len(losses)} of {len(indices)} rows', flush=True)

    meta = dict(execution='live', prefill='sequential_live', depth_steps=spec.depth_steps,
                kv_strategy=spec.kv_strategy, legality=not args.no_legality, rows=selection, device=args.device, dtype='float32',
                provenance=provenance(), **identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    for row in range(len(losses), len(indices)):
        started = time.monotonic()
        text = ''.join(itos[int(value)] for value in (*inputs[row], targets[row][-1]))
        row_losses, row_correct, row_records, row_skipped = evaluate_row(
            model, inputs[row], targets[row], text, spec, stoi, itos, legality=not args.no_legality)
        losses.append(row_losses)
        correct.append(row_correct)
        records.extend(dict(move_row=indices[row], **record) for record in row_records)
        skipped = {key: skipped.get(key, 0) + row_skipped[key] for key in row_skipped}
        elapsed += time.monotonic() - started
        done = row + 1
        if done % args.save_every == 0 or done == len(indices):
            print(f'{done}/{len(indices)} rows, NLL {np.mean(losses):.5f}, '
                  f'legal mass {np.mean([r["legal_mass"] for r in records]) if records else float("nan"):.4f}, '
                  f'{elapsed / done:.1f}s/row', flush=True)
            state = dict(signature=signature, skipped=skipped, seconds=elapsed)
            save_npz_atomic(partial, state=json.dumps(state), **_arrays(indices, losses, correct, records))

    arrays = _arrays(indices, losses, correct, records)
    meta.update(skipped_moves=skipped, seconds=elapsed, seconds_per_row=elapsed / len(indices))
    summary = dict(meta, nll=float(arrays['losses'].mean(dtype=np.float64)),
                   accuracy=float(arrays['correct'].mean()), target_count=int(arrays['losses'].size),
                   scored_moves=len(records))
    if records:
        summary.update({name: float(arrays[name].mean()) for name in
                        ('legal_mass', 'top_legal_match', 'greedy_legal', 'greedy_match')})
    write_text_atomic(output.with_suffix('.json'), json.dumps(summary, indent=2, allow_nan=False) + '\n')
    save_npz_atomic(output, meta=json.dumps(meta), **arrays)
    partial.unlink(missing_ok=True)
    print(json.dumps({key: summary[key] for key in summary if key not in ('provenance', 'model_args', 'rows')},
                     indent=2))


if __name__ == '__main__':
    main()
