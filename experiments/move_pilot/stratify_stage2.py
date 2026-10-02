"""Stage-2 results by position type: KL from Leela, top-move agreement and the gain from loops.

    uv run python -m experiments.move_pilot.stratify_stage2 --games 5000

Scores each final ``full_engine_{arm}`` model on the first ``--games`` Leela dev games (one game per
row) and splits the per-position metrics by features of the position and of Leela's distribution.
Recurrent models are scored in the training graph at fixed cells: depth at (0, J-1), which equals live
decoding with depth-specialised caches; hybrid at (J-1, J-1); temporal at its evaluation cell (4, 0).
Writes per-position arrays to ``results/stratify_stage2.npz`` and prints the tables.
"""

import argparse
import random
from pathlib import Path

import chess
import numpy as np
import torch

from model import GPT, GPTConfig
from models.recurrent_2d import Recurrent2DGPT, RecurrentGPTConfig
from moves.objectives import _per_position_sum, _top_move
from moves.rows import RowData
from moves.vocab import GAME_START, id_move
from recurrence.schedule import sample_schedule

ROOT = Path('experiments/move_pilot/results')
CELLS = {'transformer': [('J1', None)],
         'temporal': [('J1', (4, 0))],
         'depth': [(f'J{j}', (0, j - 1)) for j in (1, 2, 4, 5)],
         'hybrid': [(f'J{j}', (j - 1, j - 1)) for j in (1, 2, 4, 5)]}


def load(arm, device, run='full_engine', checkpoint='ckpt.pt'):
    checkpoint = torch.load(ROOT / f'{run}_{arm}' / checkpoint, map_location='cpu', weights_only=False)
    args = checkpoint['model_args']
    model = (Recurrent2DGPT(RecurrentGPTConfig.from_checkpoint(args)) if 'recurrence_mode' in args and arm != 'transformer'
             else GPT(GPTConfig(**args)))
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval()


@torch.no_grad()
def score(model, data, rows, cell, device, batch_size):
    """Per-position KL, top-move agreement and probability on Leela's top move."""
    kl, agree, p_top = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = data.build('leela_dev', rows[start:start + batch_size], objective='engine')
        kwargs = {} if cell is None else dict(schedule=sample_schedule(*cell, random.Random(0)))
        # The model runs on the device; the metrics run on the CPU (MPS lacks the integer scatter they use).
        logits, _ = model(batch.x.to(device), batch.targets.to(device), **kwargs)
        logits, t = logits.float().cpu(), batch.targets
        log_p = t._log_probabilities(logits)
        w = t.weights
        cross = -_per_position_sum(w * log_p, t.owner, t.count)
        entropy = -_per_position_sum(torch.where(w > 0, w * w.clamp_min(1e-12).log(), 0), t.owner, t.count)
        teacher_top = _top_move(w, t.moves, t.owner, t.count)
        kl.append((cross - entropy).cpu())
        agree.append((_top_move(log_p, t.moves, t.owner, t.count) == teacher_top).cpu())
        on_top = torch.where(t.moves == teacher_top[t.owner], log_p.exp(), 0)
        p_top.append(_per_position_sum(on_top, t.owner, t.count).cpu())
    return torch.cat(kl).numpy(), torch.cat(agree).numpy(), torch.cat(p_top).numpy()


def features(data, rows):
    """Per-position features, in the same order as the targets (row by row, position by position)."""
    arrays = data._split('leela_dev')
    out = {k: [] for k in ('ply', 'in_check', 'legal', 'pieces', 'top_prob', 'entropy', 'top_capture', 'top_check')}
    for row in rows:
        tokens = np.asarray(arrays['tokens'][row])
        a, b = arrays['offsets'][row], arrays['offsets'][row + 1]
        moves, weights = np.asarray(arrays['moves'][a:b]), np.asarray(arrays['weights'][a:b], dtype=np.float64)
        counts = np.asarray(arrays['legal_counts'][row])
        board, offset = chess.Board(), 0
        assert tokens[0] == GAME_START
        for t in range(len(counts)):
            n = int(counts[t])
            if n == 0:
                break
            w = weights[offset:offset + n] / weights[offset:offset + n].sum()
            top = id_move(int(moves[offset + int(np.argmax(w))]))
            out['ply'].append(t)
            out['in_check'].append(board.is_check())
            out['legal'].append(n)
            out['pieces'].append(len(board.piece_map()))
            out['top_prob'].append(w.max())
            out['entropy'].append(-(w[w > 0] * np.log(w[w > 0])).sum())
            out['top_capture'].append(board.is_capture(top))
            out['top_check'].append(board.gives_check(top))
            board.push(id_move(int(tokens[t + 1])))
            offset += n
    return {k: np.asarray(v) for k, v in out.items()}


STRATA = {
    'all': lambda f: np.ones_like(f['ply'], dtype=bool),
    'opening (ply < 20)': lambda f: f['ply'] < 20,
    'middlegame (20-80)': lambda f: (f['ply'] >= 20) & (f['ply'] < 80),
    'late (ply >= 80)': lambda f: f['ply'] >= 80,
    'endgame (<= 12 pieces)': lambda f: f['pieces'] <= 12,
    'in check': lambda f: f['in_check'],
    "Leela's top move a capture": lambda f: f['top_capture'],
    "Leela's top move a check": lambda f: f['top_check'],
    'quiet (top move neither)': lambda f: ~f['top_capture'] & ~f['top_check'] & ~f['in_check'],
    'Leela sure (top prob >= 0.8)': lambda f: f['top_prob'] >= 0.8,
    'Leela unsure (top prob < 0.4)': lambda f: f['top_prob'] < 0.4,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--games', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    arguments = parser.parse_args()
    data = RowData('data/leela_full_v1', 256, verify_hashes=False)
    rows = np.arange(min(arguments.games, data.rows('leela_dev')))
    f = features(data, rows)
    print(f'{len(rows)} games, {len(f["ply"])} positions', flush=True)
    results = {}
    for arm, cells in CELLS.items():
        model = load(arm, arguments.device)
        for name, cell in cells:
            kl, agree, p_top = score(model, data, rows, cell, arguments.device, arguments.batch_size)
            assert len(kl) == len(f['ply'])
            results[f'{arm}_{name}'] = dict(kl=kl, agree=agree, p_top=p_top)
            print(f'scored {arm} {name}: KL {kl.mean():.4f}, top-move agreement {agree.mean():.4f}', flush=True)
        del model
    np.savez_compressed(ROOT / 'stratify_stage2.npz', **f, **{f'{k}__{m}': v[m] for k, v in results.items() for m in v})
    headline = ['transformer_J1', 'temporal_J1', 'depth_J4', 'hybrid_J4']
    for metric, label in (('kl', 'KL from Leela (lower is better)'), ('agree', 'top-move agreement')):
        print(f'\n{label}, by stratum (share of positions)')
        print(f"{'stratum':<32}{'share':>7}" + ''.join(f'{k:>16}' for k in headline))
        for stratum, select in STRATA.items():
            mask = select(f)
            print(f'{stratum:<32}{mask.mean():>7.1%}' + ''.join(f'{results[k][metric][mask].mean():>16.4f}' for k in headline))
    # J=1 is out of distribution for depth and hybrid (U=0 is not trained after stage 1's first quarter), so
    # loop gains are measured between trained depths: J=2 to J=4, and J=4 to J=5.
    print('\nLoop gain: KL at the shallower J minus KL at the deeper J (positive = more passes help)')
    pairs = [('depth', 'J2', 'J4'), ('depth', 'J4', 'J5'), ('hybrid', 'J2', 'J4'), ('hybrid', 'J4', 'J5')]
    print(f"{'stratum':<32}" + ''.join(f'{f"{arm} {a}->{b}":>16}' for arm, a, b in pairs))
    for stratum, select in STRATA.items():
        mask = select(f)
        print(f'{stratum:<32}' + ''.join(
            f"{results[f'{arm}_{a}']['kl'][mask].mean() - results[f'{arm}_{b}']['kl'][mask].mean():>16.4f}" for arm, a, b in pairs))

if __name__ == '__main__':
    main()
