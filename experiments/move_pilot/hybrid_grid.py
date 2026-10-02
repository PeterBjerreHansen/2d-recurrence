"""The stage-2 hybrid on the full grid of update counts (U_T, U_D), to separate the two axes.

    uv run python -m experiments.move_pilot.hybrid_grid --games 5000

Scores the final ``full_engine_hybrid`` model in the training graph at every cell U_T, U_D in 0..4 on the
first ``--games`` Leela dev games, overall and by stratum. Holding U_T = 4 and varying U_D shows what
depth updates add given full temporal memory; holding U_D = 4 and varying U_T shows the temporal
contribution. Each cell's share of training updates is printed alongside: cells far from the diagonal
were rarely trained, so a poor score there may mean undertrained rather than useless.
"""

import argparse

import numpy as np
import torch

from experiments.move_pilot.stratify_stage2 import ROOT, STRATA, features, load, score
from moves.rows import RowData

COUNTS = range(5)


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
    checkpoint = torch.load(ROOT / 'full_engine_hybrid' / 'ckpt.pt', map_location='cpu', weights_only=False)
    trained = np.asarray(checkpoint['config']['update_probabilities'])
    model = load('hybrid', arguments.device)
    grid = {}
    for u_t in COUNTS:
        for u_d in COUNTS:
            kl, agree, _ = score(model, data, rows, (u_t, u_d), arguments.device, arguments.batch_size)
            grid[u_t, u_d] = dict(kl=kl, agree=agree)
            print(f'scored ({u_t}, {u_d}): KL {kl.mean():.4f}, agreement {agree.mean():.4f}', flush=True)
    np.savez_compressed(ROOT / 'hybrid_grid.npz', **f, **{f'{u_t}_{u_d}__{m}': v[m] for (u_t, u_d), v in grid.items() for m in v})

    def table(title, value):
        print(f'\n{title}\nrows U_T, columns U_D' + '\n' + ' ' * 8 + ''.join(f'{u_d:>10}' for u_d in COUNTS))
        for u_t in COUNTS:
            print(f'{u_t:>8}' + ''.join(f'{value(u_t, u_d):>10}' for u_d in COUNTS))

    table("Share of stage-2 training updates per cell", lambda t, d: f'{trained[t][d]:.3f}')
    for stratum, select in STRATA.items():
        mask = select(f)
        table(f'KL from Leela, {stratum} ({mask.mean():.1%} of positions)', lambda t, d: f"{grid[t, d]['kl'][mask].mean():.4f}")
    table('Top-move agreement, all positions', lambda t, d: f"{grid[t, d]['agree'].mean():.4f}")
    print('\nOne axis at a time from the trained corner (4, 4): KL increase when that axis is cut (positive = it helps)')
    print(f"{'stratum':<32}" + ''.join(f'{h:>12}' for h in ('U_D 4->3', 'U_D 4->2', 'U_D 4->0', 'U_T 4->3', 'U_T 4->2', 'U_T 4->0')))
    for stratum, select in STRATA.items():
        mask = select(f)
        base = grid[4, 4]['kl'][mask].mean()
        cuts = [grid[4, d]['kl'][mask].mean() - base for d in (3, 2, 0)] + [grid[t, 4]['kl'][mask].mean() - base for t in (3, 2, 0)]
        print(f'{stratum:<32}' + ''.join(f'{c:>12.4f}' for c in cuts))


if __name__ == '__main__':
    main()
