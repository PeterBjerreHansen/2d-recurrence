"""Draw the 20B report's trajectory figure from the collected evaluation reports.

Left: primary four-pass training-graph NLL against characters for all four arms.
Right: live NLL at the major checkpoints. Both use the 128-row selection panel.

    uv run --with matplotlib python -m experiments.long_runs.20B_recurrence.report_figures

matplotlib is not a project dependency, hence ``--with``.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('experiments/long_runs/20B_recurrence')
OUTPUT = ROOT / 'report_figures' / 'nll_trajectories.png'
CHARACTERS_PER_UPDATE = 100 * 1023
# Reference palette, categorical slots 1-4 (light mode), fixed per arm.
COLORS = {'hybrid': '#2a78d6', 'temporal': '#eb6834', 'depth': '#1baf7a', 'transformer': '#eda100'}
PRIMARY = {'hybrid': ('training_graph', '3', '3'), 'temporal': ('training_graph', '3', '0'),
           'depth': ('training_graph', '0', '3'), 'transformer': ('baseline', '0', '0')}
LIVE = {'transformer': ('baseline', ''), 'temporal': ('live', '1'), 'depth': ('live', '4'), 'hybrid': ('live', '4')}
LABELS = {'hybrid': 'Hybrid (3,3) / live J=4', 'temporal': 'Temporal (3,0) / live J=1',
          'depth': 'Depth (0,3) / live J=4', 'transformer': 'Transformer'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def rows(arm):
    for path in sorted((ROOT / f'{arm}_20B' / 'results').glob('evaluation-selection-ckpt-step*.csv')):
        step = int(path.stem.split('step')[-1])
        yield step, list(csv.DictReader(path.open()))


def series(arm, match):
    points = []
    for step, table in rows(arm):
        for row in table:
            if match(row):
                points.append((step * CHARACTERS_PER_UPDATE / 1e9, float(row['nll'])))
    return points


def label_ends(ax, ends, gap):
    """Direct-label final values, nudging labels apart so none overlap."""
    placed = []
    for arm, x, y in sorted(ends, key=lambda item: item[2]):
        position = max(y, placed[-1] + gap) if placed else y
        placed.append(position)
        ax.annotate(f'{y:.4f}', (x, y), xytext=(x + 0.5, position), textcoords='data',
                    va='center', fontsize=8, color=INK)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    for ax in axes:
        ax.set_facecolor('#fcfcfb')
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel('Training characters (billions)', color=MUTED, fontsize=9)
    left, right = axes
    left_ends, right_ends = [], []
    for arm, (execution, u_t, u_d) in PRIMARY.items():
        points = series(arm, lambda r: r['execution'] == execution and r['u_t'] == u_t and r['u_d'] == u_d)
        x, y = zip(*points)
        left.plot(x, y, color=COLORS[arm], linewidth=2, marker='o', markersize=4, label=LABELS[arm])
        left_ends.append((arm, x[-1], y[-1]))
    left.set_ylim(0.205, 0.33)
    left.set_title('Training graph, four passes (primary)', color=INK, fontsize=10, loc='left')
    left.set_ylabel('NLL (selection panel)', color=MUTED, fontsize=9)
    for arm, (execution, depth) in LIVE.items():
        if arm == 'transformer':
            points = series(arm, lambda r: r['execution'] == 'baseline')
            points = [p for p in points if round(p[0], 1) in (1.0, 4.0, 10.0, 18.0, 20.0)]
        else:
            points = series(arm, lambda r, d=depth: r['execution'] == 'live' and r['depth_steps'] == d)
        x, y = zip(*points)
        right.plot(x, y, color=COLORS[arm], linewidth=2, marker='o', markersize=4)
        right_ends.append((arm, x[-1], y[-1]))
    label_ends(left, left_ends, 0.0045)
    label_ends(right, right_ends, 0.011)
    right.set_title('Live, token by token (as trained)', color=INK, fontsize=10, loc='left')
    right.set_xlim(left.get_xlim()[0], 22.5)
    left.set_xlim(left.get_xlim()[0], 22.5)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor='white')
    print(OUTPUT)


if __name__ == '__main__':
    main()
