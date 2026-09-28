"""Probe accuracy by site for every arm, against blocks applied so far.

Writes ``results/figures/probe_curves.png``: one panel per board probe, one
line per arm, the random-init model as a grey reference. Looped arms show every
core iteration, so the x-axis is compute (blocks applied), not physical layer.
The temporal mixer output (``Tmix``) is drawn at the same x as ``L1``; depth
mixer outputs are left out. Side to move is decoded perfectly from L2 in every
arm, so it has no panel.

Run with ``uv run --with matplotlib python -m experiments.interp.board_state.figures``.
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import RESULTS

# Reference categorical palette, fixed order; each arm also has its own marker.
STYLE = {
    'transformer': ('#2a78d6', 'o'),
    'temporal': ('#eb6834', 's'),
    'depth': ('#1baf7a', '^'),
    'hybrid': ('#eda100', 'D'),
    'karvonen': ('#e87ba4', 'v'),
}
REFERENCE = ('random_init', '#8a8a86')
PANELS = (('board_karvonen_accuracy', "Board, Karvonen's setting (all squares)"),
          ('board_changed_accuracy', 'Board, whole rows (changed squares only)'))
LABEL_GAP = 0.03  # minimum vertical gap between end labels, as a fraction of the y range


def main():
    metrics = {}
    for path in sorted((RESULTS / 'probes').glob('*/metrics.json')):
        data = json.loads(path.read_text())
        metrics[data['arm']] = data['sites']
    fig, axes = plt.subplots(1, len(PANELS), figsize=(12, 4.8), facecolor='#fcfcfb')
    for ax, (key, title) in zip(axes, PANELS):
        ax.set_facecolor('#fcfcfb')
        if REFERENCE[0] in metrics:
            sites = metrics[REFERENCE[0]]
            ax.plot([s['applications'] for s in sites], [s[key] for s in sites], color=REFERENCE[1],
                    linewidth=1.5, linestyle='--', label='random init')
        ends = []
        for arm, (colour, marker) in STYLE.items():
            if arm not in metrics:
                continue
            sites = [s for s in metrics[arm] if s['kind'] != 'mixer' or s['name'] == 'Tmix']
            xs, ys = [s['applications'] for s in sites], [s[key] for s in sites]
            ax.plot(xs, ys, color=colour, linewidth=2, marker=marker, markersize=5, label=arm)
            ends.append([ys[-1], xs[-1], arm])
        low, high = ax.get_ylim()
        gap = LABEL_GAP * (high - low)
        for group in _groups_by_x(ends):
            group.sort()
            for i in range(1, len(group)):
                group[i][0] = max(group[i][0], group[i - 1][0] + gap)
            for y, x, arm in group:
                ax.annotate(arm, (x, y), xytext=(5, 0), textcoords='offset points', fontsize=8,
                            color='#3d3d3a', va='center')
        ax.set_title(title, fontsize=10, color='#1a1a19', loc='left')
        ax.set_xlabel('Transformer blocks applied', fontsize=9, color='#5f5e5a')
        ax.grid(color='#e8e7e2', linewidth=0.6)
        ax.tick_params(colors='#5f5e5a', labelsize=8)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color('#c3c2b7')
    axes[0].set_ylabel('Held-out probe accuracy', fontsize=9, color='#5f5e5a')
    axes[0].legend(fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout()
    out = RESULTS / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'probe_curves.png', dpi=150, facecolor=fig.get_facecolor())
    print(f'wrote {out / "probe_curves.png"}')


def _groups_by_x(ends):
    groups = {}
    for end in ends:
        groups.setdefault(end[1], []).append(end)
    return groups.values()


if __name__ == '__main__':
    main()
