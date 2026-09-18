"""Apply the predeclared A/B selection rule to exact panel grid reports."""

import json
from pathlib import Path

from data_loader import file_hash


ROOT = Path(__file__).resolve().parents[3]
STEPS = [750, 1000]
RUNS = {
    'A': ('experiments/long_runs/baseline/configs/lr3e4.py', 'experiments/long_runs/baseline/results/lr3e-4'),
    'B': ('experiments/long_runs/baseline/configs/lr1e4.py', 'experiments/long_runs/baseline/results/lr1e-4'),
}


def read_grid(directory, step):
    path = ROOT / directory / f'grid-selection-step{step:06d}.json'
    report = json.loads(path.read_text())
    if report.get('panel_split') != 'selection' or report.get('row_count') != 128:
        raise ValueError(f'{path}: not the complete frozen selection panel')
    cells = {(cell['u_t'], cell['u_d']): cell for cell in report['cells']}
    if len(cells) != 9 or any(not isinstance(cell['nll_mean'], (int, float)) for cell in cells.values()):
        raise ValueError(f'{path}: incomplete or nonnumeric grid')
    if any(not (float('-inf') < cell['nll_mean'] < float('inf')) for cell in cells.values()):
        raise ValueError(f'{path}: non-finite trained-support NLL')
    return report


def main():
    panel_path = ROOT / 'experiments/long_runs/baseline/results/panels.json'
    panel_hash = file_hash(panel_path)
    candidates = {}
    for label, (config, directory) in RUNS.items():
        reports = {step: read_grid(directory, step) for step in STEPS}
        by_cell = {
            str(step): {(cell['u_t'], cell['u_d']): cell['nll_mean'] for cell in reports[step]['cells']}
            for step in STEPS
        }
        s = sum(by_cell[str(step)][(3, 3)] for step in STEPS) / len(STEPS)
        weighted = []
        for step in STEPS:
            report = reports[step]
            probabilities = {(t, d): report['cells'][i * 3 + j]['training_probability']
                             for i, t in enumerate([0, 1, 3]) for j, d in enumerate([0, 1, 3])}
            weighted.append(sum(by_cell[str(step)][cell] * probabilities[cell]
                                for cell in by_cell[str(step)]))
        candidates[label] = dict(
            eligible=True, config_path=str((ROOT / config).resolve()),
            config_sha256=file_hash(ROOT / config), directory=str((ROOT / directory).resolve()),
            checkpoint_sha256={str(step): file_hash(ROOT / directory / f'ckpt-step{step:06d}.pt')
                               for step in [0, 250, 500, 750, 1000]},
            report_sha256={str(step): file_hash(ROOT / directory / f'grid-selection-step{step:06d}.json')
                          for step in [0, 250, 500, 750, 1000]},
            S=s, W=sum(weighted) / len(weighted),
            s_by_checkpoint={str(step): by_cell[str(step)][(3, 3)] for step in STEPS},
            weighted_by_checkpoint={str(step): weighted[index] for index, step in enumerate(STEPS)},
        )
    a, b = candidates['A'], candidates['B']
    conditions = dict(
        b_S_margin=b['S'] <= a['S'] - 0.005,
        b_33_not_worse=all(b['s_by_checkpoint'][str(step)] <= a['s_by_checkpoint'][str(step)]
                            for step in STEPS),
        b_W_tolerance=b['W'] <= a['W'] + 0.005,
    )
    selected = 'B' if all(conditions.values()) else 'A'
    result = dict(rule='Choose B only if S_B <= S_A - 0.005, B (3,3) is no worse at 750 and 1000, '
                       'and W_B <= W_A + 0.005; otherwise choose A.',
                  panel_path=str(panel_path.resolve()), panel_sha256=panel_hash,
                  steps=STEPS, candidates=candidates, rule_conditions=conditions,
                  selected_config=selected, selected_directory=candidates[selected]['directory'],
                  outcome='B satisfies all conditions' if selected == 'B' else 'A selected by default rule')
    output = ROOT / 'experiments/long_runs/baseline/results/selection.json'
    if output.exists():
        raise FileExistsError(f'{output} exists; selection must be written once before continuation')
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
