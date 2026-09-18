"""Record a complete text patch and hashes for the frozen training protocol."""

import difflib
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
FILES = [
    'configurator.py', 'data_loader.py', 'model.py', 'train.py', 'training_utils.py',
    'data/chess_v1/prepare.py', 'models/recurrent_2d.py', 'recurrence/schedule.py',
    'experiments/sweeps/baseline_lr_selection/configs/lr3e4.py', 'experiments/sweeps/baseline_lr_selection/configs/lr1e4.py',
    'evaluation/all_rows.py', 'evaluation/feedback_diagnostic.py', 'evaluation/freeze_panel.py',
    'evaluation/panels.py', 'evaluation/recurrence_grid.py',
    'experiments/sweeps/baseline_lr_selection/PLAN.md', 'experiments/sweeps/baseline_lr_selection/HANDOFF.md',
    'docs/RECURRENCE_CONTRACT.md',
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    output = ROOT / 'experiments/sweeps/baseline_lr_selection/results/preflight'
    if (output / 'provenance.json').exists() or (output / 'training-provenance.patch').exists():
        raise FileExistsError('Protocol already frozen; preserve it and define a new experiment for changed code')
    output.mkdir(parents=True, exist_ok=True)
    patch = subprocess.run(['git', 'diff', '--binary', 'HEAD', '--', *FILES], cwd=ROOT,
                           check=True, stdout=subprocess.PIPE).stdout
    for relative in FILES:
        path = ROOT / relative
        tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', relative], cwd=ROOT,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if path.exists() and not tracked:
            lines = path.read_text().splitlines(keepends=True)
            patch += ''.join(difflib.unified_diff([], lines, fromfile='/dev/null',
                                                   tofile=f'b/{relative}')).encode()
    patch_path = output / 'training-provenance.patch'
    patch_path.write_bytes(patch)
    files = {relative: sha256(ROOT / relative) for relative in FILES if (ROOT / relative).exists()}
    manifest = dict(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                        text=True).strip(),
                    branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT,
                                                   text=True).strip(),
                    files=files, patch_path=str(patch_path.resolve()),
                    patch_sha256=sha256(patch_path))
    (output / 'provenance.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
