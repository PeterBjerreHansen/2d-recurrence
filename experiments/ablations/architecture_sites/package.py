"""Bundle current source and frozen data, including uncommitted experiment files.

This does not include credentials, .git, environments, or training outputs.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

from data_loader import ChessData, file_hash
from evaluation.panels import load_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='experiments/ablations/architecture_sites/results/transfer.tar.gz')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose another name')
    data = ChessData('data/chess_long_v1', 1023)
    panel_path = Path('experiments/ablations/architecture_sites/panel.json')
    panel = load_panel(panel_path, data, split='selection')
    files = set(Path('.').glob('*.py'))
    for directory in ('configs', 'models', 'recurrence', 'evaluation', 'experiments', 'tests', 'docs'):
        files.update(p for p in Path(directory).rglob('*') if p.suffix in ('.py', '.md') and 'results' not in p.parts)
    files.update(Path(name) for name in ('pyproject.toml', 'uv.lock', '.gitignore', 'README.md', 'LICENSE') if Path(name).exists())
    files.update(Path('data/chess_v1') / name for name in ('prepare.py', 'meta.pkl'))  # Test fixtures.
    files.add(Path('docs/upstream.json'))
    files.add(panel_path)
    files.add(Path('experiments/long_runs/baseline/panel.json'))
    files.add(Path('experiments/relocations.json'))
    files.update(Path('data/chess_long_v1') / name for name in ('train.bin', 'val.bin', 'meta.pkl', 'manifest.json'))
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Expected a regular local file: {path}')
    manifest = dict(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    description='Current working-tree snapshot, including uncommitted files; base commit alone is insufficient.',
                    dataset_manifest_hash=data.manifest_hash, panel_sha256=panel['sha256'],
                    files={str(p): file_hash(p) for p in sorted(files)})
    encoded = (json.dumps(manifest, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    try:
        with tarfile.open(temporary, 'w:gz') as archive:
            for path in sorted(files):
                archive.add(path, arcname=str(path), recursive=False)
            info = tarfile.TarInfo('TRANSFER_MANIFEST.json')
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
        # Reject concurrent source/data edits rather than emitting an ambiguous bundle.
        if any(file_hash(path) != digest for path, digest in manifest['files'].items()):
            raise RuntimeError('A bundled file changed during packaging; retry with a fresh output')
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = file_hash(output)
    output.with_suffix(output.suffix + '.sha256').write_text(f'{digest}  {output.name}\n')
    print(json.dumps(dict(bundle=str(output.resolve()), sha256=digest, files=len(files)), indent=2))


if __name__ == '__main__':
    main()
