"""Build an explicit source/data transfer bundle for the frozen 5B study."""
import argparse
import io
import json
from pathlib import Path
import tarfile

from data_loader import ChessData, file_hash
from evaluation.panels import load_panel
from .run import DATASET, PANEL, PROTOCOL, recorded_protocol


def package(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new transfer path')
    protocol = recorded_protocol()
    data = ChessData(DATASET, 1023)
    panel = load_panel(PANEL, data, split='selection')
    files = {Path(name) for name in protocol['source']['files'] if Path(name).is_file()}
    files.update(DATASET / name for name in ('train.bin', 'val.bin', 'meta.pkl', 'manifest.json'))
    files.add(PANEL)
    files.add(PROTOCOL)
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Expected a regular file: {path}')
    manifest = dict(
        study=protocol['study'],
        base_commit=protocol['source']['head_commit'],
        branch=protocol['source']['branch'],
        working_tree_patch_sha256=protocol['source']['working_tree_patch_sha256'],
        description='Frozen working-tree snapshot; base commit alone is insufficient.',
        dataset_manifest_hash=data.manifest_hash,
        panel_sha256=panel['sha256'],
        files={str(path): file_hash(path) for path in sorted(files)},
    )
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
        if any(file_hash(path) != digest for path, digest in manifest['files'].items()):
            raise RuntimeError('A bundled file changed during packaging; retry')
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = file_hash(output)
    sidecar = output.with_suffix(output.suffix + '.sha256')
    sidecar.write_text(f'{digest}  {output.name}\n')
    return dict(bundle=str(output.resolve()), sha256=digest, files=len(files),
                protocol_sha256=file_hash(PROTOCOL))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))


if __name__ == '__main__':
    main()
