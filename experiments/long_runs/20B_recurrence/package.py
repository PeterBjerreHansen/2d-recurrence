"""Build a verified source/data bundle for the frozen 20B study."""
import argparse
import io
import json
from pathlib import Path
import tarfile

from data_loader import file_hash
from evaluation.panels import load_panel
from .run import DATASET, PANEL, PROTOCOL, verify_protocol


def package(output):
    output = Path(output)
    sidecar = output.with_suffix(output.suffix + '.sha256')
    temporary = output.with_suffix(output.suffix + '.partial')
    if output.exists() or sidecar.exists() or temporary.exists():
        raise FileExistsError(f'{output}, its checksum, or its partial file exists; choose a new transfer path')
    protocol, data, _ = verify_protocol()
    if data is None:
        raise RuntimeError(f'{DATASET} must be fully materialized before packaging')
    panel = load_panel(PANEL, data, split='selection')
    files = set()
    for name, digest in protocol['source']['files'].items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'Frozen source file is missing or changed: {path}')
        files.add(path)
    files.update(DATASET / name for name in ('train.bin', 'val.bin', 'meta.pkl', 'manifest.json'))
    files.update((PANEL, PROTOCOL))
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Expected a regular file in the transfer bundle: {path}')
    manifest = dict(
        study=protocol['study'],
        base_commit=protocol['source']['head_commit'],
        branch=protocol['source']['branch'],
        working_tree_patch_sha256=protocol['source']['working_tree_patch_sha256'],
        description='Frozen source/data snapshot; the base commit alone is insufficient.',
        dataset_manifest_hash=data.manifest_hash,
        panel_sha256=panel['sha256'],
        files={str(path): file_hash(path) for path in sorted(files)},
    )
    encoded = (json.dumps(manifest, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(temporary, 'w:gz') as archive:
            for path in sorted(files):
                archive.add(path, arcname=str(path), recursive=False)
            info = tarfile.TarInfo('TRANSFER_MANIFEST.json')
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
        if any(file_hash(path) != digest for path, digest in manifest['files'].items()):
            raise RuntimeError('A bundled file changed during packaging; retry the operation')
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = file_hash(output)
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
