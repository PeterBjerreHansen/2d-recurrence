"""Convert Karvonen's released Lichess checkpoints for the evaluation battery.

Downloads ``lichess_{8,16}layers_ckpt_no_optimizer.pt`` from
huggingface.co/adamkarvonen/chess_llms unless already present, checks the
SHA-256, strips the ``torch.compile`` prefix from the weights, and adds the
metadata ``evaluation.position_losses`` needs (dataset, manifest hash,
vocabulary). The weights are not changed. The release carries no vocabulary; it uses
the same 32-character one as ours, which the Stage 1 check against the upstream
model confirmed. A different vocabulary size is refused.

    python -m experiments.evaluation_battery.karvonen_reference --layers 8 --dir <download dir>
    python -m evaluation.position_losses --checkpoint <dir>/lichess_8layers_adapted.pt \\
        --output-dir experiments/evaluation_battery/results/karvonen-reference/karvonen_8L --cell 0,0 \\
        --device cuda --panel-file experiments/long_runs/20B_recurrence/results/panel.json \\
        --panel-split confirmation --subset-seed 0 --limit 8192
"""

import argparse
from pathlib import Path
import urllib.request

import torch

from data_loader import ChessData, file_hash

URL = 'https://huggingface.co/adamkarvonen/chess_llms/resolve/main/{name}'
RELEASES = {
    8: ('lichess_8layers_ckpt_no_optimizer.pt', '84076e42e006c8edeabcb593e0b67469bdea17e1beabe78a459c2dd1f22c0668'),
    16: ('lichess_16layers_ckpt_no_optimizer.pt', '6997273be54da26e8f757436e2a65262f200d714407ed1ab7290a69e67b3a8b3'),
}
DATASET = 'chess_8M_v1'  # prepared from the same lichess_6gb_blocks.zip and split


def adapt(release, data, source_name, source_sha256):
    """Return a battery-loadable checkpoint with the release's weights unchanged."""
    if release['model_args']['vocab_size'] != len(data.meta['stoi']):
        raise ValueError('Release vocabulary size differs from the dataset')
    return dict(
        model={key.removeprefix('_orig_mod.'): value for key, value in release['model'].items()},
        model_args=dict(release['model_args']),
        config=dict(architecture='baseline', dataset=DATASET, seed=None, source='karvonen_release'),
        iter_num=release['iter_num'], manifest_hash=data.manifest_hash, meta=data.meta,
        source=dict(file=source_name, sha256=source_sha256,
                    best_val_loss=float(release.get('best_val_loss', float('nan'))),
                    dataset=release.get('dataset')),
        evaluation_only=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--layers', type=int, choices=sorted(RELEASES), default=8)
    parser.add_argument('--dir', required=True, help='Download and output directory')
    args = parser.parse_args()
    name, expected = RELEASES[args.layers]
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / name
    if not source.exists():
        urllib.request.urlretrieve(URL.format(name=name), source)
    digest = file_hash(source)
    if digest != expected:
        raise SystemExit(f'{source}: SHA-256 {digest} differs from the release {expected}')
    release = torch.load(source, map_location='cpu', weights_only=False)
    data = ChessData(Path('data') / DATASET, release['model_args']['block_size'], verify_hashes=False)
    output = directory / f'lichess_{args.layers}layers_adapted.pt'
    torch.save(adapt(release, data, name, digest), output)
    print(output)


if __name__ == '__main__':
    main()
