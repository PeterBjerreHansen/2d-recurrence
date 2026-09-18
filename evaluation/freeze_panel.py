"""Create the deterministic selection/confirmation validation panel once."""

import argparse
import json
import random
from pathlib import Path

from data_loader import ChessData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='chess_long_v1')
    parser.add_argument('--output', required=True, help='Write the frozen panel inside the consuming experiment')
    args = parser.parse_args()
    data = ChessData(Path('data') / args.dataset, 1023)
    output = Path(args.output)
    count = len(data.rows['val'])
    if count == 1:
        raise ValueError('The validation split has only one row; no confirmation panel is possible')
    if count >= 129:
        selection = sorted(random.Random(2027).sample(range(count), 128))
    else:
        selection = list(range(max(1, count // 2)))
    confirmation = [index for index in range(count) if index not in set(selection)]
    panel = dict(dataset_manifest_hash=data.manifest_hash, validation_row_count=count,
                 selection_seed=2027, selection_indices=selection,
                 confirmation_indices=confirmation)
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != panel:
            raise FileExistsError(f'{output} exists but does not match the frozen panel')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(panel, indent=2) + '\n')
    print(json.dumps(panel, indent=2))


if __name__ == '__main__':
    main()
