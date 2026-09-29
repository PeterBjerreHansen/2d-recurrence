"""Move-token datasets and packed training batches.

A dataset directory holds, for each split, ``{split}.bin`` (uint16 ply ids, games
concatenated) and ``{split}_offsets.npy`` (int64 game boundaries), described by
``manifest.json``. An engine-annotated dataset also holds, per split,
``{split}_legal_offsets.npy``, ``{split}_legal_moves.bin`` and one
``{split}_labels_{budget}.npy`` per node budget. Positions are aligned with plies:
position ``j`` is the board before ply ``j`` of the split's token array.

Batches pack whole games into rows of ``context_length + 1`` tokens, each game
starting with ``GAME_START``; games never cross rows and row tails are padded.
A game that doesn't fit starts the next row of the same batch. The inputs are
the first ``context_length`` tokens and position ``t`` predicts token ``t + 1``
when that token is a ply of the same game. State carries across games within a
row. Each row comes from human games or, with probability ``random_fraction``,
from uniformly random games, which are generated on the fly and truncated to
the space left in the row.

For evaluation, ``one_game_per_row`` places a single game at the start of each
row, so no game reads another game's state.

Training batches (``training_batches``) are a pure function of the seed, the
rank and the batch index, so worker processes can build them ahead of time,
exact resume needs only the step count, and the number of workers cannot change
what is trained on.
"""

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np
import torch

from moves.games import MAX_PLIES, legal_sets, random_game
from moves.objectives import LegalTargets, ValueTargets
from moves.vocab import GAME_START, MOVE_COUNT, PAD, VOCAB_SIZE, VOCABULARY_SHA256

OBJECTIVES = ('human', 'legal', 'engine')


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class SplitWriter:
    """Streams games of one split to ``{split}.bin`` and ``{split}_offsets.npy``."""

    def __init__(self, directory, split):
        self.directory, self.split = Path(directory), split
        self.stream = open(self.directory / f'{split}.bin', 'wb')
        self.offsets = [0]
        self.max_plies = 0

    def append(self, game):
        game = np.asarray(game, dtype=np.uint16)
        if not 1 <= len(game) <= MAX_PLIES:
            raise ValueError(f'A game must have between 1 and {MAX_PLIES} plies, not {len(game)}')
        self.stream.write(game.tobytes())
        self.offsets.append(self.offsets[-1] + len(game))
        self.max_plies = max(self.max_plies, len(game))

    def close(self):
        self.stream.close()
        np.save(self.directory / f'{self.split}_offsets.npy', np.asarray(self.offsets, dtype=np.int64))
        return dict(games=len(self.offsets) - 1, plies=self.offsets[-1], max_plies=self.max_plies,
                    tokens_sha256=file_hash(self.directory / f'{self.split}.bin'),
                    offsets_sha256=file_hash(self.directory / f'{self.split}_offsets.npy'))


@dataclass
class MoveBatch:
    x: torch.Tensor
    targets: object      # [batch, time] ply ids for 'human', else LegalTargets or ValueTargets
    counts: dict         # human_plies, random_plies, supervised_positions, row_tokens

    def to(self, device):
        if str(device).startswith('cuda'):
            return MoveBatch(self.x.pin_memory().to(device, non_blocking=True),
                             self.targets.pin_memory().to(device, non_blocking=True), self.counts)
        return MoveBatch(self.x.to(device), self.targets.to(device), self.counts)


class MoveData:
    def __init__(self, directory, context_length, verify_hashes=True, value_budget=None):
        directory = Path(directory)
        self.directory = directory
        self.manifest_hash = file_hash(directory / 'manifest.json')
        self.manifest = json.loads((directory / 'manifest.json').read_text())
        if self.manifest.get('kind') != 'moves':
            raise ValueError(f'{directory} is not a move-token dataset')
        if self.manifest['vocabulary_sha256'] != VOCABULARY_SHA256:
            raise ValueError('Dataset move vocabulary differs from moves.vocab')
        if context_length < self.manifest['max_plies']:
            raise ValueError('context_length must hold the longest game after its game-start token')
        self.context_length = context_length
        self.meta = dict(kind='moves', vocab_size=VOCAB_SIZE, output_size=MOVE_COUNT)
        self.tokens, self.offsets, self.values = {}, {}, {}
        for split, info in self.manifest['splits'].items():
            tokens, offsets = directory / f'{split}.bin', directory / f'{split}_offsets.npy'
            if verify_hashes and (file_hash(tokens) != info['tokens_sha256'] or
                                  file_hash(offsets) != info['offsets_sha256']):
                raise ValueError(f'{split}: file hash differs from the manifest')
            self.tokens[split] = np.memmap(tokens, dtype=np.uint16, mode='r')
            self.offsets[split] = np.load(offsets, mmap_mode='r')
            if len(self.offsets[split]) != info['games'] + 1 or self.offsets[split][-1] != len(self.tokens[split]):
                raise ValueError(f'{split}: offsets do not match the token array')
        self.value_budget = value_budget
        if value_budget is not None:
            values = self.manifest.get('values')
            if not values or value_budget not in values['budgets']:
                raise ValueError(f'Dataset has no engine labels for budget {value_budget!r}')
            for split, info in values['splits'].items():
                files = dict(legal_offsets=directory / f'{split}_legal_offsets.npy',
                             legal_moves=directory / f'{split}_legal_moves.bin',
                             labels=directory / f'{split}_labels_{value_budget}.npy')
                if verify_hashes and any(file_hash(path) != info['sha256'][path.name] for path in files.values()):
                    raise ValueError(f'{split}: engine label hash differs from the manifest')
                self.values[split] = dict(
                    legal_offsets=np.load(files['legal_offsets'], mmap_mode='r'),
                    legal_moves=np.memmap(files['legal_moves'], dtype=np.uint16, mode='r'),
                    q=np.load(files['labels'], mmap_mode='r')['q'])

    def __getstate__(self):
        # Worker processes reopen the memory-mapped files instead of copying them.
        return dict(directory=self.directory, context_length=self.context_length, value_budget=self.value_budget)

    def __setstate__(self, state):
        self.__init__(state['directory'], state['context_length'], verify_hashes=False,
                      value_budget=state['value_budget'])

    def game(self, split, index):
        offsets = self.offsets[split]
        return self.tokens[split][offsets[index]:offsets[index + 1]]

    def batch(self, split, batch_size, device, generator, *, objective, random_fraction=0.0,
              one_game_per_row=False):
        return self.build(split, batch_size, generator, objective=objective, random_fraction=random_fraction,
                          one_game_per_row=one_game_per_row).to(device)

    def build(self, split, batch_size, generator, *, objective, random_fraction=0.0, one_game_per_row=False):
        """A batch on the CPU; ``batch`` also moves it to a device."""
        if objective not in OBJECTIVES:
            raise ValueError(f'objective must be one of {OBJECTIVES}')
        if not 0 <= random_fraction <= 1:
            raise ValueError('random_fraction must be between zero and one')
        if objective == 'engine' and (random_fraction or split not in self.values):
            raise ValueError('The engine objective needs engine-labelled human games only')
        length = self.context_length + 1
        sequences = np.full((batch_size, length), PAD, dtype=np.int64)
        placed = []  # (row, start, ply ids, legal sets or None, game index or None)
        pending = None
        for row in range(batch_size):
            start = 0
            if random_fraction and float(torch.rand((), generator=generator)) < random_fraction:
                rng = random.Random(int(torch.randint(2 ** 62, (), generator=generator)))
                while length - start >= 2:
                    game, sets = random_game(rng, min(MAX_PLIES, length - start - 1))
                    placed.append((row, start, np.asarray(game, dtype=np.int64), sets, None))
                    start += len(game) + 1
                    if one_game_per_row:
                        break
                continue
            while True:
                if pending is None:
                    index = int(torch.randint(len(self.offsets[split]) - 1, (), generator=generator))
                    pending = (index, np.asarray(self.game(split, index), dtype=np.int64))
                index, game = pending
                if len(game) + 1 > length - start:
                    break
                placed.append((row, start, game, None, index))
                start += len(game) + 1
                pending = None
                if one_game_per_row:
                    break
        counts = dict(human_plies=0, random_plies=0, supervised_positions=0,
                      row_tokens=batch_size * self.context_length)
        for row, start, game, sets, index in placed:
            sequences[row, start] = GAME_START
            sequences[row, start + 1:start + 1 + len(game)] = game
            counts['human_plies' if index is not None else 'random_plies'] += len(game)
            counts['supervised_positions'] += len(game)
        x = torch.from_numpy(sequences[:, :-1].copy())
        if objective == 'human':
            y = sequences[:, 1:]
            targets = torch.from_numpy(np.where(y < MOVE_COUNT, y, -1))
        else:
            targets = self._sparse_targets(split, objective, placed)
        return MoveBatch(x, targets, counts)

    def _sparse_targets(self, split, objective, placed):
        positions, owner, moves, values = [], [], [], []
        for row, start, game, sets, index in placed:
            base = row * self.context_length + start
            if objective == 'legal':
                sets = sets if sets is not None else legal_sets(game)
            else:
                table = self.values[split]
                first = int(self.offsets[split][index])
                bounds = table['legal_offsets'][first:first + len(game) + 1]
            for ply in range(len(game)):
                owner_index = len(positions)
                positions.append(base + ply)
                if objective == 'legal':
                    legal = np.asarray(sets[ply], dtype=np.int64)
                else:
                    legal = np.asarray(table['legal_moves'][bounds[ply]:bounds[ply + 1]], dtype=np.int64)
                    values.append(np.asarray(table['q'][bounds[ply]:bounds[ply + 1]], dtype=np.float32))
                moves.append(legal)
                owner.append(np.full(len(legal), owner_index, dtype=np.int64))
        fields = dict(positions=torch.tensor(positions, dtype=torch.long),
                      owner=torch.from_numpy(np.concatenate(owner)),
                      moves=torch.from_numpy(np.concatenate(moves)))
        if objective == 'legal':
            return LegalTargets(**fields)
        return ValueTargets(**fields, values=torch.from_numpy(np.concatenate(values)))


def write_manifest(directory, manifest):
    """Write the manifest last: it marks the dataset as complete."""
    manifest = dict(format_version=1, kind='moves', vocabulary_sha256=VOCABULARY_SHA256,
                    move_count=MOVE_COUNT, game_start=GAME_START, pad=PAD, **manifest)
    manifest['max_plies'] = max(info['max_plies'] for info in manifest['splits'].values())
    (Path(directory) / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def batch_seed(seed, rank, index):
    """Seed of training batch ``index`` on ``rank``, independent of every other batch."""
    digest = hashlib.sha256(f'{seed}:{rank}:{index}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') >> 1


class _TrainingBatches(torch.utils.data.Dataset):
    """Training batch ``index`` of one rank, built on the CPU."""

    def __init__(self, data, batch_size, objective, random_fraction, seed, rank):
        self.data, self.batch_size, self.objective = data, batch_size, objective
        self.random_fraction, self.seed, self.rank = random_fraction, seed, rank

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(batch_seed(self.seed, self.rank, index))
        return self.data.build('train', self.batch_size, generator, objective=self.objective,
                               random_fraction=self.random_fraction)


def training_batches(data, batch_size, *, objective, random_fraction, seed, rank, start, workers=0, prefetch=4):
    """CPU training batches ``start, start + 1, ...`` in order; ``workers`` processes build them ahead."""
    dataset = _TrainingBatches(data, batch_size, objective, random_fraction, seed, rank)
    if not workers:
        return (dataset[index] for index in itertools.count(start))
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, sampler=itertools.count(start),
                                         num_workers=workers, prefetch_factor=prefetch)
    return iter(loader)
