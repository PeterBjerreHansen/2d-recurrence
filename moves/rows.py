"""Fixed move-row datasets: one format for every stage (docs/engine_policy_plan.md, Datasets).

A dataset directory holds ``dataset.json`` and, per split, fixed rows of packed
games with a sparse move list for every position:

``{split}_tokens.bin``   uint16 [rows, context + 1]: game start, plies, next game, ..., padding
``{split}_counts.bin``   uint8 [rows, context]: moves listed for each position (0: no target)
``{split}_moves.bin``    uint16: the listed move ids, row after row, position after position
``{split}_rows.npy``     int64 [rows + 1]: where each row's moves start
``{split}_plies.npy``    int32 [rows, sources]: plies of each source in each row
``{split}_weights.bin``  float16, per listed move (policy datasets)
``{split}_values.bin``   float16 [rows, context, 2]: position value q (win minus loss), d (draw) (policy datasets)
``{split}_visits.bin``   uint16 [rows, context]: search visits (policy datasets)

Position ``t`` of a row is the board after token ``t``; its next-ply target is
token ``t + 1``. Builders produce games as ``GameRecord``s; ``pack`` writes them
into rows in the given order, and ``RowData`` reads rows back. No chess code runs
when reading.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_loader import file_hash
from moves.games import MAX_PLIES
from moves.objectives import LegalTargets, PolicyTargets
from moves.vocab import GAME_START, MOVE_COUNT, PAD, VOCAB_SIZE, VOCABULARY_SHA256

TARGETS = ('legal', 'policy')
OBJECTIVES = ('human', 'legal', 'engine')


@dataclass
class MoveBatch:
    x: torch.Tensor
    targets: object      # [batch, time] ply ids for 'human', else LegalTargets or PolicyTargets
    counts: dict         # plies per source, supervised positions, row tokens

    def to(self, device):
        if str(device).startswith('cuda'):
            return MoveBatch(self.x.pin_memory().to(device, non_blocking=True),
                             self.targets.pin_memory().to(device, non_blocking=True), self.counts)
        return MoveBatch(self.x.to(device), self.targets.to(device), self.counts)


@dataclass
class GameRecord:
    source: int            # index into the dataset's sources
    plies: np.ndarray      # [n] uint16 ply ids
    counts: np.ndarray     # [n] uint8 moves listed for the position before each ply
    moves: np.ndarray      # [sum(counts)] uint16
    weights: np.ndarray | None = None   # [sum(counts)] float16
    values: np.ndarray | None = None    # [n, 2] float16
    visits: np.ndarray | None = None    # [n] uint16


def game_hash(plies):
    return hashlib.sha256(np.asarray(plies, dtype=np.uint16).tobytes()).digest()


def split_of(plies, test_per_mille=5, dev_per_mille=5):
    """Split of a game from the sha256 of its ply ids: the first 8 bytes, little-endian, modulo 1000."""
    bucket = int.from_bytes(game_hash(plies)[:8], 'little') % 1000
    return 'test' if bucket < test_per_mille else 'dev' if bucket < test_per_mille + dev_per_mille else 'train'


class _Streams:
    """Append-only binary files for one split, hashed on close."""

    def __init__(self, directory, split, names):
        self.paths = {name: Path(directory) / f'{split}_{name}.bin' for name in names}
        self.files = {name: open(path, 'wb') for name, path in self.paths.items()}

    def write(self, name, array):
        self.files[name].write(np.ascontiguousarray(array).tobytes())

    def close(self):
        for stream in self.files.values():
            stream.close()
        return {path.name: file_hash(path) for path in self.paths.values()}


def pack(directory, split, games, *, context, sources, policy, one_game_per_row=False):
    """Pack ``games`` (GameRecords, in order) into rows; return the split's manifest entry.

    Games never cross rows: a game that doesn't fit starts a new row, and row
    tails are padded. With ``one_game_per_row`` each row holds one game.
    """
    length = context + 1
    names = ['tokens', 'counts', 'moves'] + (['weights', 'values', 'visits'] if policy else [])
    streams = _Streams(directory, split, names)
    row_offsets, row_plies = [0], []
    info = dict(games=0, positions=0, moves=0, max_plies=0)
    state = {}

    def new_row():
        state.update(tokens=np.full(length, PAD, dtype=np.uint16), counts=np.zeros(context, dtype=np.uint8),
                     plies=np.zeros(len(sources), dtype=np.int32), start=0, moves=[], weights=[],
                     values=np.zeros((context, 2), dtype=np.float16), visits=np.zeros(context, dtype=np.uint16))

    def flush():
        if state['start'] == 0:
            return
        streams.write('tokens', state['tokens'])
        streams.write('counts', state['counts'])
        moves = np.concatenate(state['moves']) if state['moves'] else np.zeros(0, dtype=np.uint16)
        streams.write('moves', moves)
        if policy:
            streams.write('weights', np.concatenate(state['weights']))
            streams.write('values', state['values'])
            streams.write('visits', state['visits'])
        row_offsets.append(row_offsets[-1] + len(moves))
        row_plies.append(state['plies'])
        new_row()

    new_row()
    for game in games:
        n = len(game.plies)
        if not 1 <= n <= min(MAX_PLIES, context):
            raise ValueError(f'A game must have between 1 and {min(MAX_PLIES, context)} plies, not {n}')
        if len(game.counts) != n or int(game.counts.sum()) != len(game.moves):
            raise ValueError('A game needs one move count per ply and exactly that many moves')
        if policy and (game.weights is None or game.values is None or game.visits is None):
            raise ValueError('Policy datasets need weights, values and visits for every game')
        if n + 1 > length - state['start']:
            flush()
        start = state['start']
        state['tokens'][start] = GAME_START
        state['tokens'][start + 1:start + 1 + n] = game.plies
        state['counts'][start:start + n] = game.counts
        state['moves'].append(np.asarray(game.moves, dtype=np.uint16))
        if policy:
            state['weights'].append(np.asarray(game.weights, dtype=np.float16))
            state['values'][start:start + n] = game.values
            state['visits'][start:start + n] = game.visits
        state['plies'][game.source] += n
        state['start'] = start + n + 1
        info['games'] += 1
        info['positions'] += n
        info['moves'] += len(game.moves)
        info['max_plies'] = max(info['max_plies'], n)
        if one_game_per_row:
            flush()
    flush()
    hashes = streams.close()
    np.save(Path(directory) / f'{split}_rows.npy', np.asarray(row_offsets, dtype=np.int64))
    np.save(Path(directory) / f'{split}_plies.npy',
            np.asarray(row_plies, dtype=np.int32).reshape(-1, len(sources)))
    for name in ('rows', 'plies'):
        hashes[f'{split}_{name}.npy'] = file_hash(Path(directory) / f'{split}_{name}.npy')
    return dict(rows=len(row_offsets) - 1, one_game_per_row=one_game_per_row, sha256=hashes, **info)


def write_dataset_manifest(directory, *, context, sources, targets, splits, eval_splits, **provenance):
    """Write ``dataset.json`` last: it marks the dataset as complete."""
    if targets not in TARGETS:
        raise ValueError(f'targets must be one of {TARGETS}')
    manifest = dict(format_version=1, kind='move_rows', vocabulary_sha256=VOCABULARY_SHA256, context=context,
                    sources=list(sources), targets=targets, splits=splits, eval_splits=list(eval_splits),
                    **provenance)
    (Path(directory) / 'dataset.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


class GameStoreWriter:
    """Game-major storage of ``GameRecord``s, for data that must be converted before it can be packed."""

    def __init__(self, directory, policy):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        names = ['plies', 'counts', 'moves'] + (['weights', 'values', 'visits'] if policy else [])
        self.streams = _Streams(self.directory, 'store', names)
        self.game_offsets, self.move_offsets = [0], [0]

    def append(self, game):
        self.streams.write('plies', np.asarray(game.plies, dtype=np.uint16))
        self.streams.write('counts', np.asarray(game.counts, dtype=np.uint8))
        self.streams.write('moves', np.asarray(game.moves, dtype=np.uint16))
        if self.policy:
            self.streams.write('weights', np.asarray(game.weights, dtype=np.float16))
            self.streams.write('values', np.asarray(game.values, dtype=np.float16))
            self.streams.write('visits', np.asarray(game.visits, dtype=np.uint16))
        self.game_offsets.append(self.game_offsets[-1] + len(game.plies))
        self.move_offsets.append(self.move_offsets[-1] + len(game.moves))

    def close(self):
        self.streams.close()
        np.save(self.directory / 'store_games.npy', np.asarray(self.game_offsets, dtype=np.int64))
        np.save(self.directory / 'store_move_offsets.npy', np.asarray(self.move_offsets, dtype=np.int64))
        return len(self.game_offsets) - 1


def _memmap(path, dtype):
    # An empty file can't be memory-mapped; a split with no games has empty files.
    return np.memmap(path, dtype=dtype, mode='r') if path.stat().st_size else np.zeros(0, dtype=dtype)


class GameStore:
    """Random access to the games of a ``GameStoreWriter`` directory."""

    def __init__(self, directory):
        directory = Path(directory)
        self.policy = (directory / 'store_weights.bin').exists()
        self.games = np.load(directory / 'store_games.npy')
        self.move_offsets = np.load(directory / 'store_move_offsets.npy')
        self.plies = _memmap(directory / 'store_plies.bin', np.uint16)
        self.counts = _memmap(directory / 'store_counts.bin', np.uint8)
        self.moves = _memmap(directory / 'store_moves.bin', np.uint16)
        if self.policy:
            self.weights = _memmap(directory / 'store_weights.bin', np.float16)
            self.values = _memmap(directory / 'store_values.bin', np.float16).reshape(-1, 2)
            self.visits = _memmap(directory / 'store_visits.bin', np.uint16)

    def __len__(self):
        return len(self.games) - 1

    def record(self, index, source):
        a, b = self.games[index], self.games[index + 1]
        m, n = self.move_offsets[index], self.move_offsets[index + 1]
        if not self.policy:
            return GameRecord(source, np.array(self.plies[a:b]), np.array(self.counts[a:b]), np.array(self.moves[m:n]))
        return GameRecord(source, np.array(self.plies[a:b]), np.array(self.counts[a:b]), np.array(self.moves[m:n]),
                          np.array(self.weights[m:n]), np.array(self.values[a:b]), np.array(self.visits[a:b]))


class RowData:
    """Reads a move-row dataset. Training rows are read in stored order, then fixed permutations."""

    def __init__(self, directory, context_length, verify_hashes=True):
        directory = Path(directory)
        self.directory = directory
        self.manifest_hash = file_hash(directory / 'dataset.json')
        self.manifest = json.loads((directory / 'dataset.json').read_text())
        if self.manifest.get('kind') != 'move_rows':
            raise ValueError(f'{directory} is not a move-row dataset')
        if self.manifest['vocabulary_sha256'] != VOCABULARY_SHA256:
            raise ValueError('Dataset move vocabulary differs from moves.vocab')
        if context_length != self.manifest['context']:
            raise ValueError(f"The dataset was packed for context {self.manifest['context']}, not {context_length}")
        self.context_length = context_length
        self.sources = tuple(self.manifest['sources'])
        self.targets = self.manifest['targets']
        self.eval_splits = tuple(self.manifest['eval_splits'])
        self.count_keys = tuple(f'{source}_plies' for source in self.sources) + ('supervised_positions', 'row_tokens')
        self.meta = dict(kind='move_rows', vocab_size=VOCAB_SIZE, output_size=MOVE_COUNT)
        self._arrays = {}
        self._permutations = {}
        if verify_hashes:
            for split, info in self.manifest['splits'].items():
                for name, digest in info['sha256'].items():
                    if file_hash(directory / name) != digest:
                        raise ValueError(f'{name}: file hash differs from the manifest')

    def rows(self, split):
        return self.manifest['splits'][split]['rows']

    def _split(self, split):
        if split not in self._arrays:
            d, t = self.directory, self.context_length
            arrays = dict(tokens=np.memmap(d / f'{split}_tokens.bin', dtype=np.uint16, mode='r').reshape(-1, t + 1),
                          counts=np.memmap(d / f'{split}_counts.bin', dtype=np.uint8, mode='r').reshape(-1, t),
                          moves=np.memmap(d / f'{split}_moves.bin', dtype=np.uint16, mode='r'),
                          offsets=np.load(d / f'{split}_rows.npy'), plies=np.load(d / f'{split}_plies.npy'))
            if self.targets == 'policy':
                arrays.update(
                    weights=np.memmap(d / f'{split}_weights.bin', dtype=np.float16, mode='r'),
                    visits=np.memmap(d / f'{split}_visits.bin', dtype=np.uint16, mode='r').reshape(-1, t))
            self._arrays[split] = arrays
        return self._arrays[split]

    def training_rows(self, first, count):
        """Rows ``first .. first + count`` of the training order.

        Pass 0 is the stored order (shuffled when the dataset was built); pass ``k``
        is a fixed permutation seeded by ``k``.
        """
        total = self.rows('train')
        indices = np.arange(first, first + count)
        passes, within = indices // total, indices % total
        rows = np.empty(count, dtype=np.int64)
        for number in np.unique(passes):
            if number == 0:
                order = None
            else:
                if number not in self._permutations:
                    self._permutations[number] = np.random.default_rng(int(number)).permutation(total)
                order = self._permutations[number]
            select = passes == number
            rows[select] = within[select] if order is None else order[within[select]]
        return rows

    def build(self, split, rows, *, objective):
        """A CPU ``MoveBatch`` of the given rows."""
        if objective not in OBJECTIVES:
            raise ValueError(f'objective must be one of {OBJECTIVES}')
        if (objective == 'legal') != (self.targets == 'legal') and objective != 'human':
            raise ValueError(f"A '{self.targets}' dataset can't train the {objective} objective")
        arrays = self._split(split)
        rows = np.asarray(rows, dtype=np.int64)
        tokens = np.asarray(arrays['tokens'][rows], dtype=np.int64)
        x, y = tokens[:, :-1], tokens[:, 1:]
        counts = dict(zip(self.count_keys, [int(v) for v in arrays['plies'][rows].sum(0)] +
                          [int((y < MOVE_COUNT).sum()), len(rows) * self.context_length]))
        if objective == 'human':
            return MoveBatch(torch.from_numpy(x.copy()), torch.from_numpy(np.where(y < MOVE_COUNT, y, -1)), counts)
        per_position = np.asarray(arrays['counts'][rows], dtype=np.int64)
        supervised = per_position > 0
        positions = np.flatnonzero(supervised.reshape(-1))
        spans = [(arrays['offsets'][row], arrays['offsets'][row + 1]) for row in rows]
        moves = np.concatenate([arrays['moves'][a:b] for a, b in spans]).astype(np.int64)
        owner = np.repeat(np.arange(len(positions)), per_position.reshape(-1)[positions])
        fields = dict(positions=torch.from_numpy(positions), owner=torch.from_numpy(owner),
                      moves=torch.from_numpy(moves))
        if objective == 'legal':
            return MoveBatch(torch.from_numpy(x.copy()), LegalTargets(**fields), counts)
        weights = np.concatenate([arrays['weights'][a:b] for a, b in spans]).astype(np.float32)
        # Stored at float16; renormalise so each position's weights sum to one.
        totals = np.zeros(len(positions))
        np.add.at(totals, owner, weights)
        return MoveBatch(torch.from_numpy(x.copy()),
                         PolicyTargets(**fields, weights=torch.from_numpy((weights / totals[owner]).astype(np.float32))),
                         counts)

    def batch(self, split, rows, device, **options):
        return self.build(split, rows, **options).to(device)

