# Post-training evaluation battery

This battery answers the questions the 20B study needs after training, using only evaluation:

1. Does the hybrid beat each single-axis arm, and by how much?
2. Is the hybrid still better at equal *inference* cost?
3. Can one hybrid checkpoint stand in for each single-axis model?
4. Where does the gain come from: which characters, how far into a game, and how often the model writes legal moves?

It does not measure playing strength.

The [5B dry run](#5b-dry-run) exercises every stage on the completed 5B arms.

## Tests

| Test | Question | Tool |
| --- | --- | --- |
| **Training-graph NLL**, all confirmation rows | The main result: hybrid `(3,3)` against temporal `(3,0)`, depth `(0,3)` and the transformer. Every other comparison is paired on the same targets. | `evaluation.position_losses` |
| **Test-depth cells** 1/2/4/8/16 passes | How each arm uses extra passes it never trained on | same |
| **Hybrid at single-axis and asymmetric cells** `(3,0)`, `(0,3)`, `(3,1)`, `(1,3)`, `(1,0)`, `(0,1)` | Can one hybrid run replace the dedicated temporal and depth arms? | same |
| **Live NLL**: temporal J=1, hybrid J=1/2/4/8, depth J=4, transformer | Deployed, token-by-token execution. Hybrid J=1 against temporal J=1 compares equal inference cost. Also the gap between live and training-graph execution. Depth J=4 must equal its `(0,3)` cell. | `evaluation.live_legality` |
| **Legal-move probability** at every move start | Probability mass on legal moves (with the correct `+`/`#`), how often the greedy move is legal, and how often the top legal move is the move played | same |
| **Stratified NLL**: character class × ply, row position, game ordinal | Is the gain in move choice (`move_first`) or in board-determined characters (`check_slot`: whether `+`/`#` follows)? Does it grow later in the game? | `evaluation.compare_losses` |
| **Same-host cost**: seconds per update for each arm's late phase and whole curriculum; live ms/token | Is the second axis free to train? What does each deployed setting cost? | `experiments.benchmarks.arm_cost` |

Every difference is reported as `A − B` with a 95% bootstrap interval that resamples validation rows. **The interval reflects which rows were evaluated, not seed-to-seed variation**; only extra training seeds measure that.

### Character classes

A target is classified by the role of the character it predicts (see `evaluation/pgn_annotations.py`):

| Class | Meaning |
| --- | --- |
| `move_number` | Digits and `.` of a move number. Predictable from the move count. |
| `move_first` | First SAN character. The move choice starts here. |
| `move_body` | The rest of the SAN: disambiguation, captures, the destination square, promotions |
| `check_slot` | The character right after the SAN: `+`/`#`, or a delimiter. Whether it is a check marker depends only on the board. |
| `delimiter` | The delimiter after `+`/`#` |

### Legal-move probability

The model is teacher-forced along the real game. At each move start, the live state is branched over every legal SAN from the python-chess replay. The probability of each `SAN + delimiter` is then computed exactly by walking the SAN trie.

Branches share the teacher-forced prefix's KV caches without copying them (`inference/branch.py`). Tests check them against ordinary live decoding in every mode and cache strategy. Moves too close to the 1,023-character limit to branch every legal move are skipped and counted.

## Running on the 20B checkpoints

The full battery, with every confirmation row, 1,000 live rows and up to 16 passes:

```sh
uv run python -m experiments.evaluation_battery.run --preset 20B \
    --tg-device cuda --live-device cpu --live-workers <cores> --live-limit 1000
uv run python -m experiments.benchmarks.arm_cost --device cuda --output <path>.json   # on one 4090
```

`--step 175954` evaluates the 18B pre-decay checkpoints instead. That has not been run.

### The run behind the 20B report

The [20B report](../long_runs/20B_recurrence/REPORT.md) uses a reduced run on one Secure RTX 4090. It also includes the post-hoc live-aligned checkpoints from [`align.py`](../ablations/live_warm_start/align.py):

```sh
ARGS="--preset 20B --aligned-dir experiments/ablations/live_warm_start/results \
  --output-root experiments/evaluation_battery/results/20B-final-aligned-live400 \
  --tg-limit 8192 --max-passes 8 --tg-device cuda --tg-batch-size 8 \
  --live-limit 400 --max-live-depth 4 --live-device cuda --live-workers 16 --live-threads 1"
uv run python -m experiments.evaluation_battery.run $ARGS --stages training_graph
uv run python -m experiments.evaluation_battery.run $ARGS --stages live
uv run python -m experiments.evaluation_battery.run $ARGS --stages compare
```

- **Rows:** 8,192 training-graph rows and 400 live rows.
- **Skipped:** the 16-pass cells and eight-iteration live execution, which the full battery would take hours longer to run.
- **Parallel stages:** the training-graph and live stages ran at the same time.
- **Aligned checkpoints:** the ones evaluated were later re-exported without resume state. The weights are unchanged, but the file hashes are not, so a rerun needs a new `--output-root`.

- **Resuming:** each stage skips outputs that already exist, so rerunning the command resumes it. Live shards also resume from their own partial files.
- **Settings are fixed per output directory:** `battery.json` records the settings, and a rerun with different settings is refused.
- **Rows:** by default, the training graph uses every confirmation row of the 20B panel (82,675 rows); `--tg-limit` takes the first rows of a seeded permutation instead. Live evaluation uses the first `--live-limit` rows of the same permutation, so every live row also has training-graph losses.
- **Outputs:** results go to `experiments/evaluation_battery/results/<preset>-step<step>/`. That folder holds per-arm `cell_*.npz` and `live_J*.shard*.npz` files, plus `compare_training_graph.{json,md}` and `compare_live.{json,md}`.

Run the cost benchmark on the same host type as the arms: all four arms together, one process, one GPU.

## 5B dry run

```sh
uv run python -m experiments.evaluation_battery.run --preset 5B --tg-limit 1000 --live-limit 24 \
    --max-passes 8 --max-live-depth 4 --tg-device mps --live-device cpu --live-workers 6
```

- **Stand-in transformer:** the 5B transformer arm was never completed, so the dry run uses the 20B transformer at 4B characters (step 39,101). That transformer has a different learning-rate schedule and horizon, so its comparisons check the tooling only.
- **Sample sizes:** the dry run uses 1,000 training-graph rows and 24 live rows.
- **Results:** see `results/5B-dry-run/`.
