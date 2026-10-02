# Stage 1: legality, full runs

**Question.** At the shallow end of the target range (predicting the legal moves, which needs the board and the rules but no calculation), which recurrence axis pays off? The plan expected temporal recurrence to do well and depth to add little ([plan](../../docs/engine_policy_plan.md)).

**Answer at this scale.** No axis pays off at the end. All four arms finish within 0.011 of each other on random games, inside the seed noise measured in the pilot. The recurrent arms learn faster in the middle of training, and the transformer catches up during its learning-rate decay. Extra passes help up to two to four passes and no further.

One seed per arm. Runs `full_legal_{transformer,temporal,depth,hybrid}` in [`pilot.py`](pilot.py), code at `f787962`.

## Setup

- **Data:** `stage1_full_v1`, 1.85B positions in 8,796,376 rows: 1.48B from Lichess games (`chess_8M_v1`, every game replayed, no parse failures, none truncated) and 0.37B from uniformly random legal games. One pass, the same rows in the same order for every arm. Two independent builds produced byte-identical files.
- **Target:** one sigmoid per move over all 1,968 moves; 1 for legal, 0 for illegal; final pass only.
- **Model:** 8 layers, width 512 (prelude 1, T-buffer 1, core 4, T-source 1, coda 1), context 256, untied output layer.
- **Training:** 21,990 updates of 400 rows; AdamW at 1e-3 for every arm, 3% warmup, flat, the last 10% decaying linearly to 1e-4. Initial weights seed 1337.
- **Update schedule** (largest update count U per update; U + 1 passes):

  | From | U = 0 | 1 | 2 | 3 | 4 |
  | --- | --- | --- | --- | --- | --- |
  | update 0 | 0.50 | 0.35 | 0.15 | 0 | 0 |
  | 25% | 0 | 0.55 | 0.30 | 0.15 | 0 |
  | 50% | 0 | 0.30 | 0.30 | 0.25 | 0.15 |
  | 75% to the end | 0 | 0.15 | 0.20 | 0.30 | 0.35 |

  Hybrid puts 80% of each U on (U, U). Temporal gets warm-start batches in the decay.
- **Evaluation** every 1,000 updates: training graph on 1,000 human dev games and 1,000 random dev games; live decoding on 100 of the human dev games, with the training graph scored on the same 100 games.

## Results

**At the end** (dev sets; exact set = the moves above 0.5 are exactly the legal set):

| Arm | Human exact set | Human separation | Random exact set | Random separation | Random legal / illegal loss |
| --- | --- | --- | --- | --- | --- |
| Transformer | **0.993** | 0.999 | **0.941** | 0.986 | 0.0100 / 0.0001 |
| Temporal | 0.991 | 0.998 | 0.930 | 0.984 | 0.0114 / 0.0001 |
| Depth | 0.992 | 0.999 | 0.940 | **0.987** | 0.0099 / 0.0001 |
| Hybrid | 0.991 | 0.999 | 0.937 | **0.987** | 0.0095 / 0.0001 |

The constant baseline is 0.080 (human) and 0.070 (random) nats; every arm is far below it. Precision and recall on random games are about 0.998 for every arm, so errors are rare single moves. Still, about 6% of random-game positions have at least one wrong move, against under 1% of human-game positions.

**During training,** random-game exact set (human-game exact set in brackets):

| Update | Transformer | Temporal | Depth | Hybrid |
| --- | --- | --- | --- | --- |
| 3,000 | 0.220 (0.767) | 0.216 (0.760) | **0.234** (0.763) | 0.181 (0.683) |
| 9,000 | 0.552 (0.918) | **0.644** (0.938) | 0.628 (0.929) | 0.555 (0.890) |
| 12,000 | 0.638 (0.929) | 0.703 (0.952) | 0.720 (0.951) | **0.733** (0.944) |
| 15,000 | 0.684 (0.931) | 0.757 (0.960) | **0.793** (0.964) | 0.761 (0.959) |
| 19,000, before the decay | 0.776 (0.968) | 0.772 (0.966) | **0.805** (0.970) | 0.769 (0.963) |
| 21,990, the end | **0.941** (0.993) | 0.930 (0.991) | 0.940 (0.992) | 0.937 (0.991) |

- **From about update 9,000 to the decay, the recurrent arms lead,** depth by up to 0.11 on random games.
- **The decay changes the picture:** it adds 0.14–0.17 on random games for every arm (transformer +0.165, temporal +0.158, depth +0.135, hybrid +0.168). Depth, which led before it, gains least.

**Passes,** live decoding on 100 human dev games (exact set), at the end:

| Arm | J = 1 | J = 2 | J = 4 | J = 5 | Training graph, same games |
| --- | --- | --- | --- | --- | --- |
| Temporal | 0.986 | — | — | — | 0.986 |
| Depth | 0.900 | 0.987 | 0.990 | 0.991 | 0.991 |
| Hybrid | 0.976 | 0.988 | 0.990 | 0.990 | 0.990 |

- **Live execution matches the training graph** at the trained depths, for all three recurrent arms.
- **Gains stop after two to four passes.**
- **J = 1 is out of distribution** for depth and hybrid: single-pass updates (U = 0) stopped after the first quarter. It is not a block-matched comparison with the transformer for these runs.

**Cost:** wall time with two runs sharing an A100 40 GB spot instance (Verda, $0.65/h):

| Arm | Training time | Median time per update | Time evaluating |
| --- | --- | --- | --- |
| Transformer | 2.9 h | 0.48 s | 0.5% |
| Hybrid | 4.3 h | 0.64 s | 2.4% |
| Depth | 5.6 h | 0.92 s | 2.0% |
| Temporal | 6.1 h | 0.99 s | 0.7% |

The recurrent arms cost 1.3–2× the transformer per update. Hybrid ran alone for its last 1.5 hours, which lowers its median. The comparison matches data, not compute. Building the dataset took 5 hours per 22-vCPU instance.

## What changed from the pilot, and why it mattered

- **Pass schedule.** The pilot sampled up to three updates from the first step (40% at U ≥ 2), and its recurrent arms left the early plateau late: updates 300–540, against 150 for the transformer. The transformer then led at every step. Here, with half the first quarter at U = 0 and the rest at most U = 2, temporal and depth keep pace from the start, hybrid catches up by update 9,000, and all three lead by the middle of training.
- **Learning rate.** Short checks (a fifth of a pilot run) favoured 3e-4 for temporal and depth. Full pilot runs reversed that: temporal ended at 0.931 against 0.834 human exact set, at 1e-3 against 3e-4. At that length the recurrent arms had barely left the plateau, so the checks measured how fast they leave it. Every arm here uses 1e-3.

## Limits

- **One seed per arm.** At pilot scale, two hybrid seeds ended 0.036 apart on human games and 0.136 apart on random games. The final spread here, 0.011 on random games, is far inside that. Whether the mid-training lead of the recurrent arms holds across seeds is not measured.
- **The learning rate was chosen on the transformer,** in a short check, and checked for the recurrent arms only at pilot scale.
- **The end point depends on the decay.** Every arm gained most of its last 0.15 on random games in the final 10% of updates. A longer stable phase or decay could change the ranking.
- **95% exact set on random games is not reached** by any arm (best 0.941).
- **Matched data, not compute** (see Cost). Temporal-only also has no configuration with the hybrid's block count at J = 4.

## Next

Stage 2 (`full_engine_{arm}`) continues each final model on 400M positions of Leela's search distribution, at the final update mixture above. The comparison of interest is whether the pattern changes at the deep end: the pilot's short engine stage put temporal ahead of the transformer there, with depth and hybrid behind.
