# Stage 9 continued pilot results

Historical record of the completed experiment. Paths below describe the retired experiment; recorded configurations, measurements, and raw artifact provenance retain their original meaning. Checkpoints and run scripts were retired; retained raw reports are in the adjacent results directory.

This is the controlled continuation of the initial Stage 9 pilot. The architecture and recurrence distribution were unchanged. The work added an exact validation-row evaluation, a feedback-content intervention, and two fresh 1,000-update hybrid runs with a 1,000-update learning-rate horizon. No grid expansion or live-feedback implementation was started.

## Decision

The longer pilot changes the provisional conclusion about repeated refinement: additional refinement becomes useful with training. At step 100, `(1,0)` remains the best or nearly best setting. By step 250, `(3,0)` is best in both seeds; by step 500, `(3,3)` is best in both; and at step 1,000 `(3,3)` is best in both exact-row evaluations.

At step 1,000, `(3,3)` improves over `(1,0)` by 0.01645 NLL for seed 1337 and 0.01599 for seed 1338. Depth-only `(0,3)` is still much weaker than temporal execution, but it has become modestly useful relative to `(0,0)`, improving NLL by 0.01257 and 0.01375. Repeated temporal updating `(3,0)` also improves over `(1,0)` by 0.00802 and 0.00871.

The compute tradeoff remains material. The evaluator's forward matrix-multiplication estimate for `(3,3)` is 1.8315 times `(1,0)` (and the estimate excludes normalization, nonlinearities, backward, and kernel overhead). The long-run result now supports investigating a compute-controlled refinement schedule or a simpler temporal-only restriction in a later stage, but does not justify changing this architecture or expanding the update grid yet.

## Exact 41-row evaluation

The existing retained checkpoints were evaluated using the canonical validation rows 0 through 40 exactly once, in 21 batches of at most two rows. Both recurrent seeds and the separately trained ordinary baseline used the same row order, batch size, dataset manifest, and batch fingerprint.

| Item | Value |
| --- | --- |
| Rows | 41 of 41 validation rows, each target row exactly once |
| Batches | 21; requested batch size 2; final batch has one row |
| Batch fingerprint | `06a6e2bf7e6897efbb4f8708be243aa4a6600aa06ae5ba68092b4af1ed615150` |
| Manifest | `36b7025c4331ec6f6fafc03fa0248ef90fc55f4ab2ecd69c763707a6ef56952d` |
| Recurrent reports | 0/25/50/75/100 for seeds 1337 and 1338, all nine pilot cells |
| Ordinary report | existing ordinary step-100 checkpoint, one-pass execution |

At the original step-100 checkpoint, the exact-row results are:

| checkpoint | `(0,0)` NLL | `(1,0)` NLL | `(0,3)` NLL | `(3,0)` NLL | `(3,3)` NLL | accuracy at `(1,0)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed 1337 | 1.797030 | 1.483078 | 1.800831 | 1.489655 | 1.484676 | 0.4844 |
| seed 1338 | 1.804646 | 1.505079 | 1.810056 | 1.509065 | 1.504632 | 0.4805 |
| ordinary baseline | 1.725761 | — | — | — | — | 0.3960 |

The exact-row comparison makes the two initial best settings effectively a near-tie: `(1,0)` wins seed 1337 by 0.00160 NLL, while `(3,3)` wins seed 1338 by 0.00045. This is why the longer unchanged pilot was warranted. The ordinary baseline is reported separately; `(0,0)` is ordinary execution using a hybrid checkpoint and is not a separately trained hybrid-free model.

Raw reports: exact recurrent/baseline JSON (historical path: `../../sweeps/recurrence_grid/results/all_rows.json`), exact recurrent/baseline CSV (historical path: `../../sweeps/recurrence_grid/results/all_rows.csv`). The report's nested placement results retain all 13 requested mask placements per recurrent cell.

## Feedback-content intervention at `(1,0)`

For each of the 41 validation rows, the normal run reads temporal source output produced from the same row. The intervention keeps the schedule, source block, temporal mixer, projections, and all weights unchanged, but swaps the first temporal source output across independent rows at the same sequence positions. The donor is the next validation row modulo 41. Target rows are still evaluated exactly once; donor rows are only reused as diagnostic inputs.

| training seed / checkpoint | normal NLL | corrupted-feedback NLL | ΔNLL | normal accuracy | corrupted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1337 / step 100 | 1.483078 | 2.708050 | +1.224973 | 0.4844 | 0.3076 |
| 1338 / step 100 | 1.505079 | 2.740726 | +1.235647 | 0.4805 | 0.3007 |

This large degradation indicates that the temporal path is using content aligned to the current row, not merely contributing an extra generic transformation. It remains a diagnostic intervention, not a definitive architecture comparison: swapping memory changes the model's inputs to the same computation and is not a separately trained condition.

Raw reports: feedback diagnostic JSON (historical path: `../../sweeps/recurrence_grid/results/feedback_diagnostic.json`), per-row feedback CSV (historical path: `../../sweeps/recurrence_grid/results/feedback_diagnostic.csv`).

## Long unchanged hybrid pilot

The fresh runs used `experiments/pilots/recurrence/config.py` with the same eight-block width-512 architecture, batch/accumulation settings, recurrence support and probabilities, MPS float32 eager execution, ten-step warmup, and seeds 1337/1338 with recurrence seeds 1729/1730. Only `max_iters` and `lr_decay_iters` changed from the initial pilot, both to 1,000. Training logged evaluations every 100 updates and retained only steps 0, 100, 250, 500, and 1,000.

The run metadata is retained in seed 1337 `run.json` (historical path: `results/seed1337/run.json`) and seed 1338 `run.json` (historical path: `results/seed1338/run.json`). Their provenance marks the tree dirty because the exact-step checkpoint hook and evaluation-only tools were added for this continuation; the model architecture and optimizer/distribution settings remained fixed.

| seed | step 0 val NLL | step 100 | step 250 | step 500 | step 1000 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1337 | 3.7166 | 1.4307 | 1.3225 | 1.0428 | 0.8773 |
| 1338 | 3.7498 | 1.4356 | 1.3280 | 1.1293 | 0.9161 |

These are the trainer's four-batch validation logs. The execution-setting comparisons below use the stronger exact 41-row reports.

### Seed 1337: exact-row NLL

| updates | `(0,0)` | `(1,0)` | `(0,3)` | `(3,0)` | `(3,3)` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.574016 | 3.737708 | 3.724115 | 3.733579 | 3.710067 |
| 100 | 1.708893 | 1.430050 | 1.714170 | 1.434534 | 1.440246 |
| 250 | 1.551865 | 1.304074 | 1.554095 | 1.281880 | 1.287688 |
| 500 | 1.286675 | 1.089950 | 1.275164 | 1.060193 | 1.053586 |
| 1000 | 0.999586 | 0.914070 | 0.987021 | 0.906046 | 0.897618 |

### Seed 1338: exact-row NLL

| updates | `(0,0)` | `(1,0)` | `(0,3)` | `(3,0)` | `(3,3)` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.645696 | 3.762676 | 3.763278 | 3.755164 | 3.754421 |
| 100 | 1.784407 | 1.437812 | 1.748941 | 1.438069 | 1.435615 |
| 250 | 1.537909 | 1.303362 | 1.543075 | 1.278540 | 1.285100 |
| 500 | 1.320490 | 1.124979 | 1.291350 | 1.102851 | 1.096010 |
| 1000 | 1.002968 | 0.920795 | 0.989217 | 0.912087 | 0.904808 |

At step 1,000 the NLL differences relative to `(0,0)` are:

| seed | `(1,0)` | `(0,3)` | `(3,0)` | `(3,3)` |
| ---: | ---: | ---: | ---: | ---: |
| 1337 | -0.085516 | -0.012565 | -0.093540 | -0.101968 |
| 1338 | -0.082173 | -0.013751 | -0.090881 | -0.098159 |

The raw long-run report and flat table are exact 41-row JSON (historical path: `results/all_rows.json`) and CSV (historical path: `results/all_rows.csv`). Derived tables are per-seed curves (historical path: `results/curves/grid-curves-by-seed.csv`), seed-summary curves (historical path: `results/curves/grid-curves-seed-summary.csv`), selected comparisons (historical path: `results/curves/selected-comparisons.csv`), and trainer evaluations (historical path: `results/curves/training-evaluations.csv`).

The retained final checkpoints are seed 1337 step 1000 (historical path: `results/seed1337/ckpt-step001000.pt`) and seed 1338 step 1000 (historical path: `results/seed1338/ckpt-step001000.pt`); the same directories contain steps 0/100/250/500.

!Long-pilot exact-row NLL curves (historical path: `results/curves/nll-vs-updates.png`)

!Long-pilot differences versus one temporal update (historical path: `results/curves/delta-nll-vs-10.png`)

### Estimated forward matrix multiplications

The structural estimate is constant across checkpoints because it depends on the architecture, sequence length, and schedule, not learned weights.

| cell | estimated forward matmul count per sequence | ratio versus `(1,0)` |
| ---: | ---: | ---: |
| `(0,0)` | 68,669,128,704 | 0.5982 |
| `(1,0)` | 114,784,462,848 | 1.0000 |
| `(0,3)` | 174,840,619,008 | 1.5223 |
| `(3,0)` | 207,015,131,136 | 1.8035 |
| `(3,3)` | 210,233,210,880 | 1.8315 |

These are forward matrix-multiplication estimates only. They are not measured hardware FLOPs and do not include backward or several other costs.

## Diagnostics and scope

All exact-row reports and long-run trainer evaluations were finite. The feedback intervention used eval-mode forward passes only. It does not demonstrate that every stored feature is semantically useful, but the strong matched-versus-swapped gap supports the claim that row-aligned temporal content matters.

One initial seed-1337 long-run attempt reached step 300 before I corrected checkpoint timing; it is preserved as `experiments/pilots/recurrence/results/partial-seed1337-step300` (historical path: `results/partial-seed1337-step300`) and excluded from every result above because it lacked the requested step-250 snapshot. The clean restart reproduced its trajectory and completed all milestones.

No architecture changes, grid expansion, hyperparameter search, component-baseline training, or live-feedback implementation were performed. The next justified experiment is a compute-controlled comparison focused on whether the late `(3,3)` gain is worth its approximately 1.83× forward estimate relative to `(1,0)`; that is outside this continuation.
