# Baseline selection and continuation results

Historical record of the completed experiment. Paths below follow the current repository layout; recorded configurations, measurements, and raw artifact provenance retain their original meaning. See the experiment README for current commands.

Status: complete on 2026-09-18. This report covers only the authorized protocol: two 1,000-update learning-rate candidates and one selected continuation to 10,000 total updates. No extra seeds, sweep points, objectives, architecture changes, or cloud resources were used.

## Result in brief

Candidate A (peak learning rate `3e-4`) won the predeclared rule and continued to 10,000 updates. Both candidates were eligible: all trained-support grid evaluations were finite. Candidate A had the lower selection score and weighted grid score, so B was not continued.

After the longer run, additional recurrence was modestly useful in this fixed architecture and data regime. On the frozen 1,317-row confirmation panel, `(3,3)` beat `(1,0)` by 0.00553 NLL at 5,000 updates and 0.00760 NLL at 10,000 updates. The same direction held on the 128-row selection panel. This is an observed validation effect, not evidence of reasoning. `(3,3)` costs about 1.832 times the estimated forward matrix-multiply work of `(1,0)`; the estimate excludes backward work, normalization, elementwise operations, and hardware overhead.

All recurrence evaluation and generation in this report uses the exact parallel training graph. It is not live-feedback generation; the latter remains a later-stage capability. Mask-placement variation is reported separately from training variation. A and B are matched learning-rate candidates with the same model seed (`1337`) and schedule seed (`1729`), not independent training seeds.

## Protocol and provenance

The run used the current width-512 recurrent model: 2 prelude blocks, 4 shared core blocks, 1 temporal source block, and 1 coda block; context 1,023; 32-character vocabulary; AdamW (`betas=(0.9, 0.95)`, weight decay 0.1); gradient clip 1; microbatch 2 with accumulation 4; final-only loss; zero dropout; fixed update support `{0,1,3}` and fixed update-probability matrix; MPS, float32, eager execution.

The full pinned dataset contains 143,017 training rows / 146,449,408 characters and 1,445 validation rows / 1,479,680 characters. It was prepared uncapped from `lichess_100mb_blocks.zip`, dataset revision `1a932e1abca935aae585f417ede39ecde4f2a620`. The archive SHA-256 is `e8dca0c6ef531e9dc080b82a15f9cd24ee102c23b460c567fb731f83787d4b54`; the dataset manifest SHA-256 is `a517b15768ea375e8e38d6408820cde4017e362f1c9b5f755f65b74a4ba6eee5`.

The evaluation panel is frozen in [`panels.json`](results/panels.json), SHA-256 `8860782ceee0acf48ea554174c59fda042e63d667ecc7cbdd7da840264f91cea`. It contains 128 selection rows chosen with `random.Random(2027)` and the sorted 1,317-row complement. Every fixed-panel report records row indices, panel hash, dataset identity, batch fingerprint, and target count; the final selection target count is 130,944 and the final confirmation target count is 1,347,291.

Training code/config provenance was frozen before training in [`provenance.json`](results/preflight/provenance.json) and [`training-provenance.patch`](results/preflight/training-provenance.patch). The frozen base commit was `b43e2d22baaa68798f74c39b3a67a56e1aced9b4`; the complete patch SHA-256 is `09ab859753832e5106d62e9e0e8bebc189a2d0d5ec677cccbf6b1b445b0bcc53`. The final test suite passed (`71 passed`). The local environment reported PyTorch `2.14.0`, NumPy `2.4.6`, `chess 1.11.2`, and MPS available/built.

The existing uncommitted work was preserved before changes and restored after preservation. Earlier stage artifacts were retained; no prior result was overwritten. The training logs record 8,184 target characters per optimizer update.

## Candidate selection

The rule used `S`, mean `(3,3)` NLL at updates 750 and 1,000, and `W`, the same-checkpoint NLL weighted by the committed training update-probability matrix. B could win only if all three declared conditions held.

| candidate | peak / minimum LR | S | W | eligible | decision inputs |
| --- | ---: | ---: | ---: | --- | --- |
| A | `3e-4 / 3e-5` | 0.86028 | 0.88862 | yes | `(3,3)`: 0.90503 / 0.81554 |
| B | `1e-4 / 1e-5` | 1.02002 | 1.07935 | yes | `(3,3)`: 1.05692 / 0.98312 |

B failed the S margin, the `(3,3)` no-worse condition, and the W tolerance. The mechanical result is recorded in [`selection.json`](results/selection.json): **A selected by default rule**. Selection used only candidate selection-panel results; confirmation and generation were not used to override it.

The candidate selection curves are in [`selection-nll-vs-updates.png`](results/selection-nll-vs-updates.png) and [`selection-grid-curves.csv`](results/selection-grid-curves.csv). The complete raw nine-cell reports are retained under [`experiments/sweeps/baseline_lr_selection/results/lr3e-4`](results/lr3e-4) and [`experiments/sweeps/baseline_lr_selection/results/lr1e-4`](results/lr1e-4).

## Training and clipping

Candidate A ran 10,000 optimizer updates: 1,000 for selection and 9,000 resumed updates for the continuation. Candidate B ran 1,000 updates. Thus the authorized total was exactly 11,000 optimizer updates, processing 90,024,000 target characters across both candidates; the selected run processed 81,840,000 characters.

The logged per-update `seconds` values sum to 10,790.7 seconds for A and 1,054.6 seconds for B. These are optimizer-update compute times and exclude periodic validation and report-generation overhead. The continuation preserved optimizer moments, sampler/RNG state, warmup state, and the cosine schedule; A reached its configured minimum-LR neighborhood only at the end of the 10,000-update horizon.

| run | updates | clipped updates | clipped fraction | mean pre-clip norm | p95 | maximum | mean applied coefficient |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A, 0–10k | 10,000 | 487 | 4.87% | 0.42193 | 0.98207 | 14.48377 | 0.98281 |
| B, 0–1k | 1,000 | 972 | 97.20% | 2.31762 | 4.15543 | 15.07213 | 0.52285 |

These are aggregate accumulated-update statistics after four microbatches. They are not per-cell clipping probabilities and should not be interpreted as proof that any recurrence path was ignored. Raw metrics and the machine-readable summary are [`metrics.jsonl`](results/lr3e-4/metrics.jsonl), [`metrics.jsonl`](results/lr1e-4/metrics.jsonl), and [`clipping-summary.json`](results/clipping-summary.json).

## Selected learning curves

The ordinary execution `(0,0)`, temporal-only execution `(1,0)`, depth-only execution `(0,3)`, and repeated hybrid `(3,3)` provide the most compact comparison. Values below are placement means on the frozen 128-row selection panel.

| updates | `(0,0)` ordinary | `(1,0)` temporal-only | `(0,3)` depth-only | `(3,0)` repeated temporal | `(3,3)` hybrid |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.57122 | 3.73668 | 3.73140 | 3.73140 | 3.71223 |
| 250 | 1.61769 | 1.31469 | 1.61766 | 1.27662 | 1.26987 |
| 500 | 1.31520 | 1.09017 | 1.29084 | 1.05882 | 1.04468 |
| 750 | 1.02537 | 0.91777 | 1.00861 | 0.91028 | 0.90503 |
| 1,000 | 0.90462 | 0.82342 | 0.89259 | 0.81992 | 0.81554 |
| 2,000 | 0.66918 | 0.62598 | 0.64437 | 0.62811 | 0.62215 |
| 5,000 | 0.50709 | 0.48294 | 0.48954 | 0.48270 | 0.47743 |
| 8,000 | 0.44843 | 0.42956 | 0.42960 | 0.42951 | 0.42174 |
| 10,000 | 0.43315 | 0.41548 | 0.41408 | 0.41536 | 0.40710 |

The full nine-cell selected trajectory is in [`selected-grid-curves.csv`](results/selected-grid-curves.csv), with [`selected-nll-vs-updates.png`](results/selected-nll-vs-updates.png) and [`selected-delta-nll-vs-10.png`](results/selected-delta-nll-vs-10.png). The ordinary and single-axis executions are training-graph evaluations of the same checkpoint, so the comparison holds parameters and data fixed.

At 10,000 selection-panel updates, `(3,3)` is 0.00838 NLL below `(1,0)`. Repeated temporal updating without depth, `(3,0)`, is essentially tied with `(1,0)` (0.41536 vs 0.41548); the longer-run gain is concentrated in the hybrid that also refines depth state. Depth-only `(0,3)` is slightly below `(1,0)` at this milestone, but the difference is small. This supports a modest benefit from the jointly repeated path, not a claim that every extra update is useful.

The estimated forward matrix-multiply counts per sequence at 10,000 are: `(0,0)` 68,669,128,704; `(1,0)` 114,784,462,848; `(0,3)` 174,840,619,008; `(3,0)` 207,015,131,136; `(3,3)` 210,233,210,880. Thus `(3,3)/(1,0) = 1.83155` under the report's stated estimate.

## Confirmation panel

The confirmation panel is the frozen complement of the selection rows, not a game-disjoint generalization set. It was evaluated only after selection and continuation.

| updates | `(0,0)` | `(1,0)` | `(0,3)` | `(3,0)` | `(3,3)` | `(3,3) - (1,0)` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.49805 | 0.47437 | 0.48027 | 0.47429 | 0.46885 | -0.00553 |
| 10,000 | 0.42293 | 0.40609 | 0.40525 | 0.40615 | 0.39849 | -0.00760 |

The complete nine-cell JSON/CSV reports are [`grid-confirmation-step005000.json`](results/lr3e-4/grid-confirmation-step005000.json) and [`grid-confirmation-step010000.json`](results/lr3e-4/grid-confirmation-step010000.json). All nine cells were finite and all fixed rows were evaluated exactly once per placement.

## Diagnostics

The step-1,000 candidate stress checks used the prescribed eight-pass diagnostic schedules `(7,0)`, `(7,7)`, and `(1,7)`. They are outside the trained support and therefore are reported as extrapolation diagnostics, not acceptance tests. [`stress-selection-step001000.json`](results/stress-selection-step001000.json) records finite logits/losses for both candidates and the per-pass state RMS values.

The grid reports contain bounded first-batch observational diagnostics at each milestone. At selected step 10,000, the `(1,0)` trajectory has core RMS 1.314 → 1.412 and a 0.364 relative change on its second pass. Its temporal mixer reports mean gate values `alpha=0.140`, `beta=0.572`, with memory-value contribution RMS 0.381 versus prelude-value contribution RMS 0.964. For `(3,3)`, core RMS is 1.314 → 1.532 → 1.553 → 1.555 and successive relative changes are 0.396, 0.101, and 0.039; depth state and anchor value RMS are approximately 0.50 and 0.85. Sampled adjacent-token cosine similarities are recorded without constructing quadratic token-correlation matrices.

These diagnostic gradients were obtained with an eval-mode `autograd.grad` probe and leave training `.grad` buffers untouched. They are not accumulated optimizer gradients and are not evidence by themselves of useful recurrence. The NLL/accuracy comparisons above are the evidence used for the recurrence assessment.

## Generation observations

Generation used the fixed prompt list in [`prompts.json`](results/prompts.json), seed 2027, temperature 1, top-k 0, 128 new characters, mask seed 11, and 20 samples per checkpoint/cell. It used `sample.py --execution=training_graph --device=mps`; no live-feedback execution was attempted.

Mean legal continuation length increased with training. The four cells at each milestone were:

| checkpoint | `(0,0)` | `(1,0)` | `(3,0)` | `(3,3)` |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 7.25 | 6.60 | 6.25 | 7.65 |
| 5,000 | 10.00 | 11.75 | 11.70 | 11.15 |
| 10,000 | 14.80 | 16.85 | 14.60 | 16.55 |

At 10,000, completed-move legality was 0.940, 0.955, 0.936, and 0.948 respectively for those four cells. Most samples still terminated on an illegal or malformed move; none achieved the report's valid-termination criterion. These are developmental generation observations, not a quality gate. Raw JSON files are retained as `generation-stepNNNNNN-cellU-V.json` under [`experiments/sweeps/baseline_lr_selection/results/lr3e-4`](results/lr3e-4).

## Assessment and boundary

The baseline continuation answers the stated question provisionally: one temporal update is a strong baseline, and a jointly repeated temporal/depth path becomes modestly useful by 5,000–10,000 updates at a cost of about 1.83× estimated forward matrix multiplications. The effect survives the 1,317-row confirmation panel, but it is small relative to the absolute NLL and has not been tested across independent training seeds in this run. The result does not justify expanding the grid or changing the architecture yet.

The next decision should therefore use this fixed evidence. If further work is authorized, it should be a separately specified experiment that preserves the distinction between training-graph recurrence, live feedback, mask placement variation, training-seed variation, and diagnostic-gradient measurements. This report does not start that later stage.
