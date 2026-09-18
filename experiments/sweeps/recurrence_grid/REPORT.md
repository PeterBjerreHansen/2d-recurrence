# Stage 9 initial MPS results

Historical record of the completed experiment. Paths below follow the current repository layout; recorded configurations, measurements, and raw artifact provenance retain their original meaning. See the experiment README for current commands.

Status: complete within the 45-minute initial-experiment budget. Both requested 100-update training runs, all 10 nine-cell grid reports (90 cells and 130 placement evaluations), the retained checkpoints, and the requested chess-continuation checks completed with finite values. No later-stage work was launched.

## Decision

The initial result supports continuing the current temporal-recurrence pilot, but does not justify grid expansion or live-feedback work yet. At the final checkpoint, temporal-only execution `(U_T,U_D)=(3,0)` improved fixed-batch NLL relative to ordinary execution `(0,0)` by 0.3025 and 0.2903 for training seeds 1337 and 1338. Depth-only `(0,3)` was neutral to slightly worse (+0.0022 and +0.0071). Adding depth to temporal recurrence `(3,3)` gave only a small further improvement over `(3,0)` (about 0.0053 and 0.0022 NLL).

This is a preliminary fixed-data, two-seed result. The effect is not monotonic in update count: one temporal update was slightly better than three on the final fixed batches, and untrained step 0 generally became worse when extra passes were enabled. The useful signal therefore appears to be learned behavior, not a generic benefit from executing more passes. A longer curve is the appropriate next experiment if this pilot is continued; no hyperparameters or architecture were changed after seeing validation results.

Follow-up exact-row evaluation, feedback-content intervention, and the unchanged 1,000-update pilot are documented in [Stage 9 continued results](../../long_runs/recurrence_pilot/REPORT.md).

## Configuration and provenance

The runs used the committed `experiments/sweeps/recurrence_grid/config.py` configuration and the exact handoff commands:

```sh
uv run python train.py experiments/sweeps/recurrence_grid/config.py --out_dir=experiments/sweeps/recurrence_grid/results/seed1337 --seed=1337 --recurrence_seed=1729
uv run python train.py experiments/sweeps/recurrence_grid/config.py --out_dir=experiments/sweeps/recurrence_grid/results/seed1338 --seed=1338 --recurrence_seed=1730
```

| Item | Value |
| --- | --- |
| Git branch / commit | `mvp-2d-recurrence` / `b43e2d22baaa68798f74c39b3a67a56e1aced9b4` |
| Device / dtype / execution | Apple MPS / float32 / eager (`compile=False`) |
| Model | 8 blocks, width 512, 8 heads, context 1,023; 2 prelude + 4 shared core + 1 temporal source + 1 coda |
| Training | 100 optimizer updates; batch 2; accumulation 4; effective batch 8; AdamW; learning rate 0.0003 to 0.00003; warmup 10; `lr_decay_iters=100`; dropout 0 |
| Recurrence support | `U_T,U_D ∈ {0,1,3}` with probabilities `[[.10,.12,.04],[.12,.26,.08],[.04,.08,.16]]` |
| Dataset | `data/smoke_real`, prepared 4,096-row subset; 4,055 train rows / 41 validation rows; row size 1,024 |
| Dataset manifest | `36b7025c4331ec6f6fafc03fa0248ef90fc55f4ab2ecd69c763707a6ef56952d` |
| Fixed grid evaluation | validation split; 8 batches; batch size 2; data seed 2027; mask seeds 11, 23, 37 |
| Grid identity | every report has 9 cells, 13 placement evaluations, and batch fingerprint `6f2a452a926a87e7fb7a9c9cbe739a5484e4399415bb9e5fb11e6b768ff18075` |

The dataset is the committed `smoke_real` preparation from `adamkarvonen/chess_games`, source revision `1a932e1abca935aae585f417ede39ecde4f2a620`, with no exact encoded row shared between splits. Rows remain independent while causal context across internal `;` markers is preserved.

The training configurations, package versions, manifest hash, and checkpoint provenance are retained in [seed 1337 `run.json`](results/seed1337/run.json) and [seed 1338 `run.json`](results/seed1338/run.json). The initial preflight passed all 50 tests and verified `torch.backends.mps.is_available() == True`.

## Retained checkpoints and training curves

Each seed retained checkpoints at optimizer steps 0, 25, 50, 75, and 100, plus the latest `ckpt.pt`. The intermediate snapshots are [seed 1337 step 0](results/seed1337/ckpt-step000000.pt), [25](results/seed1337/ckpt-step000025.pt), [50](results/seed1337/ckpt-step000050.pt), [75](results/seed1337/ckpt-step000075.pt), [100](results/seed1337/ckpt-step000100.pt), and the corresponding [seed 1338 step 0](results/seed1338/ckpt-step000000.pt), [25](results/seed1338/ckpt-step000025.pt), [50](results/seed1338/ckpt-step000050.pt), [75](results/seed1338/ckpt-step000075.pt), [100](results/seed1338/ckpt-step000100.pt).

The plots use the same fixed validation batches at every checkpoint. The first plot shows raw NLL; the second subtracts `(0,0)` at each checkpoint so ordinary language-model learning is not mistaken for a recurrence gain.

![Validation NLL versus optimizer updates](results/curves/nll-vs-updates.png)

![NLL difference versus ordinary execution](results/curves/delta-nll-vs-00.png)

The reproducible tables are [per-seed grid curves](results/curves/grid-curves-by-seed.csv), [seed-summary curves](results/curves/grid-curves-seed-summary.csv), and [placement-level results](results/curves/placement-results.csv). The small [plotting/aggregation helper](curves.py) reads the raw JSON reports and writes these tables and plots.

### Seed 1337: nine-cell NLL curve

| updates | (0,0) | (0,1) | (0,3) | (1,0) | (1,1) | (1,3) | (3,0) | (3,1) | (3,3) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.57703 | 3.67194 | 3.72172 | 3.73506 | 3.66602 | 3.71209 | 3.73072 | 3.70561 | 3.70826 |
| 25 | 2.13511 | 2.12837 | 2.13640 | 1.82960 | 1.82534 | 1.83030 | 1.85966 | 1.85507 | 1.86414 |
| 50 | 2.00585 | 1.95582 | 1.96686 | 1.63217 | 1.64096 | 1.63798 | 1.63497 | 1.63470 | 1.63597 |
| 75 | 1.87691 | 1.86159 | 1.86256 | 1.52246 | 1.53393 | 1.52985 | 1.53050 | 1.53047 | 1.52652 |
| 100 | 1.79087 | 1.79026 | 1.79307 | 1.48218 | 1.48597 | 1.48415 | 1.48836 | 1.48575 | 1.48301 |

### Seed 1338: nine-cell NLL curve

| updates | (0,0) | (0,1) | (0,3) | (1,0) | (1,1) | (1,3) | (3,0) | (3,1) | (3,3) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.64683 | 3.73311 | 3.76622 | 3.76593 | 3.72876 | 3.75960 | 3.75842 | 3.74998 | 3.75672 |
| 25 | 2.21458 | 2.16578 | 2.17273 | 1.80973 | 1.86330 | 1.83874 | 1.81597 | 1.83455 | 1.81970 |
| 50 | 1.99224 | 1.97975 | 1.98269 | 1.65031 | 1.66191 | 1.65862 | 1.65430 | 1.65528 | 1.65518 |
| 75 | 1.85726 | 1.84568 | 1.84479 | 1.54549 | 1.55395 | 1.54958 | 1.55415 | 1.55296 | 1.55211 |
| 100 | 1.79647 | 1.79888 | 1.80355 | 1.50281 | 1.50532 | 1.50506 | 1.50622 | 1.50448 | 1.50399 |

### Final checkpoint comparisons

`(0,0)` is ordinary one-pass execution using the hybrid checkpoint's weights; it is not a separately trained ordinary baseline. `(3,0)` is temporal-only, `(0,3)` is depth-only, and `(3,3)` uses both axes.

#### Seed 1337, step 100

| `(U_T,U_D)` | NLL | Δ vs `(0,0)` | accuracy | placement SD |
| ---: | ---: | ---: | ---: | ---: |
| (0,0) | 1.790866 | +0.000000 | 0.371395 | 0.000000 |
| (0,1) | 1.790264 | -0.000602 | 0.372923 | 0.000000 |
| (0,3) | 1.793074 | +0.002209 | 0.374328 | 0.000000 |
| (1,0) | 1.482178 | -0.308687 | 0.479044 | 0.000000 |
| (1,1) | 1.485966 | -0.304899 | 0.482221 | 0.000000 |
| (1,3) | 1.484146 | -0.306719 | 0.481366 | 0.003956 |
| (3,0) | 1.488364 | -0.302502 | 0.476723 | 0.000000 |
| (3,1) | 1.485747 | -0.305118 | 0.480348 | 0.004232 |
| (3,3) | 1.483015 | -0.307851 | 0.481855 | 0.000000 |

#### Seed 1338, step 100

| `(U_T,U_D)` | NLL | Δ vs `(0,0)` | accuracy | placement SD |
| ---: | ---: | ---: | ---: | ---: |
| (0,0) | 1.796474 | +0.000000 | 0.371823 | 0.000000 |
| (0,1) | 1.798883 | +0.002409 | 0.372617 | 0.000000 |
| (0,3) | 1.803552 | +0.007078 | 0.374756 | 0.000000 |
| (1,0) | 1.502806 | -0.293668 | 0.474585 | 0.000000 |
| (1,1) | 1.505315 | -0.291158 | 0.476234 | 0.000000 |
| (1,3) | 1.505059 | -0.291415 | 0.476499 | 0.002179 |
| (3,0) | 1.506219 | -0.290255 | 0.474523 | 0.000000 |
| (3,1) | 1.504479 | -0.291995 | 0.476967 | 0.003778 |
| (3,3) | 1.503988 | -0.292486 | 0.476417 | 0.000000 |

Placement SD is variation across the three distinct masks only for asymmetric cells `(1,3)` and `(3,1)`; it is not a training-seed confidence interval. The raw JSON retains every mask and placement result. In particular, the seed-to-seed variation in final `(3,0)` ΔNLL is about 0.0061 population SD, and for `(3,3)` it is about 0.0077.

## Diagnostics

The evaluator ran one fixed-batch eval-mode diagnostic per cell and retained the traces in every grid JSON and in [diagnostics.csv](results/curves/diagnostics.csv). Activations and gradients were finite. Representative final-step values are below; RMS entries are listed by core/source invocation in pass order.

| seed | cell | core output RMS | source output RMS | temporal mixer grad L2 | depth mixer grad L2 |
| ---: | ---: | --- | --- | ---: | ---: |
| 1337 | (0,0) | 1.0038 | 1.0938 | null (unused) | null (unused) |
| 1337 | (3,0) | 1.0038 / 1.4987 / 1.4273 / 1.4380 | 1.0938 / 1.5971 / 1.5219 / 1.5327 | 0.1638 | null (unused) |
| 1337 | (0,3) | 1.0038 / 1.6448 / 1.6410 / 1.6401 | 1.7250 | null (unused) | 0.2799 |
| 1337 | (3,3) | 1.0038 / 1.6107 / 1.6399 / 1.6346 | 1.0938 / 1.7028 / 1.7324 / 1.7263 | 0.1323 | 0.2632 |
| 1338 | (0,0) | 1.0486 | 1.1520 | null (unused) | null (unused) |
| 1338 | (3,0) | 1.0486 / 1.5544 / 1.4810 / 1.4923 | 1.1520 / 1.6659 / 1.5896 / 1.6013 | 0.1654 | null (unused) |
| 1338 | (0,3) | 1.0486 / 1.6912 / 1.6817 / 1.6789 | 1.7736 | null (unused) | 0.3759 |
| 1338 | (3,3) | 1.0486 / 1.6474 / 1.6740 / 1.6693 | 1.1520 / 1.7520 / 1.7799 / 1.7748 | 0.1300 | 0.3013 |

These are diagnostic gradients from one eval batch, not accumulated training gradients. A null unused-mixer gradient is expected. Nonzero mixer gradients and changing state RMS establish participation in the computation, not useful memory content or causal improvement. The report's FLOP values estimate forward matrix multiplications and exclude normalization, softmax, nonlinearities, elementwise work, backward, and kernel overhead; they are not a complete compute frontier.

## Chess-continuation checks

All recurrent samples used `execution=training_graph`, `prefill=full_prefix_recomputation_each_character`, `mask-seed=11`, and a four-prompt JSON list: `";1."`, `";1.e4 e5 2."`, `";1.d4 d5 2."`, and `";1.e4 c5 2."`. Each setting used 20 samples, sampling seed 2027, temperature 1, top-k 0, and 128 new tokens. These are training-graph prefix recomputations, not live-feedback generation. No legal-move filtering or retries were used.

Completed-move legality counts the failed completed attempt in the denominator. `mean legal continuation` is the average number of legal moves completed before stopping. Zero attempts would make legality undefined; none of these aggregate rows should be read as zero-attempt perfect legality.

| run / execution | cell | completed-move legality | mean legal continuation | mean attempted moves | stop reasons (20 samples) |
| --- | ---: | ---: | ---: | ---: | --- |
| ordinary checkpoint / baseline | — | 0.474 | 0.90 | 1.90 | 16 illegal, 4 malformed |
| seed 1337 / training graph | (0,0) | 0.487 | 0.95 | 1.95 | 14 illegal, 6 malformed |
| seed 1337 / training graph | (3,0) | 0.524 | 1.10 | 2.10 | 16 illegal, 4 malformed |
| seed 1337 / training graph | (0,3) | 0.459 | 0.85 | 1.85 | 14 illegal, 6 malformed |
| seed 1337 / training graph | (3,3) | 0.512 | 1.05 | 2.05 | 17 illegal, 3 malformed |
| seed 1338 / training graph | (0,0) | 0.231 | 0.30 | 1.30 | 15 illegal, 5 malformed |
| seed 1338 / training graph | (3,0) | 0.355 | 0.55 | 1.55 | 15 illegal, 5 malformed |
| seed 1338 / training graph | (0,3) | 0.333 | 0.50 | 1.50 | 14 illegal, 6 malformed |
| seed 1338 / training graph | (3,3) | 0.394 | 0.65 | 1.65 | 18 illegal, 2 malformed |

All rows had PGN parse success rate 0 and valid-termination rate 0 under the 128-token limit. The small sample and seed spread do not establish a generation win. The ordinary checkpoint is the existing [100-update baseline checkpoint](../../smoke/results/baseline-pilot/ckpt.pt), evaluated only as a contextual reference; its training validation metric was NLL 1.7267 / accuracy 0.400 at step 100, but it was not evaluated by the recurrent grid. Its run metadata is [here](../../smoke/results/baseline-pilot/run.json).

Raw generation reports are [ordinary baseline](../../smoke/results/baseline-pilot/generation-stage09.json), [seed 1337 `(0,0)`](results/seed1337/generation-0-0.json), [seed 1337 `(3,0)`](results/seed1337/generation-3-0.json), [seed 1337 `(0,3)`](results/seed1337/generation-0-3.json), [seed 1337 `(3,3)`](results/seed1337/generation-3-3.json), [seed 1338 `(0,0)`](results/seed1338/generation-0-0.json), [seed 1338 `(3,0)`](results/seed1338/generation-3-0.json), [seed 1338 `(0,3)`](results/seed1338/generation-0-3.json), and [seed 1338 `(3,3)`](results/seed1338/generation-3-3.json). The aggregate [generation summary CSV](results/curves/generation-summary.csv) retains attempted-move and termination counts.

## Raw grid reports

Every JSON report retains masks, placement-level results, diagnostics, checkpoint hash/step, fixed-batch fingerprint, and evaluation provenance. Each JSON has an adjacent flat CSV.

| seed | step 0 | step 25 | step 50 | step 75 | step 100 |
| ---: | --- | --- | --- | --- | --- |
| 1337 | [JSON](results/seed1337/grid-step000000.json) / [CSV](results/seed1337/grid-step000000.csv) | [JSON](results/seed1337/grid-step000025.json) / [CSV](results/seed1337/grid-step000025.csv) | [JSON](results/seed1337/grid-step000050.json) / [CSV](results/seed1337/grid-step000050.csv) | [JSON](results/seed1337/grid-step000075.json) / [CSV](results/seed1337/grid-step000075.csv) | [JSON](results/seed1337/grid-step000100.json) / [CSV](results/seed1337/grid-step000100.csv) |
| 1338 | [JSON](results/seed1338/grid-step000000.json) / [CSV](results/seed1338/grid-step000000.csv) | [JSON](results/seed1338/grid-step000025.json) / [CSV](results/seed1338/grid-step000025.csv) | [JSON](results/seed1338/grid-step000050.json) / [CSV](results/seed1338/grid-step000050.csv) | [JSON](results/seed1338/grid-step000075.json) / [CSV](results/seed1338/grid-step000075.csv) | [JSON](results/seed1338/grid-step000100.json) / [CSV](results/seed1338/grid-step000100.csv) |

The prompts are saved as proper JSON in [experiments/sweeps/recurrence_grid/results/prompts.json](results/prompts.json). No checks were started for grid expansion, extensive component baselines, hyperparameter search, live-feedback inference, or later stages.
