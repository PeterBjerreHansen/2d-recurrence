# Usage guide

Setup, data, training, resume, evaluation and generation. For the ideas behind the model, start with [concepts](concepts.md). The exact training graph is in the [recurrence contract](RECURRENCE_CONTRACT.md), and token-by-token execution in the [inference contract](INFERENCE_CONTRACT.md).

## Setup

Use Python 3.11 and uv. The lockfile records dependencies. Commands run from the repository root.

```sh
uv sync --frozen --python 3.11
uv run pytest -q
```

## Data

The reference data is Karvonen's Lichess PGN corpus, prepared into fixed 1,024-character rows:

```sh
uv run python data/chess_v1/prepare.py --file lichess_100mb_blocks.zip --out-dir data/chess_143K_v1
```

The full-corpus runs use `lichess_6gb_blocks.zip`, prepared as `data/chess_8M_v1`.

- The vocabulary uses `uint8` character IDs. Stored rows contain 1,024 characters; inputs and targets contain 1,023. Short test contexts preserve the storage stride.
- Internal game markers (`;`) keep their inherited causal context within a row.
- The split uses seed 2357 and 1% validation. Preparation rejects malformed rows, unknown characters and exact row overlap between splits. It does not assert game-disjoint generalization.
- Every dataset has a `manifest.json` with source revision and file hashes. Training validates the manifest, vocabulary and data hashes. Reuse a prepared dataset only if its manifest matches.

## Training

```sh
uv run python train.py configs/local/recurrent_mps.py
uv run python train.py configs/local/transformer_mps.py
```

These are bounded batch-8 local checks on Apple MPS, not the comparison profile. CPU smoke configs live under `experiments/smoke/configs/`. The shared batch-100 CUDA profile is `experiments/serious.py`; see the [experiment index](../experiments/README.md).

Recurrent training needs an explicit update distribution. The trainer has no default:

- `update_support`: the allowed counts per axis, for example `[0, 1, 3]`.
- `update_probabilities`: the joint matrix over `(U_T, U_D)` in that support.
- `update_probability_schedule` (optional): a piecewise-constant sequence of matrices keyed by absolute optimizer step, for curricula.
- `recurrence_mode`: `hybrid` (default), `temporal` or `depth`. Specialized modes reject probability mass on the missing axis.

For each sampled pair the model performs `max(U_T, U_D) + 1` passes, trains only the final prediction by default, and keeps gradients through all held states. `deep_supervision=True` adds intermediate-pass losses; the [deep-supervision comparison](../experiments/ablations/deep_supervision/README.md) selected final-only.

Learning-rate schedules: `lr_schedule='cosine'`, `'constant'` or `'wsd'` (warmup–stable–decay). If `lr_schedule` is omitted, the historical `decay_lr` flag decides between cosine and constant.

## Configurable layout

Five block counts follow physical block order: prelude, buffer, core, source, coda. They sum to `n_layer`; the core must be nonempty. Other segments may be empty or contain several blocks.

| Layout | `n_prelude` | `n_buffer` | `n_core` | `n_source` | `n_coda` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default A | 1 | 1 | 4 | 1 | 1 |
| B, coincident | 1 | 0 | 6 | 0 | 1 |
| Original experiments | 2 | 0 | 4 | 1 | 1 |

Changing counts moves the injection and source boundaries within one ordered stack. An empty buffer means adjacent injection sites, and an empty source means both states come from the core output. These are not new module types or duplicated weights. Historical configs pin their layout explicitly. A was chosen after a near-tied [A/B comparison](../experiments/ablations/architecture_sites/REPORT.md). B was slightly better at late predictive NLL, but below the predeclared selection margin.

## Checkpoints and resume

Checkpoints include model, optimizer, scaler, completed update count, random generators, recurrence sampler, configuration, dataset and panel identity, and environment provenance.

- Resume with the same config plus `--init_from=resume`. `max_iters` is an absolute stopping step.
- Output, logging, evaluation and checkpoint-interval settings may change on resume, as may the panel file's location if its content is unchanged. Other training-setting changes are rejected.
- Raising the stopping step does not extend the LR decay schedule.
- New checkpoints store the complete layout. Old checkpoints load with their original missing-field defaults, not today's A defaults.
- Exact resume is tested on CPU. A 100-update CUDA test matched bitwise with deterministic kernels (`RECURRENCE_TORCH_DETERMINISTIC=1`); a longer deterministic test is pending. With ordinary CUDA kernels, resumed and uninterrupted runs match statistically, not bitwise, because two uninterrupted runs also diverge.
- Under DDP, rank zero samples and broadcasts each microbatch schedule, and global accumulation must divide evenly across workers.
- Load only trusted checkpoints and vocabulary files.

## Evaluation

### Training-graph grid

Evaluate a checkpoint on the `(U_T, U_D)` surface using the parallel training graph:

```sh
uv run python -m evaluation.recurrence_grid \
  --checkpoint experiments/ablations/architecture_sites/results/separated/ckpt-step010000.pt \
  --panel-file experiments/ablations/architecture_sites/panel.json \
  --device=mps --output experiments/ablations/architecture_sites/results/recheck-mps.json
```

- By default the grid covers the nine cells in `{0,1,3}²`.
- A frozen panel evaluates every listed validation row exactly once. Without one, the evaluator samples fixed batches with replacement. All cells use identical batches.
- Asymmetric cells are evaluated with three distinct write-mask placements (`--mask-seeds`). Cells with only one possible placement are evaluated once. Placement variation is not variation across training seeds.
- Evaluation uses its own fixed schedules and never advances the training RNG.
- Reported FLOPs estimate forward matrix multiplications, not total training compute. State and gradient diagnostics probe one fixed batch without altering model gradients.

### Live execution

Live execution runs token by token with real temporal feedback and incremental KV caches; see the [inference contract](INFERENCE_CONTRACT.md). `evaluation.live_inference.evaluate_teacher_forced` computes teacher-forced live NLL on validation rows. The 20B study records it at major checkpoints. Training-graph and live results measure different executions and are reported separately.

## Generation

```sh
uv run python sample.py --checkpoint path/to/ckpt.pt --execution live --depth-steps 4
uv run python sample.py --checkpoint path/to/ckpt.pt --execution training_graph --u-t 3 --u-d 3
```

- `--execution live` needs `--depth-steps` for depth and hybrid checkpoints; temporal-only checkpoints run exactly one core pass per token. `--kv-strategy` selects `final_depth` or `depth_specialized` caches.
- `--execution training_graph` recomputes the full prefix with a fixed write schedule for every new token.
- Training update counts are not live loop counts.
- Generation samples without a legal-move mask and stops at the first completed illegal or malformed move, with no retry or repair. Reports keep the offending text, board, legal continuation length and stop reason.
- An unfinished move at a length or context limit counts as truncation. Game separators are distinct from valid terminal positions or declared results.
- Output defaults to the checkpoint's directory and never overwrites an existing report.

## Repository layout

```text
model.py, train.py, sample.py      # baseline GPT, trainer, generation CLI
models/recurrent_2d.py             # configurable layout, mixers, training trajectory
recurrence/schedule.py             # update counts, write masks, sampler state
inference/                         # live execution, KV caches, slow reference oracle
evaluation/                        # reusable evaluators (grid, live NLL, stress checks, chess)
configs/local/                     # small local MPS configs
data/chess_v1/prepare.py           # dataset preparation
experiments/
  serious.py, run_serious.py       # shared batch-100 CUDA profile and its runner
  ablations/                       # architecture sites, deep supervision, gate init, update schedule
  sweeps/baseline_lr_selection/    # LR selection and 10k continuation
  long_runs/                       # 1B_baseline, 5B_axis, 20B_recurrence, 64B_core
  benchmarks/                      # runtime, throughput and resume probes
  smoke/                           # small pipeline checks and historical validations
  relocations.json                 # old paths in immutable provenance -> current locations
docs/                              # concepts, contracts, usage, plans, figures
tests/
```

Each experiment owns its `results/` directory; there is no global results directory. Code, configs, protocols and concise reports are tracked. Checkpoints, logs, plots and raw reports stay local and ignored. Paths embedded in historical checkpoints and receipts are left unchanged; [relocations.json](../experiments/relocations.json) records where they moved.

Reusable metrics live in `evaluation/`. Experiment-specific analysis lives with its experiment. For example, this rebuilds the completed A/B summary on CPU from the preserved protocol and checkpoints:

```sh
uv run python -m experiments.ablations.architecture_sites.run summarize
```
