# Two-axis recurrent ChessGPT

A character-level chess language model combining temporal feedback across positions with recurrent depth. The default is **variation A**: one prelude block, one block between temporal and depth injection, four recurrent-core blocks, one temporal-source block, and one coda block. The eight-block backbone has width 512, eight heads, learned positions, and tied embedding/unembedding weights.

```text
embeddings -> L1 -> temporal mix -> L2 -> depth mix -> L3-L6 -> L7 -> L8 -> head
                      ^                    ^           |      |
                      |                    +-- depth --+      |
                      +-------- shifted temporal memory -----+
```

Temporal memory comes from L7 and depth state from L6. Every available state is read; randomized masks control writes only. The prelude runs once per training trajectory. The buffer and core run on each pass; L7 runs for temporal writes and final prediction, and L8 runs only for final prediction. The [contract](docs/RECURRENCE_CONTRACT.md) contains the detailed sketch, equations, initialization, and masking semantics. See also the [proposal](proposal.md) and [implementation plan](implementation_plan.md).

A is the practical default after a near-tied [A/B comparison](experiments/ablations/architecture_sites/REPORT.md). B was slightly better at late predictive NLL, but below the predeclared selection margin. The implementation retains both layouts and configurable boundaries; the result does not establish a universal advantage for separation.

## Setup and training

Use Python 3.11 and uv. The lockfile records dependencies. Commands run from the repository root.

```sh
uv sync --frozen --python 3.11
uv run pytest -q
uv run python data/chess_v1/prepare.py --file lichess_100mb_blocks.zip --max-rows 4096 --out-dir data/smoke_real
uv run python train.py configs/recurrent.py
```

Reuse an existing prepared dataset only if its manifest matches; preparation refuses to overwrite one. `configs/recurrent.py` is a bounded MPS pilot of default A, with results under `experiments/long_runs/separated/results/`. For CUDA, explicitly override the device; CPU and MPS require float32. Keep compilation off for variable recurrence schedules. This pilot is not a new long-training recommendation.

For a small CPU pipeline check, use `experiments/smoke/configs/baseline.py` or `experiments/smoke/configs/recurrent.py`. These preserve historical small test architectures. `configs/baseline_chessgpt.py` retains the ordinary eight-layer reference and its long CUDA schedule; do not mistake it for a quick test.

## Configurable architecture

The five counts follow physical block order: prelude, buffer, core, source, coda. They sum to `n_layer`; the core must be nonempty. Other segments may be empty or contain several blocks.

| Layout | `n_prelude` | `n_buffer` | `n_core` | `n_source` | `n_coda` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default A | 1 | 1 | 4 | 1 | 1 |
| B, coincident | 1 | 0 | 6 | 0 | 1 |
| Original experiments | 2 | 0 | 4 | 1 | 1 |

Changing counts moves source/destination boundaries within the ordered architecture. Empty buffer means adjacent injection sites; empty source means both candidates are the core output. These are not new module types or independent duplicated weights. Historical configs explicitly pin their layout.

## Experiments and results

The [experiment index](experiments/README.md) is the entry point for protocols, configs, scripts, reports, and retained artifacts.

```text
experiments/
  ablations/architecture_sites/   # A versus B; configs/, run.py, REPORT.md, results/
  sweeps/recurrence_grid/         # initial nine-cell evaluation, two seeds, results/
  long_runs/
    recurrence_pilot/            # 1k-update pilot continuation, results/
    baseline/                    # LR selection + selected 10k run, results/
    separated/                   # current default-A pilot, results/ when run
  smoke/                         # pipeline checks, validation notes, results/
```

Each experiment owns its `results/`; there is no global results directory. Code, configs, protocols, and concise reports are tracked; checkpoints, logs, plots, and raw reports stay local and ignored. Shared datasets remain in `data/`. Original paths embedded in historical checkpoints and receipts are intentionally unchanged; [relocations.json](experiments/relocations.json) records where they moved. The architecture ablation retains its original transfer archive, which can reproduce the pre-cleanup source snapshot.

Reusable metrics live in `evaluation/`. Experiment-specific analysis lives with its experiment. For example, rebuild the completed A/B summary on CPU without provisioning a GPU:

```sh
uv run python -m experiments.ablations.architecture_sites.run summarize
```

This reads the preserved protocol/checkpoints and writes derived summaries under that experiment's `results/analysis/`, preserving the original decision. The runner requires the frozen CUDA environment for new training/evaluation; use a fresh `--results-dir` inside the experiment for an explicitly specified rerun. It does not automatically launch additional runs.

## Evaluation and execution modes

Training samples exact update counts from a configured distribution and randomly places the writes. The default pilot uses counts in `{0,1,3}`. For each pair, the model performs `max(U_T,U_D)+1` passes, trains only the final prediction, and retains gradients through all held states. Evaluation uses independent fixed schedules without advancing training RNG.

```sh
uv run python -m evaluation.recurrence_grid \
  --checkpoint experiments/ablations/architecture_sites/results/separated/ckpt-step010000.pt \
  --panel-file experiments/ablations/architecture_sites/panel.json \
  --device=mps --output experiments/ablations/architecture_sites/results/recheck-mps.json
```

A provided panel evaluates every specified row exactly once. Without a panel the evaluator samples fixed batches with replacement. All nine pilot cells use identical batches; asymmetric cells have three distinct mask placements, and deterministic cells are evaluated once. Placement variation is not variation across training seeds. Reported FLOPs estimate forward matrix multiplications, not total training compute. State/gradient diagnostics probe one fixed batch without altering model gradients.

All current recurrent evaluation and generation uses the **parallel training graph**. Generation must specify `--execution=training_graph` and recomputes the prefix using a fixed write schedule. Live temporal-feedback generation remains planned; in A it will run L2 once before the depth loop and L7 once after it. Do not interpret training update counts as live inference loop counts.

Generation samples without a legal-move mask and stops at the first completed illegal or malformed move, with no retry or repair. Reports retain the offending text, board, legal continuation length, and stop reason. Unfinished moves at a length/context limit count as truncation. Game separators are distinct from valid terminal board positions or declared results. Generation output defaults beside its checkpoint and refuses to overwrite an existing report.

## Data and reproducibility

The reference vocabulary uses `uint8` character IDs. Stored rows contain 1,024 characters; normal inputs and targets contain 1,023. Short test contexts preserve the storage stride. Internal game markers retain inherited causal context. The split uses seed 2357 and 1% validation; preparation rejects malformed rows, unknown characters, and exact overlap between splits. It does not assert game-disjoint generalization. Training validates the manifest, vocabulary, and data hashes.

Checkpoints include model, optimizer, scaler, completed update count, random generators, recurrence sampler, configuration, dataset/panel identity, and environment provenance. Resume with the same config plus `--init_from=resume`; `max_iters` is an absolute stopping step. Moving a panel without changing its content is allowed. Other training-setting changes are rejected. Raising the stopping step does not extend the LR decay schedule.

New checkpoints store the complete layout. Old checkpoints are loaded with their original missing-field defaults, not today's A defaults. Resume equivalence is tested on CPU; CUDA can introduce small numerical differences. Under DDP, rank zero broadcasts each microbatch schedule and global accumulation must divide evenly across workers. Load only trusted checkpoints and vocabulary files.

The project retains Karvonen/nanoGPT ancestry and the MIT license. [Upstream provenance](docs/upstream.json) records the reference revision; the ordinary transformer computation remains in `model.py`.
