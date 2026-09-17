# Two-axis recurrent ChessGPT

Stages 0–9 implement the ordinary character-level chess baseline and two-axis recurrent training and evaluation. Both use the eight-layer, eight-head, width-512 Karvonen/nanoGPT backbone with learned positions and tied embedding/unembedding weights. The recurrent model partitions those blocks into 2 prelude, 4 shared core, 1 temporal source, and 1 coda. See [the proposal](proposal.md), [implementation plan](implementation_plan.md), and [recurrence contract](docs/RECURRENCE_CONTRACT.md).

This is a local fork retaining the upstream Git ancestry and MIT license. [Upstream provenance](docs/upstream.json) records the reference revision. `model.py` retains its transformer computation; GPT-2 checkpoint import was removed. The training loop keeps AdamW, cosine decay, gradient accumulation, mixed precision, optional compilation, and DDP, with complete resume state and explicit data validation added.

## Setup and quick verification

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). `uv.lock` records the tested dependency versions.

```sh
uv sync --python 3.11
uv run pytest -q
uv run python data/chess_v1/prepare.py --file lichess_100mb_blocks.zip --max-rows 4096 --out-dir data/smoke_real
uv run python train.py configs/smoke.py
uv run python sample.py --checkpoint out-smoke/ckpt.pt --num-samples 10 --output out-smoke/generation.json
```

The smoke run deliberately uses a smaller model and context to test the pipeline. Its generation quality is not an architecture result. The preparation command downloads the approximately 55 MB reference archive, then takes the first 4,096 rows before applying the upstream shuffled split. Prepared data directories are immutable: use a different output directory to change a version rather than overwriting it.

## Eight-layer baseline

For a short initial run of the full model on Apple Silicon:

```sh
uv run python train.py configs/baseline_pilot.py
uv run python sample.py --checkpoint out-baseline-pilot/ckpt.pt --device=mps --num-samples 20 --output out-baseline-pilot/generation.json
```

This configuration uses the verified subset, full 1,023-character context, 100 optimizer updates, microbatch size 2, four accumulation steps, float32, and MPS. It is an initial pipeline and learning check, not a full-corpus reproduction. Results and limits of the validation performed during implementation are recorded in [stage validation](docs/STAGE_01_VALIDATION.md).

For the full reference dataset and CUDA configuration:

```sh
uv run python data/chess_v1/prepare.py
uv run python train.py configs/baseline_chessgpt.py --batch_size=20 --gradient_accumulation_steps=5
```

The default archive is approximately 3.17 GB compressed; preparation also needs space for extracted/cached rows and token files. The full configuration retains the upstream 600,000-update schedule. Adjust hardware settings explicitly rather than treating this as a quick test. The example preserves an effective batch of 100 rows by accumulating five microbatches of 20. `gradient_accumulation_steps` is a global count divided across DDP ranks, as in upstream. It must be divisible by world size. CPU and MPS use `--dtype=float32`; use `--compile=False` on MPS. The CUDA path has not been exercised on this Mac.

To test two CPU DDP workers locally without depending on hostname resolution:

```sh
uv run torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29671 train.py configs/smoke.py --backend=gloo --gradient_accumulation_steps=2 --out_dir=out-ddp --max_iters=2 --eval_iters=1
```

## Data contract

Preparation retains the reference vocabulary and `uint8` representation. Every stored row has 1,024 characters and starts with `;`. Training takes 1,023 inputs and their shifted targets from that row. Shorter smoke-test contexts still sample at the 1,024-character storage stride. Rows can contain multiple games separated by `;`; causal attention across those internal boundaries is preserved as in the reference. The manifest records how often this occurs.

The shuffled split uses seed 2357 and 1% validation. Preparation rejects unknown characters, malformed row sizes, and exact rows shared between splits. It reports duplicates within a split but does not claim that all games or related openings are disjoint. `manifest.json` records the dataset revision, source and output hashes, vocabulary, preparation code identity, split details, and boundary behavior. Training verifies the files against that manifest before use. For offline fixtures, preparation also accepts `--input path/to/rows.jsonl` or CSV with a `transcript` column.

## Evaluation policy

Generation samples characters without a legal-move mask. Once a move boundary is reached, `python-chess` checks the complete move. The first illegal or malformed move ends the sample; there is no retry or repair. Reports preserve the text, failing move, board before failure, legal continuation length, and stopping reason. Legal-move rate includes the failed completed attempt in its denominator. A length/context limit with an unfinished move is recorded as truncation, not as an illegal move.

The upstream `;` game separator ends a sample with `game_boundary`; this is recorded separately from a terminal chess position or declared result. PGN parse success alone is not evidence of a complete legal game. `valid_termination` requires a terminal board or declared result, and the report retains the exact reason. No draw claim or resignation is inferred from a separator.

Training logs fixed-batch train/validation NLL and character accuracy to `metrics.jsonl`. Generation is a separate command with explicit prompt, temperature, top-k, seed, and limits. `--temperature=0` selects greedy decoding; the default is unfiltered temperature-1 sampling. `--prompts` accepts a JSON list. Prompts must start at a game boundary and end at whitespace or a bare move number, such as `;1.` or `;1.e4 e5 2.`. The evaluator stops at the model context limit rather than discarding old board context.

## Checkpoints and resume

```sh
uv run python train.py configs/baseline_pilot.py --init_from=resume
uv run python train.py configs/baseline_pilot.py --init_from=resume --eval_only=True
```

`ckpt.pt` stores model, optimizer, scaler, completed update count, per-rank random-generator states, batch-generator state, vocabulary, dataset-manifest hash, configuration, and code/environment provenance. Saves are atomic. `max_iters` is the total desired completed updates, not additional updates. Resuming with the same limit performs no extra training. A longer continuation can set a larger `max_iters` while retaining the original learning-rate schedule; changing that schedule constitutes a new experiment and is rejected by exact resume.

Resume requires the same model, dataset, training settings, device, and world size. Logging/evaluation intervals and the total stopping step may change. CPU exact resume is tested against uninterrupted training with dropout. GPU backends can have additional numerical nondeterminism. Load only trusted local checkpoints and vocabulary files, which use Python serialization. `run.json`, append-only `events.jsonl`, and checkpoint provenance record the configuration and source state for each invocation.

## Recurrent training

After preparing the smoke dataset above, run a small CPU check or the full-width pilot:

```sh
uv run python train.py configs/recurrent_smoke.py
uv run python train.py configs/recurrent_2d_pilot.py
```

The pilot samples exact temporal/depth write counts from $\{0,1,3\}^2$ and randomly places the writes. All available states are read on every pass, including when held. Training uses final-pass cross entropy with full gradients through every pass. Evaluation uses the explicit fixed pair `eval_u_t=3`, `eval_u_d=3` and an independent fixed mask RNG. It does not consume training schedules.

Recurrent checkpoints include the sampler state and accumulated pair/pass histograms. The ordinary resume command applies unchanged. Under DDP, rank zero broadcasts each microbatch schedule; global accumulation must be divisible by world size. Use `compile=False` for recurrent training.

These runs execute the parallel training graph. Recurrent generation requires explicit `--execution=training_graph`; it recomputes the prefix for each character with a fixed write schedule. Live temporal-feedback generation remains Stage 14. [Stages 2–8 validation](docs/STAGE_02_08_VALIDATION.md) records the semantic, resume, distributed, and device checks.

## Stage 9: nine-cell evaluation

```sh
uv run python train.py configs/stage09_mps.py
uv run python -m evaluation.recurrence_grid --checkpoint out-stage09-seed1337/ckpt-step000100.pt --device=mps --output out-stage09-seed1337/grid-step000100.json
uv run python sample.py --checkpoint out-stage09-seed1337/ckpt-step000100.pt --device=mps --execution=training_graph --u-t=3 --u-d=3 --output out-stage09-seed1337/generation-3-3.json
```

`stage09_mps.py` retains checkpoints at steps 0, 25, 50, 75, and 100. Checkpoints and reports stay in ignored output directories. The evaluator writes detailed JSON and a flat CSV. Every cell uses identical fixed validation batches, sampled with replacement. Defaults use eight batches of two rows, data seed 2027, and mask seeds 11/23/37. The two asymmetric cells `(1,3)` and `(3,1)` each have three distinct placements; all other pilot cells have one. The default evaluates all placements, without repeating deterministic cells to manufacture replication. Standard deviations describe variation across placements only.

Reports include paired NLL differences and argmax prediction changes relative to `(0,0)` in the same checkpoint. That reference is not a separately trained baseline. Forward matrix-multiply FLOPs include actual held-state reads; the report lists excluded operations, so these estimates must not be presented as complete compute accounting. State RMS by pass and parameter-group gradient norms come from one separate eval-mode backward probe per cell, using the first fixed batch and first placement. Unused groups have null gradients. Diagnostics leave model weights and `.grad` buffers unchanged. `--no-diagnostics` skips these probes.

The experiment workflow and exact commands are in [the MPS agent handoff](docs/STAGE09_MPS_HANDOFF.md). [Stage 9 validation](docs/STAGE_09_VALIDATION.md) separates implemented tooling from experiments still to run.
