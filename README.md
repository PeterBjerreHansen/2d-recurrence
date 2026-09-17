# Two-axis recurrent ChessGPT

Stages 0 and 1 implement the ordinary character-level chess baseline. The model is the eight-layer, eight-head, width-512 Karvonen/nanoGPT transformer with learned positions and tied embedding/unembedding weights. Recurrence is not implemented yet. The later 2/4/1/1 partition is specified in [the proposal](proposal.md) and [implementation plan](implementation_plan.md).

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
