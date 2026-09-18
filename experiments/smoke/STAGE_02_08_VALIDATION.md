# Stages 2–8 validation

Historical record of the completed experiment. Paths below follow the current repository layout; recorded configurations, measurements, and raw artifact provenance retain their original meaning. See the experiment README for current commands.

Implemented on `mvp-2d-recurrence`. The frozen `baseline-chessgpt` branch and `baseline-stage01` tag remain unchanged. The proposal and implementation plan remain in the repository. The maintained execution and initialization reference is [RECURRENCE_CONTRACT.md](../../docs/RECURRENCE_CONTRACT.md).

## Scope

The implementation adds the 2/4/1/1 backbone partition, explicit immutable write schedules, temporal and depth mixers, the dedicated temporal-source block, and full-gradient hybrid training. The trainer samples and broadcasts one schedule per microbatch and checkpoints the sampler alongside the existing training state. Fixed-pair evaluation is labeled `training_graph` and uses separate random generators. Live-feedback generation, grid experiments, and full compute accounting remain later stages.

The source and coda retain their original block positions and checkpoint names. At zero updates, logits, loss, and backbone gradients match the ordinary eight-block model exactly with matched weights and controlled dropout. The recurrent model has 27,814,912 parameters with the full pilot configuration, including positions and both mixers.

## Automated checks

`uv run pytest -q` passes 45 tests. They include the existing baseline, data, and chess-evaluation tests, plus:

- All nine pilot cells and asymmetric counts beyond that support, with finite losses and gradients, exact source/core/coda call counts, and causal logits.
- A hand-calculated noncommuting mixer example that distinguishes temporal-before-depth ordering, raw source feedback, held depth, and repeated one-position reads of held temporal memory.
- Exact comparison of held-state logits and parameter gradients against a history-based reference unroll, including an early and a latest-eligible temporal write. Single-axis reductions bypass the unused mixer.
- Gates reading normalized sources before value projection, exact first-position bypass, and valid zero-valued memories elsewhere.
- Exact mask constraints, empirical pair frequencies and uniform conditional placements over 12,000 seeded draws, and sampler resume independent of global randomness.
- Bit-for-bit model and optimizer resume with dropout, accumulation, and changing schedules, in both one-process and two-rank CPU training. Evaluation does not advance the training sampler.
- A two-rank gradient comparison with the equivalent single-process global batch. It covers mixed accumulated microbatches where parameters change from used to unused and vice versa, plus temporal-only/depth-only switches.

The distributed tests launch two local Gloo workers. They verify actual reduction and checkpoint behavior rather than assuming that schedule broadcast alone makes dynamic execution safe.

## Device and data checks

Both manual runs use the prepared 4,096-row real-data subset described in [stages 0–1 validation](STAGE_01_VALIDATION.md). The CPU smoke uses four small blocks, width 32, context 32, and 12 optimizer updates. Fixed $(3,3)$ validation NLL fell from 3.4584 to 3.3330.

```sh
uv run python train.py experiments/smoke/experiments/smoke/configs/recurrent_pilot.py
uv run python train.py experiments/smoke/configs/recurrent_pilot.py --out_dir=experiments/smoke/results/recurrent-mps-validation --max_iters=4 --eval_interval=2 --eval_iters=1 --log_interval=1 --warmup_iters=0
```

The MPS check uses all eight blocks, width 512, eight heads, context 1,023, batch size 2, and four accumulated microbatches per optimizer update. Fixed $(3,3)$ validation NLL was 3.7204 initially, 4.1273 after two updates, and 3.3338 after four. These few updates, with warmup disabled for this check, establish execution and finite learning behavior only. They do not establish a benefit from recurrence or a reliable learning curve. The committed pilot retains its ten-step warmup.

Local ignored artifacts are `experiments/smoke/results/recurrent-smoke/` and `experiments/smoke/results/recurrent-mps-validation/`, including configuration, provenance, metrics, and checkpoints. The checks were run with CPU/Gloo and Apple MPS, using PyTorch 2.14.0. CUDA/NCCL and mixed precision have not been exercised here. The frozen baseline checkpoint also remains loadable by the refactored trainer.

## Simplification review

Keep one stored copy of the backbone blocks and one short recurrent forward loop. Derive counts and pass length from immutable masks; storing both would create avoidable consistency rules. The row format makes only position zero lack predecessor memory, so a generic validity-mask interface adds no capability needed here. Keep the gate as one dense affine map plus sigmoid and use direct learned value projections. Additional controller networks, writers, state caches, and execution frameworks are unnecessary for the current question.

Deleted the unused inherited generation helper, context-cropping helper, parameter-count method, and A100 FLOP-utilization estimate. The supported chess sampler remains; the inherited helper duplicated it without its stopping policy. The old compute estimate would misrepresent recurrent work. Parameter totals are now recorded after constructing the complete model.

Three assumptions remain experimental: random write timing improves robustness; the chosen normalization and projection initialization make refinement easy to learn; additional recurrent computation improves predictions enough to justify its cost. Correct state routing and nonzero gradients do not establish any of these. In particular, normalized mixer inputs change the residual-stream scale relative to the absent-state path, so identity projections do not make recurrent execution an identity at initialization. Stage 9 should evaluate the nine-cell loss surface over training before adding gates, curricula, caching, or larger sweeps. No convergence claim follows from withholding iteration counts.
