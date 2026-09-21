# Experiments

Run modules from the repository root. Each experiment owns an ignored `results/` directory; datasets remain shared under `data/`. There is no global results directory.

| Experiment | Status and role |
| --- | --- |
| [Transformer 1B](long_runs/1B_baseline/transformer_1B/README.md) / [A 1B](long_runs/1B_baseline/recurrent_a_1B/README.md) | Completed matched 1B pair using the selected final-only objective |
| [Transformer 64B](long_runs/64B_core/transformer_64B/README.md) / [A 64B](long_runs/64B_core/recurrent_a_64B/README.md) | Prepared reference-scale profiles; no 64B run launched |
| [Architecture sites](ablations/architecture_sites/README.md) | Retained completed A/B comparison; A is the practical default |
| [Deep-supervision comparison](ablations/deep_supervision/README.md) | Controlled A6000 comparison; final-only selected |
| [Baseline LR selection](sweeps/baseline_lr_selection/README.md) | Retained LR sweep and selected 10k continuation |
| [5B recurrence-axis study](long_runs/5B_axis/README.md) | Frozen four-arm protocol; depth, temporal, and hybrid CUDA runs retained, transformer result incomplete |
| [1B time-dependent update schedule](ablations/1B_update_schedule/README.md) | New six-arm ablation protocol for fixed versus hard 1-to-3 update growth; freeze before launch |
| [Archived early pilots](archive/early_pilots/README.md) | Reports and small artifacts retained; obsolete scripts and checkpoints deleted |
| [Smoke checks](smoke/README.md) | Small reproducible pipeline checks and historical validation notes |

## Serious profile

`serious.py` is the shared source for batch-100 runs: full `lichess_6gb_blocks.zip` pinned to revision `1a932e1abca935aae585f417ede39ecde4f2a620`, upstream 1% split with seed 2357, context 1,023, eight blocks, width 512, eight heads, AdamW 3e-4 to 3e-5, betas .9/.95, weight decay .1, clipping 1, dropout 0. The physical batch is 5 with 20 accumulation steps. CUDA BF16 and eager execution apply to both models. Precision and microbatch feasibility must pass the A6000 benchmark; any needed revision is shared and made before the experiment is frozen. These are comparable reference settings, not a claim of bitwise reproduction of upstream software.

The 1B and 64B labels count target characters rounded up to complete updates: 9,776 updates / 1,000,084,800 characters, and 625,611 updates / 64,000,005,300 characters. Karvonen's exact published schedule is 600,000 updates / 61.38B characters. The paired runs are data-matched; training GPU time is reported separately. Each horizon starts from scratch and has its own cosine schedule. Warmup is 2% capped at 2,000 updates.

`run_serious.py` performs preflight, benchmarking, frozen budget creation, resumable ablation training, exact-panel evaluation, explicit supervision selection, and a foreground paired queue. It never provisions hardware, invents a supervision result, or launches the 64B series after 1B. See the [handoff](ablations/deep_supervision/HANDOFF.md). The ablation defaults to equal time corresponding to 250M characters in its final-only arm; its LR follows consumed training time. A changed physical batch changes recurrence schedule averaging and requires a new shared protocol.

`configs/local/recurrent_mps.py` and `configs/local/transformer_mps.py` retain affordable batch-8 local checks. They are not serious-run comparators. The old ambiguous top-level baseline configs and unlaunched pilot directories were removed.

## Retained evidence

The architecture ablation, LR sweep, and supervision comparison keep their source, protocol, and local raw artifacts. Source-freeze checks may reject rerunning historical experiments under current code; use the preserved source archive for exact reproduction. The new series has separate output paths.

The early 100/1,000-update pilots and old smoke checkpoints were retired to reduce clutter. Their reports and small artifacts remain, but obsolete launch scripts and large checkpoints do not. [relocations.json](relocations.json) preserves old paths and marks retired records. Historical JSON and checkpoint contents are not rewritten to look like new runs.

The 1B pair is complete and documented in [LONG_BASELINE_RESULTS.md](long_runs/1B_baseline/LONG_BASELINE_RESULTS.md). The temporal-only, depth-only, and matched-hybrid schedule definitions are owned by the [5B recurrence-axis study](long_runs/5B_axis/README.md), alongside its runner and resolved configuration entry points.
