# Experiments

Run modules from the repository root. Each experiment owns an ignored `results/` directory; datasets remain shared under `data/`. There is no global results directory.

| Experiment | Status and role |
| --- | --- |
| [1B pair](long_runs/1B_baseline/LONG_BASELINE_RESULTS.md) | Completed matched transformer / recurrent A pair using the selected final-only objective |
| [Architecture sites](ablations/architecture_sites/README.md) | Retained completed A/B comparison; A is the practical default |
| [Deep-supervision comparison](ablations/deep_supervision/README.md) | Controlled A6000 comparison; final-only selected |
| [Baseline LR selection](sweeps/baseline_lr_selection/README.md) | Retained LR sweep and selected 10k continuation |
| [5B recurrence-axis study](long_runs/5B_axis/README.md) | Frozen four-arm protocol; depth, temporal, and hybrid CUDA runs retained, transformer result incomplete |
| [1B time-dependent update schedule](ablations/1B_update_schedule/README.md) | New six-arm ablation protocol for fixed versus hard 1-to-3 update growth; freeze before launch |
| [Temporal gate initialization](ablations/temporal_gate_init/README.md) | Completed 250M two-arm preflight; `.25` was only marginally ahead, so the 20B study kept `.10` |
| [20B recurrence-axis study](long_runs/20B_recurrence/README.md) | Completed four-arm run; results, live-execution diagnosis and post-hoc alignment in the [report](long_runs/20B_recurrence/REPORT.md) |
| [Evaluation battery](evaluation_battery/README.md) | Post-training battery for the 20B arms: training-graph and live NLL, legal-move probability, stratified paired comparisons, same-host cost |
| [Live temporal feedback](ablations/live_warm_start/PLAN.md) | Post-hoc live alignment ([`align.py`](ablations/live_warm_start/align.py), used in the [20B report](long_runs/20B_recurrence/REPORT.md)) and the open follow-ups |
| [Smoke checks](smoke/README.md) | Small reproducible pipeline checks and historical validation notes |

## Serious profile

`serious.py` is the shared source for batch-100 runs: full `lichess_6gb_blocks.zip` pinned to revision `1a932e1abca935aae585f417ede39ecde4f2a620`, upstream 1% split with seed 2357, context 1,023, eight blocks, width 512, eight heads, AdamW 3e-4 to 3e-5, betas .9/.95, weight decay .1, clipping 1, dropout 0. The physical batch is 5 with 20 accumulation steps. CUDA BF16 and eager execution apply to both models. Precision and microbatch feasibility are benchmarked on the target GPU before an experiment is frozen (A6000 for the 1B series, RTX 4090 for the 20B study); any needed revision is shared. These are comparable reference settings, not a claim of bitwise reproduction of upstream software.

The 1B label counts target characters rounded up to complete updates: 9,776 updates / 1,000,084,800 characters. The paired runs are data-matched; training GPU time is reported separately. The 20B study builds its configurations on the same base (`serious.base`) with its own schedule.

`run_serious.py` performs preflight, benchmarking, frozen budget creation, resumable ablation training, exact-panel evaluation, explicit supervision selection, and a foreground paired queue. It never provisions hardware or invents a supervision result. See the [handoff](ablations/deep_supervision/HANDOFF.md). The ablation defaults to equal time corresponding to 250M characters in its final-only arm; its LR follows consumed training time. A changed physical batch changes recurrence schedule averaging and requires a new shared protocol.

`configs/local/recurrent_mps.py` and `configs/local/transformer_mps.py` retain affordable batch-8 local checks. They are not serious-run comparators. The old ambiguous top-level baseline configs and unlaunched pilot directories were removed.

## Retained evidence

The architecture ablation, LR sweep, and supervision comparison keep their source, protocol, and local raw artifacts. Source-freeze checks may reject rerunning historical experiments under current code; use the preserved source archive for exact reproduction. The new series has separate output paths.

The early 100/1,000-update pilots and old smoke checkpoints were retired to reduce clutter. Their archive directory was removed in commit `739d45d` and remains available in Git history. [relocations.json](relocations.json) preserves old paths and marks retired records. Historical JSON and checkpoint contents are not rewritten to look like new runs.

The 1B pair is complete and documented in [LONG_BASELINE_RESULTS.md](long_runs/1B_baseline/LONG_BASELINE_RESULTS.md). The temporal-only, depth-only, and matched-hybrid schedule definitions are owned by the [5B recurrence-axis study](long_runs/5B_axis/README.md), alongside its runner and resolved configuration entry points.
