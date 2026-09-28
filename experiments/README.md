# Experiments

Run modules from the repository root. Each experiment owns an ignored `results/` directory; datasets are shared under `data/`.

## Main study

| Experiment | What it is |
| --- | --- |
| [20B recurrence-axis study](long_runs/20B_recurrence/README.md) | Four separately trained arms; results in the [report](long_runs/20B_recurrence/REPORT.md) |
| [Evaluation battery](evaluation_battery/README.md) | Paired training-graph and live comparisons, with legal-move and per-character breakdowns |
| [Live temporal feedback](ablations/live_warm_start/PLAN.md) | Aligning temporal memory with live execution (`align.py`, `aligned_decay.py`) and the open follow-ups |
| [1B update schedule](ablations/1B_update_schedule/README.md) | Prepared ablation of fixed versus growing update schedules; not run |

## Decisions the 20B study builds on

| Experiment | Decision |
| --- | --- |
| [Architecture sites](ablations/architecture_sites/README.md) | Layout A (a near tie with B) |
| [Deep supervision](ablations/deep_supervision/README.md) | Final-pass loss only |
| [Baseline LR selection](sweeps/baseline_lr_selection/README.md) | Peak learning rate 3e-4 |
| [Temporal gate initialization](ablations/temporal_gate_init/README.md) | Gate initialised at 0.10 |

Earlier runs ([1B pair](long_runs/1B_baseline/LONG_BASELINE_RESULTS.md), [5B study](long_runs/5B_axis/README.md)) and [smoke checks](smoke/README.md) are kept in their folders.

## Shared profile

[`serious.py`](serious.py) holds the settings every batch-100 run shares:
- **Data:** the full `lichess_6gb_blocks.zip` with the upstream 1% split.
- **Model:** context 1,023, eight blocks, width 512, eight heads, dropout 0.
- **Optimiser:** AdamW from 3e-4 to 3e-5, betas 0.9/0.95, weight decay 0.1, clipping 1.0.
- **Batching:** 5 × 20 microbatches, in CUDA BF16.

The 20B study builds its configurations on `serious.base`. Small local configs for Apple MPS are in `configs/local/`.
