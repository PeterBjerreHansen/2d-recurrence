# Deep-supervision training ablation

This is the first objective ablation after the completed recurrence pilot and 10,000-update baseline. It compares the existing final-only objective with one normalized auxiliary-loss condition:

```text
L = (L_final + 0.25 * mean(L_intermediate)) / 1.25
```

The auxiliary mean contains the nonfinal pass predictions for multipass trajectories. One-pass trajectories use `L_final` alone. The auxiliary prediction path reuses a source activation when a temporal write already needs it; it does not alter state writes or evaluation.

An auxiliary weight of zero is treated as final-only execution: intermediate
source/coda/readout forwards are skipped, so it does not consume additional
dropout RNG. The weight must be a finite nonnegative numeric value; Boolean
values are rejected. The normalized objective is retained as the experiment
definition and is not treated as a bug.

The pair is matched on architecture A, full `chess_long_v1`, optimizer, learning-rate schedule, recurrence distribution, model seed, schedule seed, batch, panel, and 10,000-update budget. The backend/device is not matched: the auxiliary run used MPS and the reused final-only control used CUDA on an A6000. The final-only condition is the control. The completed architecture-site A run at [`experiments/ablations/architecture_sites/results/separated`](../architecture_sites/results/separated) already is this exact final-only control, so it is reused rather than retrained. This is one matched seed, as specified by the current plan; it is not a seed-variance study.

Run from the repository root:

```sh
uv run pytest -q
uv run python train.py experiments/ablations/deep_supervision/configs/auxiliary_lambda025.py
```

Both runs retain checkpoints at 0, 1,000, 2,000, 5,000, 8,000, and 10,000. Evaluate the retained checkpoints with the existing nine-cell evaluator and the same selection/confirmation panels as the completed long baseline. Evaluation remains final-only, so the reported surface is comparable across objectives.

The experiment must be interpreted by absolute NLL and accuracy first, then recurrence differences and estimated compute. Auxiliary-loss metrics in `metrics.jsonl` are training diagnostics; evaluation NLL is the decision metric.

The historical timing comparison is not a valid overhead estimate because
auxiliary supervision ran on MPS while the reused final-only control ran on
CUDA/A6000. The NLL comparison is also cross-backend. Do not use those numbers
to claim a speed ratio or an objective effect. A matched same-backend benchmark
is required before making that comparison.

The completed selection-panel result is in [REPORT.md](REPORT.md). The large confirmation split was intentionally deferred because it contains 1,317 rows and is much more expensive than the 128-row selection panel.
