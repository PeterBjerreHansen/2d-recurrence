# Recurrence axis ablation

This directory is a new-run scaffold for comparing the two recurrence axes:

```text
temporal-only:  (K, 0)
depth-only:     (0, K)
matched hybrid: (K, K)
```

All three conditions use the same support and the same distribution over the
active update count `K`: 0.10 at 0, 0.50 at 1, and 0.40 at 3. The active write
mask is therefore deterministic for the matched cells. The physical backbone,
dataset, and shared training settings come from `experiments.serious.base`.

These files define comparable component conditions; they do not constitute a
final frozen scientific protocol. Training horizon, supervision mode,
hardware/time matching, and dataset exposure still require an explicit review.
No expensive run is launched by this scaffold.

Configurations:

- `configs/temporal.py`: temporal recurrence only, evaluated at `(3, 0)`.
- `configs/depth.py`: depth recurrence only, evaluated at `(0, 3)`.
- `configs/hybrid_matched.py`: diagonal hybrid schedule, evaluated at `(3, 3)`.

Outputs belong under this experiment's ignored `results/` directory.
