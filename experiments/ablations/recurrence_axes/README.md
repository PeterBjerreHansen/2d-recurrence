# Recurrence axis ablation

This directory is a new-run scaffold for comparing the two recurrence axes:

```text
temporal-only:  (K, 0)
depth-only:     (0, K)
near-diagonal hybrid: mostly `(K, K)`, with declared asymmetric cases
```

The temporal-only and depth-only conditions use the same support and the same
distribution over the active update count `K`: 0.10 at 0, 0.50 at 1, and 0.40
at 3. The hybrid uses the fixed symmetric table below. It preserves the same
distribution over `max(U_T, U_D)`, keeps 80% of each nonzero bucket on the
diagonal, and supplies declared asymmetric state-age cases without adding a
new update-count support.

```text
             U_D=0  U_D=1  U_D=3
U_T=0          .10     .05     .01
U_T=1          .05     .40     .03
U_T=3          .01     .03     .32
```

The physical backbone, dataset, and shared training settings come from
`experiments.serious.base`. This is pass-matched by core-pass count, not by
the sum `U_T + U_D` or by each axis marginal.

These files define comparable component conditions; they do not constitute a
final frozen scientific protocol. Training horizon, supervision mode,
hardware/time matching, and dataset exposure still require an explicit review.
No expensive run is launched by this scaffold.

Configurations:

- `configs/temporal.py`: temporal recurrence only, evaluated at `(3, 0)`.
- `configs/depth.py`: depth recurrence only, evaluated at `(0, 3)`.
- `configs/hybrid_matched.py`: near-diagonal hybrid schedule, evaluated at `(3, 3)`.

Outputs belong under this experiment's ignored `results/` directory.
