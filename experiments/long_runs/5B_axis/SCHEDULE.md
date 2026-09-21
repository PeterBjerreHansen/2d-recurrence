# 5B recurrence-axis schedule

These schedule definitions belong to the `5B_axis` study. They are kept next
to the study runner so the protocol, configs, and transfer bundle have one
owner.

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

The physical backbone, dataset, and shared training settings come from the
5B study's frozen `study.py` configuration. This is pass-matched by core-pass
count, not by the sum `U_T + U_D` or by each axis marginal.

The three config entry points delegate to `study.run_config`, so they resolve
to the same full 5B settings used by the resumable runner. They do not launch
training by themselves.

Configurations:

- `configs/temporal.py`: temporal recurrence only, evaluated at `(3, 0)`.
- `configs/depth.py`: depth recurrence only, evaluated at `(0, 3)`.
- `configs/hybrid_matched.py`: near-diagonal hybrid schedule, evaluated at `(3, 3)`.

Outputs belong under this experiment's arm-specific ignored `results/`
directories.
