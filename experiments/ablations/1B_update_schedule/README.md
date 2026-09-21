# 1B time-dependent update schedule study

This is a new six-arm study. It does not modify the historical 1B baseline or
the 5B recurrence-axis study.

All six arms use a constant learning rate of `3e-4`, with no warmup or decay.
The crossover therefore changes only the update-count distribution; it does
not coincide with a learning-rate change.

The hard-growth arms use one update (`U=1`, two executed passes) through
absolute optimizer step 8309 and three updates (`U=3`, four executed passes)
from step 8310. The six arms are:

- `depth_fixed`
- `depth_hard_1to3`
- `temporal_fixed`
- `temporal_hard_1to3`
- `hybrid_fixed`
- `hybrid_hard_1to3`

The three fixed controls (`depth_fixed`, `temporal_fixed`, and `hybrid_fixed`)
all use the same `[p_U0, p_U1, p_U3] = [0.00, .85, .15]` max-update-count
distribution as the hard arms over the full training horizon. `temporal_fixed`
is the temporal-only control. Its hard-growth counterpart,
`temporal_hard_1to3`, uses the same marginal distribution but moves all U=3
exposure to the final phase.

The expected max update count is `1.3`. Because 9,776 optimizer updates cannot
split exactly at 85%, the frozen fixed matrices use the exact horizon-weighted
values `P(U=1)=8310/9776` and `P(U=3)=1466/9776`, which round to `.85/.15` and
match the hard schedule exactly. These correspond to two and four physical
passes, respectively.

The protocol is defined in Python because this repository has no
repository-owned YAML configuration files. `run.py freeze` writes the resolved
configuration and source/data hashes to `results/protocol.json` before any
training starts.
