# 1B time-dependent recurrence study

This is a new six-arm study. It does not modify the historical 1B baseline or
the 5B recurrence-axis study.

The hard-growth arms use two executed passes (`U=1`) through absolute optimizer
step 8309 and four executed passes (`U=3`) from step 8310. The six arms are:

- `depth_fixed`
- `depth_hard_2to4`
- `temporal_fixed`
- `temporal_hard_2to4`
- `hybrid_fixed`
- `hybrid_hard_2to4`

The three fixed controls (`depth_fixed`, `temporal_fixed`, and `hybrid_fixed`)
all use the same `[p_K1, p_K2, p_K4] = [0.00, .85, .15]` pass-count
distribution as the hard arms over the full training horizon. `temporal_fixed`
is the temporal-only control. Its hard-growth counterpart,
`temporal_hard_2to4`, uses the same marginal distribution but moves all K=4
exposure to the final phase.

The expected executed pass count is `2.3`. Because 9,776 updates cannot split
exactly at 85%, the frozen fixed matrices use the exact horizon-weighted values
`P(K=2)=8310/9776` and `P(K=4)=1466/9776`, which round to `.85/.15` and match
the hard schedule exactly.

The protocol is defined in Python because this repository has no
repository-owned YAML configuration files. `run.py freeze` writes the resolved
configuration and source/data hashes to `results/protocol.json` before any
training starts.
