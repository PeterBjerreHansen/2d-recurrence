# Temporal gate initialization preflight

This is a small, two-arm ablation before the long recurrent runs. It keeps
the serious recurrent architecture, data, update distribution, optimizer,
seeds, precision, and evaluation protocol fixed. The only changed setting is
the initial temporal-memory gate coefficient:

| Arm | Initial `alpha` | Initial `beta` |
| --- | ---: | ---: |
| `conservative` | 0.10 | 0.90 |
| `higher_memory` | 0.25 | 0.75 |

The run is 250,000,000 target characters, rounded to 2,444 optimizer updates
and 250,021,200 actual characters, with 49 warmup updates. It is diagnostic,
not a selection by tiny NLL differences. Inspect the `(0,0)`, `(1,1)`, and
`(3,3)` cells, gate means, temporal-memory and anchor contribution RMS, pass
state RMS/change, clipping, and finite-value behavior.

Run from the repository root:

```text
uv run python -m experiments.ablations.temporal_gate_init.run freeze
uv run python -m experiments.ablations.temporal_gate_init.run train conservative
uv run python -m experiments.ablations.temporal_gate_init.run train higher_memory
for arm in conservative higher_memory; do
  for step in 0 100 500 1000 1500 2000 2444; do
    uv run python -m experiments.ablations.temporal_gate_init.run evaluate "$arm" --step "$step"
  done
done
uv run python -m experiments.ablations.temporal_gate_init.run summarize
```

The existing default remains `alpha=.10`, so this experiment does not alter
historical runs or silently change the long-run configuration.
