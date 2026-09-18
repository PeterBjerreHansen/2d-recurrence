# Initial recurrence-grid sweep

Historical 100-update runs with seeds 1337/1338, using the original 2/0/4/1/1 layout. This was a learning and pipeline check, not a long-training comparison. See [results](REPORT.md), [validation](VALIDATION.md), and the historical [handoff](HANDOFF.md).

`config.py` pins that original layout. `results/seed1337/` and `results/seed1338/` retain checkpoints and per-step grids; `results/` also contains full-row evaluation, feedback diagnostics, prompts, validation artifacts, and `curves/`. Rebuild plots with `uv run --with matplotlib python -m experiments.sweeps.recurrence_grid.curves` from the repository root.
