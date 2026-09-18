# Longer recurrence pilot

Historical two-seed 1,000-update continuation on the small smoke dataset, using the original 2/0/4/1/1 layout. See the [report](REPORT.md). `config.py` retains those settings; it does not inherit the new architecture default.

`results/` holds both seeds, the preserved partial run, complete-row evaluation, and curves. Rebuild plots with `uv run --with matplotlib python -m experiments.long_runs.recurrence_pilot.curves` from the repository root.
