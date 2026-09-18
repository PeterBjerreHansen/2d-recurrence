# Long baseline with initial LR selection

Completed MPS experiment: two matched 1k-update LR candidates, followed by the selected 3e-4 candidate continuing to 10k. These are optimization candidates, not the A/B architectures. Both retain the original 2/0/4/1/1 layout. See [results](REPORT.md), [plan](PLAN.md), and the historical [handoff](HANDOFF.md).

`configs/lr3e4.py` and `configs/lr1e4.py` retain the training protocol. `panel.json` is an exact copy of the original frozen input. `results/` contains both candidate directories, selection receipt, original panel copy, provenance patch, grids, generation reports, and plots. The winning continuation belongs to this same experiment; no checkpoint copies are needed to split selection from continuation.

`select.py` applies the original rule and refuses to overwrite an existing selection. `freeze_provenance.py` is for a fresh protocol, not rewriting completed provenance. Rebuild figures with `uv run --with matplotlib python -m experiments.long_runs.baseline.curves` from the repository root. A historical resume must retain its model and training settings; panel relocation is checked by content hash.
