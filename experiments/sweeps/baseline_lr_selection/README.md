# Baseline selection and continuation

Completed MPS experiment: a two-candidate 1k-update learning-rate selection sweep, followed by the selected 3e-4 candidate continuing to 10k. This is a selection record plus a continuation, not a generic “long run” or the A/B architecture comparison. Both candidates retain the original 2/0/4/1/1 layout. See [results](REPORT.md), [plan](PLAN.md), and the historical [handoff](HANDOFF.md).

`configs/lr3e4.py` and `configs/lr1e4.py` retain the training protocol. `panel.json` is an exact copy of the original frozen input. `results/` contains both candidate directories, selection receipt, original panel copy, provenance patch, grids, generation reports, and plots. The winning continuation belongs to this same experiment; no checkpoint copies are needed to split selection from continuation.

`select.py` applies the original rule and refuses to overwrite an existing selection. `freeze_provenance.py` is for a fresh protocol, not rewriting completed provenance. Rebuild figures with `uv run --with matplotlib python -m experiments.sweeps.baseline_lr_selection.curves` from the repository root. A historical resume must retain its model and training settings; panel relocation is checked by content hash.
