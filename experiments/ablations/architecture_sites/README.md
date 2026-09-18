# Source and destination site ablation

Completed matched 10k-update CUDA runs on a spot RTX A6000. [The report](REPORT.md) describes the comparison; [the historical handoff](HANDOFF.md) records its protocol. A is now the repository default as a practical near-tie choice; B remains available. This does not change either completed run.

`configs/separated.py`, `configs/coincident.py`, and `configs/common.py` define the runs. `panel.json` is an exact copy of the original frozen selection/confirmation input. Checkpoints, all raw grids, plots, timing, receipts, and the original source transfer archive are in `results/`.

From the repository root:

```sh
uv run python -m experiments.ablations.architecture_sites.run summarize
uv run python -m experiments.ablations.architecture_sites.run benchmark
uv run python -m experiments.ablations.architecture_sites.run evaluate separated --step 10000
```

Summary and benchmark commands write `results/analysis/`; existing grid reuse verifies checkpoint, panel, and dataset hashes without requiring CUDA. Fresh evaluation or training still requires the frozen CUDA/source protocol. For an explicitly planned rerun, use `--results-dir experiments/ablations/architecture_sites/results/rerun-NAME` before the subcommand. Do not bypass a frozen-protocol mismatch to continue an old run.

`package.py` can build a new current-source transfer bundle with an explicit fresh `--output`; it excludes result directories. The existing `results/transfer.tar.gz` is the immutable source snapshot used in the completed experiment and intentionally contains the former repository layout.
