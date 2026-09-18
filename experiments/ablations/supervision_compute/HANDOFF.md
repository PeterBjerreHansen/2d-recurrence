# Execute supervision selection, then the 1B pair

Use one spot RTX A6000 with CUDA, Python 3.11 and the frozen uv environment. Provisioning and spending are separate from these commands. Run from the repository root, in a persistent terminal. Do not launch a 64B run as part of this handoff. Retain experiment-local results on durable storage and transfer them before deleting the VM.

Read `experiments/serious.py` and this experiment's README. The first job is the compute-matched final-only versus deep-supervision comparison; do not assume deep supervision has already won. Do not reuse old MPS timings. Run the tests before freezing source.

```sh
uv sync --frozen --python 3.11
uv run pytest -q
uv run python data/chess_v1/prepare.py --file lichess_6gb_blocks.zip --revision 1a932e1abca935aae585f417ede39ecde4f2a620 --out-dir data/chess_full_v1
uv run python -m experiments.run_serious preflight
uv run python -m experiments.run_serious benchmark final
uv run python -m experiments.run_serious benchmark deep
uv run python -m experiments.run_serious benchmark transformer
uv run python -m experiments.run_serious benchmark deep_more
uv run python -m experiments.run_serious freeze --reference-characters 250000000
uv run python -m experiments.run_serious ablation final
uv run python -m experiments.run_serious ablation deep
uv run python -m experiments.run_serious evaluate experiments/ablations/supervision_compute/results/final/ckpt.pt
uv run python -m experiments.run_serious evaluate experiments/ablations/supervision_compute/results/deep/ckpt.pt
```

Skip preparation if the complete pinned dataset is already present and preflight validates it. The preparation routine refuses to overwrite existing data. Benchmarks use fresh scratch outputs and refuse to mix repeated timing samples. Report mean/median update time, memory, deep/final overhead, and projected 1B/64B times. If microbatch 5 or BF16 fails, stop and fix the shared profile for all conditions; do not silently alter one run. Before any substantive training, check the forecast against the allocated external compute budget.

After examining the two reports, record the choice with a substantive reason (replace the example mode/reason with the actual decision):

```sh
uv run python -m experiments.run_serious choose deep --reason='Replace with the measured equal-time result and tradeoff.'
uv run python -m experiments.run_serious evaluate experiments/ablations/supervision_compute/results/deep/ckpt.pt --split confirmation
uv run python -m experiments.run_serious pair --billions 1
```

If final-only wins, use `choose final` and confirm the `final/ckpt.pt` endpoint instead. The pair command is an explicit foreground queue: transformer first, then A, each reaching at least 1B character predictions, with an exact selection-panel evaluation after each. Re-running the same command resumes an interrupted member and skips training work already completed. It never follows on to 64B. Run independent confirmation evaluations for the completed pair when ready to report results.

Report actual training seconds, characters, optimizer updates, wall time including interruptions, validation curves, final grid results and supervision choice. The 1B pair is data-matched, not time-matched; use the logged compute axis as well. Keep `environment.json`, `protocol.json`, `panel.json`, benchmark receipts, `decision.json`, per-run `plan.json`, logs and checkpoints. A changed source/environment requires a new protocol rather than bypassing the freeze checks.
