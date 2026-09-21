# recurrent_a 64B

Prepared, not launched. The shared serious protocol is in `experiments/serious.py`; the supervision decision must exist before either member of the pair runs. This configuration processes 625,611 optimizer updates at effective batch 100 and context 1,023: 64,000,005,300 target characters. The name denotes a rounded-up data budget, not FLOPs or GPU-hours. The 64B series is slightly larger than Karvonen's 600,000-update / 61.38B-character reference.

Use `uv run python -m experiments.run_serious pair --billions 64` from the repository root to run/resume the paired series with frozen provenance. This explicitly queues transformer then A; it does not provision hardware. `config.py` also describes the resolved trainer settings, but prefer the runner for environment checks and per-run plan receipts. All output is local to `results/` here.

The two models share data order, batch, precision, optimizer and token schedule. The ordinary model has eight distinct blocks; A uses the default 1/1/4/1/1 layout and the selected supervision mode. Both process the same number of target characters; recurrent passes and auxiliary predictions do not count as new data. Compute is reported separately. Warmup is 2% of updates capped at the reference 2,000; cosine decay reaches 3e-5 at the planned endpoint. A 64B run starts from scratch with its own LR horizon. Do not resume a shorter completed cosine schedule as though it were this experiment.

Future expanded-grid and temporal-only/depth-only comparisons follow this baseline selection. They are intentionally not silently included in this queue: their inference semantics and matched update schedules must be implemented and reviewed first.
