# 20B recurrence-axis study

Status: **completed.** Results are in the [report](REPORT.md); the pre-registered protocol and question are in the [experiment plan](../../../docs/20B_experiment_plan.md).

The study trains four separate arms of the same 8-layer, width-512 model on 20,000,059,200 characters each (195,504 updates of batch 100). The arms are `transformer`, `temporal`, `depth` and `hybrid`, and they share one data order.

- **Recurrence:** the recurrent arms follow the update-count curriculum in [`study.py`](study.py), over the support {0, 1, 3}, toward four passes. The hybrid puts 80% of each nonzero bucket on the diagonal.
- **Learning rate:** warmup–stable–decay. Warmup takes 2,000 updates, the rate is constant to step 175,954, then decays linearly to `3e-5`.
- **Checkpoints:** retained at 1B, 2B, 4B, 8B, 10B, 14B, 16B, 18B and 20B.
  - At the major ones (1B, 4B, 10B, 18B and 20B), the training graph is evaluated at 1, 2, 4, 8 and 16 passes. Live NLL is also recorded on the 128-row selection panel, with stress checks.
  - The others record only the four-pass cell and the `(0,0)` reference.

## What ran

Each arm ran on its own Secure RTX 4090 on Runpod, from one frozen transfer bundle. The [`ops`](ops/README.md) tool acquired, started, monitored, collected and released the pods on an hourly tick.

- **Restarts:** none; no arm needed a resume.
- **Integrity:** every arm passed the frozen-protocol check after collection.
- **Training time:** 23–54 hours per arm, depending on the host.
- **Spend:** $134.73 in total.
- **Details:** in the report's appendix.

Post-training work lives elsewhere:
- **Evaluation:** the [evaluation battery](../../evaluation_battery/README.md).
- **Live alignment of temporal:** [`live_warm_start`](../../ablations/live_warm_start/PLAN.md).

## Commands

Run from the repository root.

```sh
uv run python -m experiments.long_runs.20B_recurrence.run dry-run          # resolved configs, no side effects
uv run python -m experiments.long_runs.20B_recurrence.run freeze           # record the protocol (refused once an arm exists)
uv run python -m experiments.long_runs.20B_recurrence.run status
uv run python -m experiments.long_runs.20B_recurrence.package --output /path/to/20B_recurrence.tar.gz
uv run python -m experiments.long_runs.20B_recurrence.run preflight <arm>  # host receipt in <arm>_20B/results/environment.jsonl
uv run python -m experiments.long_runs.20B_recurrence.run train <arm> [--resume]
uv run python -m experiments.long_runs.20B_recurrence.run evaluate <arm> --step <step>
uv run python -m experiments.long_runs.20B_recurrence.run integrity <arm> [--complete]
```

- **The frozen protocol** records source hashes, dataset, panel, all four resolved configurations, the curriculum and the evaluation definitions.
- **`train <arm>`:** starts fresh only if no checkpoint exists.
- **`--resume`:** continues a valid checkpoint and never starts fresh.
- **Moving hosts:** an arm may move to another host only if the GPU model, the Torch/CUDA build, Python and package versions all match its first host.
- **`study`:** trains all arms sequentially on one host. The 20B run instead used one `train <arm>` per pod.

## Interpretation limits

This is a one-seed run, matched in data exposure and in the distribution of executed passes. It is not matched in parameters or compute, so compute estimates stay attached to every NLL comparison.

Checkpoints along one trajectory are not independent replications. Differences below about 0.002 NLL should be read as ties.
