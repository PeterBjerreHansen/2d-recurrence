# RunPod execution helpers

These helpers support running `5B_axis` arms on separate RTX 3090 pods.
They do not modify the frozen A6000 study protocol or the resolved model and
optimizer settings.

## Batch-size probe

Run `batch_sweep.sh` once on a 3090 before starting the long arms. It uses the
hybrid `(3,3)` graph as the conservative case and tests microbatches `5, 10,
20, 25, 50` by default. Every candidate keeps effective batch size 100 through
gradient accumulation. Set `BATCH_SWEEP_STEPS` or `BATCH_SIZES` to change the
probe, but use fresh output directories; the script refuses to overwrite an
existing sample.

The probe writes `summary.json`, `summary.tsv`, per-candidate `metrics.jsonl`,
and sampled `gpu_memory.csv`. It is a throughput and memory measurement, not a
quality comparison. Select one stable microbatch for all three arms before
launching the long runs.

## Long arms

From the repository root on each pod:

```sh
setsid bash experiments/long_runs/5B_axis/runpod/run_arm.sh temporal \
  > experiments/long_runs/5B_axis/temporal_5B/results/runpod.log 2>&1 < /dev/null &
```

Use `depth` and `hybrid` on the other pods. The helper chooses `scratch` when
the arm has no checkpoint and `resume` when `ckpt.pt` exists. All checkpoints,
logs, and the pre-training `runpod_execution.json` receipt stay in that arm's
result directory.

Set `BATCH_SIZE` to the selected probe value when launching all three arms. The
helper derives `gradient_accumulation_steps=100/BATCH_SIZE` and records both
values. If an existing checkpoint uses a different pair, the helper refuses to
resume it instead of silently changing the optimization batch.

The original `run.py` preflight is intentionally not used here because its
frozen receipt requires an RTX A6000. The wrapper invokes the unchanged
`train.py` arm configuration directly and records the source commit, resolved
configuration, dataset manifest hash, protocol hash, CUDA runtime, and resume
mode before training.

Use one persistent volume per pod/arm, or otherwise give each arm an isolated
output directory. Do not let multiple pods write the same checkpoint.
