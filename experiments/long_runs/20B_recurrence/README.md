# 20B recurrence-axis study

This is a four-arm discovery study of a maximum four-pass training curriculum:
`transformer`, `hybrid`, `temporal`, and `depth`. Each arm processes 195,504
updates, or 20,000,059,200 target characters, with the same data order and
effective batch. Recurrent arms use the piecewise update-count curriculum in
`study.py`; the hybrid preserves the same distribution over
`max(U_T, U_D)` and puts 80% of each nonzero bucket on the diagonal.

The WSD learning-rate schedule is 2,000 warmup updates, a stable phase through
step 175,954, then linear cooldown to `3e-5` on the final optimizer update
(index 195,503; completed-step checkpoint 195,504). Checkpoint steps include
the curriculum boundaries and the 1B, 2B, 4B, 8B, 10B, 14B, 16B, 18B, and 20B
curve points. The minimum LR is exact on the final optimizer update.

At the five major checkpoints (1B, 4B, 10B, 18B, 20B), evaluation uses the
plan's explicit 1-, 2-, 4-, 8-, and 16-pass cells. The 2B, 8B, 14B, and 16B
curve checkpoints evaluate only each arm's primary four-pass cell (plus the
`(0,0)` reference). `study --evaluate-checkpoints` evaluates every retained
nonzero checkpoint after each arm completes. Sixteen-pass cells are
evaluation-only (training support remains `{0,1,3}`), and these grid cells
execute the training graph; live feedback is measured separately below.
Hybrid extrapolation uses the diagonal cells specified by the plan and does not
create a Cartesian grid. Major-checkpoint reports also include per-pass
activation RMS and finite-value stress checks from a single fixed batch. The
grid's gradient diagnostics are disabled above eight passes to avoid an
unnecessary full-unroll backward memory spike.

The four-pass primary cells remain required measurements. A non-finite optional
extrapolation cell or stress check is recorded in the report's
`failed_cells`/`diagnostic_failures` fields and does not invalidate primary
metrics or stop later study arms. Other evaluation or infrastructure errors
remain fatal.

Major checkpoints also record **live-feedback, teacher-forced NLL** on the
128-row selection panel: temporal at one core pass per token, hybrid at one
and four core passes, and depth at four. Live execution decodes each row
sequentially and carries temporal memory from the fully computed preceding
token, so it measures the deployed recurrence rather than the parallel
training graph. Depth and hybrid use per-depth KV caches; depth live at four
passes must equal its `(0,3)` cell and serves as a consistency check. The
transformer's live execution is its ordinary forward pass and is not
repeated. Live evaluation runs only on the selection split. On Apple MPS a
5B checkpoint took about 4 s per row at one pass and 7.5 s per row at four;
4090 timing has not been measured.

## Before freezing

The 250M temporal-gate preflight is complete. The `.25` initialization was
numerically ahead of `.10` by only about 0.001 NLL at the final `(0,0)`,
`(1,1)`, and `(3,3)` cells; this one-seed, short-run difference is not
decision-grade. Keep `.10`, the existing default, for the 20B protocol. A
dry-run previews the resolved configuration without changing files or
starting training:

```sh
uv run python -m experiments.long_runs.20B_recurrence.run dry-run
```

Freeze the complete protocol (gate initialization is fixed at `.10`):

```sh
uv run python -m experiments.long_runs.20B_recurrence.run freeze
uv run python -m experiments.long_runs.20B_recurrence.run status
```

`freeze` records source, dataset, panel, all four resolved configurations,
curriculum and evaluation definitions, fixed gate initialization, and environment.
Source file contents define identity; Git branch and commit are provenance.
Re-freezing a changed protocol requires
`--refresh` and is refused after any arm checkpoint exists.

## Hardware and recovery plan

The current plan is four Runpod Community RTX 4090 Pods, one arm per GPU, with
the reference physical batch of `5 × 20` (effective batch 100). The 200-update
RTX 4090 sweep measured only a 3.6% throughput gain for `10 × 10` over
`5 × 20`, at about 1.8× the peak VRAM and half as many schedule draws per
update. The short MPS concurrency probe did not show a useful end-to-end gain
from sharing one GPU across multiple arms. Keep one process per card.

The live deploy form showed, on 2026-09-24, a Community RTX 4090 at `$0.34/GPU-hour` with
**On-Demand** pricing and no Spot/interruptible option on that selected offer.
Community availability and the price must be checked again in the chosen
region before launch. Do not assume this is Spot pricing. The deterministic
1,000-update RTX 4090 check measured 0.845 seconds per update (about 1.18
updates/s). That projects to 46 hours of training work per arm; use 48 hours
for planning. The check used synthetic rows, so full-dataset I/O may change the
rate. Four cards at `$0.34/hour` for 48 hours cost `$65.28` for GPU time. The
monitor's storage estimate and 10% margin bring the planned balance to about
`$75.28`, under the `$80` cap. Evaluations, provisioning, and recovery are
included only through that margin.

Storage/recovery layout: run four Community RTX 4090 Pods, one arm per Pod.
Give each Pod a **30 GB Pod Volume Disk** at `/workspace` for its checkout,
local copy of the ~7.9 GB dataset, and arm outputs (comparable current result
directories are ~3.6–3.8 GB). Give each a separate **30 GB container disk**
for temporary upstream archives, package caches, and preparation scratch;
clear disposable staging data after preflight. Pod Volume Disks persist across
stop/restart and can be accessed through Runpod's [zero-GPU recovery mode](https://docs.runpod.io/pods/troubleshooting/zero-gpus),
but are tied to their Pod/host. Community Pods cannot mount Runpod [Network
Volumes](https://docs.runpod.io/storage/network-volumes) (Secure-Cloud-only),
and a Pod Volume Disk does not survive Pod termination or protect against
host/storage loss.

An hourly monitoring model is planned to check Pod/process state, progress,
logs, checkpoint hashes, frozen protocol receipts, disk health, and budget.
On GPU interruption, preserve the original Pod; if necessary, start it with
zero GPUs, retrieve the latest checkpoint, and transfer/verify it before
terminating or replacing the Pod. An external checkpoint-backup target must
be selected and tested before launch if recovery must cover host loss or Pod
replacement after the original volume becomes inaccessible. If the volume
remains accessible, verify and transfer the checkpoint plus frozen artifacts
to the replacement before resuming. The monitor must never start an arm fresh
when recovery data is missing. It should resume only the affected arm, guarded
by a per-arm lock so it cannot create duplicate workers, and alert rather than retry if a
checkpoint or hash is invalid, the protocol differs, failures repeat, or the
spend cap is reached. Transfer and SHA-256-verify each completed arm's
checkpoints and protocol/environment/panel files locally before releasing
the Pod and storage.

**Resume gate passed:** on 2026-09-25, a deterministic 1,000-update
WSD/curriculum comparison matched model, optimizer, scaler, recurrence sampler,
RNG, and every compared post-resume training metric exactly. LR formula error
was zero. It ran on Torch 2.14.0+cu130 on one Secure RTX 4090. Auto-review
blocked transfer of the local real-data sample to that Pod, so synthetic
32-token rows were generated there. This validates resume mechanics under the
locked CUDA stack, not data quality. The machine-readable result is at
`experiments/benchmarks/recurrent_runtime/results/20260925-4090-wsd-sanity-1000-deterministic-synthetic/sanity.json`.

The earlier nondeterministic run differed in model and optimizer state; a
100-update comparison also diverged without deterministic kernels and matched
bitwise with them. Production runs therefore resume with ordinary kernels,
where a resumed run is statistically, not bitwise, equivalent to an
uninterrupted one. The deterministic comparison command was:

```sh
uv run python -m experiments.benchmarks.recurrent_runtime --stage sanity \
  --updates 1000 --deterministic --run-id 20260925-4090-wsd-sanity-1000-deterministic
```

`EXACT_RESUME_GATE_PASSED` is enabled, so `train <arm> --resume` is allowed.
The full frozen dataset and host still need the normal preflight before the
20B run.
The trainer now saves the latest recovery checkpoint every 1,000 updates
without evaluation, evaluates every 10,000 updates (plus the initial and final
evaluation), and retains only the named curve checkpoints. The separate
post-training evaluation of retained curve checkpoints is unchanged. With
hourly polling, recovery could wait up to about an hour to be detected and
then replay up to one checkpoint interval of work.

Training requires complete `train.bin` and `val.bin` files and a CUDA
BF16-capable GPU per concurrent arm. The runner does not provision hardware.
The planned layout uses four isolated Community RTX 4090 Pods. The `study`
command below is the single-host sequential path; for the four-Pod layout,
launch one `train <arm>` command per Pod from the same frozen bundle, using a
separate durable volume/output directory for each arm. For a transfer run,
build a bundle after freezing:

```sh
uv run python -m experiments.long_runs.20B_recurrence.package --output /path/to/20B_recurrence.tar.gz
```

On the prepared host, install the locked environment, extract the bundle, and
run the commands from the repository root:

```sh
uv sync --frozen --python 3.11
uv run pytest -q
uv run python -m experiments.long_runs.20B_recurrence.run preflight
uv run python -m experiments.long_runs.20B_recurrence.run study --evaluate-checkpoints
```

`study` trains sequentially in transformer, hybrid, temporal, depth order and
is not the four-Pod launcher. For parallel execution, run one arm-specific
`train <arm>` command on each Pod; arms may start at different times from the
same frozen bundle.

`preflight` alone checks the host without writing anything. `preflight <arm>`,
which `train <arm>` also runs, appends this host's runtime receipt to
`<arm>_20B/results/environment.jsonl`. Each arm keeps its own log, so Pods never
write the same receipt file. An arm may move to a replacement host: the new
host is appended and marked `host_change`. The frozen protocol, GPU model,
torch/CUDA build, Python minor version, package versions and `uv.lock` must
match the arm's first host, or preflight refuses to continue. Completed arms are accepted only after their checkpoint configuration, dataset,
panel, and step match the frozen arm. `train <arm>` starts fresh only when no checkpoint exists; it refuses a
partial checkpoint. `train <arm> --resume` continues a valid checkpoint and
never starts fresh; the deterministic resume gate has passed and the flag is
set in `run.py`. `integrity <arm>` checks an arm's plan, environment log and
checkpoint against the frozen protocol (`--complete` also requires the final
step and matching evaluation reports). Checkpoint evaluation uses
`evaluate <arm> --step <step>` and always recomputes the requested report;
there is no report-cache reuse or cache-validation workflow.
`dry-run` does not freeze a protocol, write artifacts, or train. The transfer
bundle includes the data-preparation module and vocabulary required by the test
suite; fixture manifests mark Git provenance as unavailable when extracted
without `.git` metadata.


## Interpretation

This is a one-seed discovery run at matched data exposure and pass-count
**distributions**, not matched parameter counts or FLOPs. A hybrid advantage
supports this architecture under those controls; it does not isolate a
compute-independent benefit of combining axes. Keep the reported compute
estimates alongside NLL. Checkpoints along one training trajectory are not
independent replications, and placement variation is not training-seed
uncertainty. The axis and diagonal evaluation cells here each have a single
possible mask placement. Treat small differences as exploratory.

The core four-arm protocol, curriculum, WSD, and bounded 8/16-pass diagnostics
remain unchanged by the pre-launch review. No monitoring or restart automation
is installed by this runner.
