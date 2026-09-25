# 20B Pod scheduler and monitor

This tool starts the four 20B arms on Runpod Community RTX 4090s as healthy GPUs become available, and then looks after them. A small hourly agent runs one command and passes on alerts. All decisions are in code: the agent never decides anything itself.

| Module | Role |
| --- | --- |
| `policy.py` | Pure decisions and spend rules: `decide`, `acquire_blocker`, `must_stop_all`, `ledger_spend`. Unit-tested. |
| `pods.py` | Runpod API and CLI, the SSH-proxy shell (`Pod`), transfers, and the arm lifecycle on a Pod |
| `cli.py` | Configuration, local state, locking, and the `tick` loop |

This directory is excluded from the frozen source hash (`SOURCE_EXCLUDED_DIRS` in `run.py`), so it can be fixed mid-run without invalidating the protocol. Pods receive only the frozen bundle.

## What one tick does

`tick` takes a file lock and does nothing while `results/ops/PAUSED` exists. Otherwise it first checks the account, then handles each arm in priority order (hybrid, temporal, depth, transformer).

**Account checks, every tick:**

- **Spend** is the larger of two figures:
  - the local Pod ledger: GPU price plus a disk estimate, for every Pod the tool created;
  - Runpod's billed pod spend since the campaign started, which also counts Pods created by hand.
- **Stop before the cap:** if spend plus `stop_margin_hours` at the current account burn rate would reach `spend_cap_usd`, every open Pod is stopped (volumes kept) and each stop is confirmed. Any Pod that fails to stop is named in the alert.
- **Balance:** alert when the balance covers less than `min_balance_hours` at the current account burn rate. A pending arm starts only if the spend cap fits spend so far plus the projected remaining cost of running arms plus this arm, and the balance covers that total with `projected_budget_margin_fraction`. Arms not yet started don't block it. Each arm is priced at its own cloud's rate: `arm_cloud_types` (for example `{"hybrid": "SECURE"}`) overrides `cloud_type` per arm, `cloud_rates_usd_per_hr` gives the rates, and `arm_projected_hours` overrides `projected_hours_per_arm` (the transformer is budgeted at 24 h). Runpod stops Pods when the balance reaches zero.

**Per arm:**

| Arm phase | What happens |
| --- | --- |
| `pending` | Wait if Pod slots, the projected spend or the balance don't allow another arm. Otherwise create a Community 4090 Pod, and keep it only if `cuInit` succeeds, the GPU's enforced power limit is at least `min_power_fraction` (90%) of its default (hosts capped at 150–193 W of 450 W trained about 3× slower), and a 50 MB range from the exact locked Torch wheel downloads at `minimum_download_mb_per_second` or faster; faulted or slow hosts are deleted at once. Send the frozen bundle and check its SHA-256, extract it, `uv sync`, run the tests, run `preflight <arm>`, and start the job: `train <arm>` from scratch, then `evaluate` every retained checkpoint. If setup fails, the Pod is **stopped**, not deleted, and the arm alerts. |
| `running` | Check that the Pod is running and the GPU initializes. Run `run.py integrity <arm>` on the Pod: the Pod's protocol must hash to the local frozen `protocol.json`, and the arm's `plan.json`, `environment.jsonl` and `ckpt.pt` must match the frozen protocol. Check that disk use is below `disk_alert_fraction` and that the job is alive and advancing. |
| `running`, job stopped | Restart with `train <arm> --resume`, but only if `ckpt.pt` exists and is valid, `resume_enabled` is set, and the restart limit is not reached. A missing or invalid checkpoint always alerts; the tool never starts an arm fresh after its first launch. |
| `running`, job finished | Copy the arm's results home and check every file against the Pod's hash manifest. Then run `integrity <arm> --complete` locally against the frozen protocol: final step, valid checkpoint, and an evaluation report whose recorded hash matches each retained checkpoint. The results move to `<arm>_20B/results/` only if both checks pass. |
| `collected` | Delete the Pod if `allow_release` is true; otherwise alert that it is ready to release. |
| `alert` | Do nothing and repeat the alert until a human clears it. |

Alerts include the last lines of the arm's job log. Thresholds are in minutes, so they don't depend on how often ticks run:

- a Pod unreachable for `unreachable_minutes` alerts;
- a training step unchanged for `stall_minutes` alerts;
- a job with no logged step `startup_minutes` after it started alerts.

The step counter stays still during post-training evaluation, and that phase is recognized as healthy.

**Host CPUs.** Every rented Pod's CPU model is recorded in `events.jsonl` and `state.json` (`machine_cpus`), and each tick reports every running arm's CPU. The recurrent arms are limited by how fast the host CPU dispatches GPU work: on AMD EPYC 7532 hosts the hybrid ran about 1.6× slower than on the fast probe host. An hourly fast-host survey for migrations was tried and removed: it found nothing, and all arms moved to Secure Cloud.

**Notifications.** The tick opens macOS dialog windows itself, whatever the scheduling agent shows. Notification Center banners from `osascript` are silently dropped on this Mac. A dialog opens without blocking the tick and stays until you click OK. You get one:

- for each new alert (warning icon; a sound for `FAST HOST` and failures);
- for each arm that starts, resumes or is collected;
- once a day, at the first tick after `daily_summary_hour` (09:00), with a status summary.

A persisting alert opens a new dialog at most every `notify_repeat_hours` (6). `cli.py notify-test` opens a test dialog.

**Cadence.** Every tick reports `next_tick_minutes`: `fast_tick_minutes` (10) while any arm is still `pending`, otherwise `slow_tick_minutes` (60). The monitoring agent reschedules itself to match. A tick called before it is due returns `not_due` and does nothing, so a missed schedule switch is harmless. `tick --force` overrides this for a human. A tick that is acquiring or collecting can hold the lock for over an hour; later ticks then report the lock and do nothing.

## Setup (after the resume gate and the freeze)

1. Top up the Runpod balance. The locked 1,000-update RTX 4090 measurement projects about 46 GPU hours per arm. The monitor reserves 48 hours per arm and 10% margin: about $75.28 for four arms including the configured disk estimate, within the $80 spend cap. Stop other Pods before using this threshold; their charges use the same account balance.
2. Build the bundle: `uv run python -m experiments.long_runs.20B_recurrence.package --output ~/20B/20B_recurrence.tar.gz`.
3. Copy `ops/config.example.json` to `experiments/long_runs/20B_recurrence/results/ops/config.json`. Set `bundle`, `spend_cap_usd` and `resume_enabled: true`. The deterministic resume gate has passed and `EXACT_RESUME_GATE_PASSED` is set in `run.py`. `load_config` checks the bundle, its `.sha256` sidecar and the SSH key before any Pod is rented.
4. Dry run: `uv run python -m experiments.long_runs.20B_recurrence.ops.cli tick --dry-run`. It observes and decides, and acts on nothing.
5. Schedule the hourly agent below.

The machine running the ticks needs `runpodctl` with an API key, the Runpod SSH key, the bundle, and network access for `runpodctl send`/`receive`. Transfers run between this machine and the Pods.

## Hourly agent instructions

> Once an hour, from the repository root, run (the project's own Python, not `uv run`: sandboxed schedulers may be denied access to uv's cache):
>
> `.venv/bin/python -m experiments.long_runs.20B_recurrence.ops.cli tick`
>
> - If the output has no `ALERT` lines, stay silent.
> - If it has `ALERT` lines, send them to the user verbatim, with `spend_usd`, `spend_cap_usd` and `balance_usd` from the JSON.
> - If the command fails, or says another tick holds the lock, report that once and do nothing else.
> - Never run any other command. Do not create, start, stop or delete Pods, edit `config.json` or `state.json`, pause, unpause or clear alerts, or retry a failed tick in the same hour. A human does these.

## Commands for the human

```sh
uv run python -m experiments.long_runs.20B_recurrence.ops.cli status
uv run python -m experiments.long_runs.20B_recurrence.ops.cli tick --dry-run
uv run python -m experiments.long_runs.20B_recurrence.ops.cli pause
uv run python -m experiments.long_runs.20B_recurrence.ops.cli unpause
uv run python -m experiments.long_runs.20B_recurrence.ops.cli clear-alert hybrid --phase running
```

**Pause before any manual Pod work.** The tick lock only guards the tool's own commands; hand-run `runpodctl` commands can race a tick. `pause` waits for a running tick to finish, then makes later ticks do nothing until `unpause`.

`clear-alert` returns an arm to `pending`, `running` or `collected` after you have fixed the cause, for example after recovering a checkpoint onto a new Pod and updating `pod_id`/`pod_host` in `state.json`. Every action is logged to `results/ops/events.jsonl`.

## Limits of this version

- **Recovery from a lost GPU or Pod is manual:** pause, then follow the zero-GPU recovery procedure in the [20B plan](../../../../docs/20B_experiment_plan.md#checkpointing-and-recovery), then clear the alert.
- **No external checkpoint backup during training.** The Pod Volume Disk survives restarts, not host loss.
- **Billing lag:** Runpod's billing figures can trail real time. The local ledger covers the gap for Pods the tool created, but not for Pods created by hand.
- **Untested end to end:** the Runpod calls, SSH proxy and transfers have no end-to-end tests. Policy, spend rules, config validation, parsing, resume modes and the integrity check are unit-tested. Watch the first real acquire and collect by hand.
