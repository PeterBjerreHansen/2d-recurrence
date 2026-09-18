# Execute the A/B architecture comparison on Verda

Historical record of the completed experiment. Paths below follow the current repository layout; recorded configurations, measurements, and raw artifact provenance retain their original meaning. See the experiment README for current commands.

Use the current working tree at `/Users/peterbjerrehansen/Desktop/projects/coding_projects/active/2d_recurrence`. Implementations and executable experiment commands are ready. This task is to execute and report two matched runs on one Verda spot RTX A6000, sequentially. Preserve the local MPS run and all earlier artifacts. Do not use the old LR-candidate configurations: their A/B names describe learning rates, not these architectures.

## Fixed experiment

| Name | Config | Computation | Updates |
| --- | --- | --- | --- |
| A / separated | `experiments/ablations/architecture_sites/configs/separated.py` | L1 → temporal mix → L2 → depth mix → L3–L6 → depth candidate → L7 → temporal candidate → L8 | 10,000 |
| B / coincident | `experiments/ablations/architecture_sites/configs/coincident.py` | L1 → temporal mix → depth mix → L2–L7 → both candidates → L8 | 10,000 |

Every available state is read. Masks control writes; held states remain differentiable. In A, omit L7 on a nonfinal pass with no temporal write because its output is unused. In B, L7 belongs to the depth core and executes every pass. Distinct write masks can give B's two stored states different ages even though their candidate comes from the same latent. Final-only loss and full backpropagation are unchanged. Neither architecture receives update counts. Neither experiment changes the repository default.

Both use eight blocks, width 512, eight heads, context 1,023, zero dropout, AdamW at peak LR 3e-4/minimum 3e-5, warmup 100, cosine horizon 10,000, clip 1, batch size 2 with accumulation 4, model/data seed 1337, schedule seed 1729, and the existing distribution over update counts 0/1/3. One schedule is drawn per microbatch: do not replace accumulation with a larger batch. Each model sees 81.84 million target characters. Use CUDA float32 with the trainer's existing TF32 setting, eager execution, one GPU, and no autocast/compile changes. These settings favor comparability over maximum GPU throughput. Evaluation runs in float32 with its own default CUDA backend settings, identical for A and B; save those settings in the environment receipt. Do not claim bitwise equivalence with MPS.

Use the existing full `data/chess_long_v1` dataset (143,017 train rows, 1,445 validation rows) and unchanged `experiments/sweeps/baseline_lr_selection/results/panels.json` (128 selection rows, the remaining 1,317 confirmation rows). Selection evaluation covers all nine recurrence cells and all 13 distinct pilot mask placements. Do not regenerate a smaller dataset or resample the panel.

## Budget and VM preparation

The user requested this hardware but has not yet supplied a total dollar cap. Resolve that one missing constraint before incurring charges; reuse any cap subsequently provided in the conversation. It must include setup, both runs, evaluations, storage, and spot replay. Do not substitute a different GPU or on-demand billing. After the benchmark, project cost with evaluation/setup overhead and a reserve. If the allocation does not fit, retain checkpoints and ask for a choice; do not silently shorten training or switch precision. Use the provider's current price, not an assumed historical rate.

The local CLI is `/Users/peterbjerrehansen/.verda/bin/verda`. Use the existing authenticated account; do not print credential files or copy cloud credentials onto the VM. Check current resources first so a retry cannot create duplicate instances. The installed CLI supports the following discovery commands:

```sh
verda vm list -o json
verda instance-types --gpu --spot -o json
verda availability --type 1A6000.10V --spot -o json
verda images --type 1A6000.10V -o json
verda ssh-key list -o json
verda vm create --help
```

Select an available location and a compatible CUDA image. Resolve `AB_LOCATION`, `AB_IMAGE`, and `AB_SSH_KEY_ID` from actual account/catalog results. The instance type is `1A6000.10V` (48 GB GPU RAM); do not confuse it with RTX 6000 Ada. For a new VM, use the equivalent of:

```sh
verda vm create --kind gpu --instance-type 1A6000.10V \
  --location "$AB_LOCATION" --os "$AB_IMAGE" \
  --hostname recurrence-architecture-ab --ssh-key "$AB_SSH_KEY_ID" \
  --contract spot --is-spot --os-volume-size 100 \
  --os-volume-name recurrence-architecture-ab-os \
  --os-volume-on-spot-discontinue keep_detached --wait=false -o json
```

Record instance/volume IDs, creation time, quoted prices, budget, and location in `experiments/ablations/architecture_sites/results/cloud.json` locally and remotely. Verify returned spot status and volume retention policy before training. Keep the checkout and checkpoints on that retained OS volume. Spot retention is not a backup: copy completed milestones off the VM as described below. Use the installed CLI help and [Verda instance documentation](https://docs.verda.com/cli/instances/) and [storage documentation](https://docs.verda.com/cli/storage/) to resolve current lifecycle details. Poll provisioning with bounded waits. Do not repeatedly create instances if capacity is unavailable.

## Transfer the actual working tree

The experiment includes modified and untracked files, so cloning GitHub alone is insufficient. Run locally after inspecting the working tree:

```sh
uv run pytest -q
uv run python -m experiments.ablations.architecture_sites.package
```

This creates `experiments/ablations/architecture_sites/results/transfer.tar.gz` plus a SHA-256 sidecar. The archive contains source, tests, docs, lockfile, frozen data, and panel; it excludes credentials, environments, `.git`, and training outputs. It also includes `TRANSFER_MANIFEST.json` with every file hash and the original base commit. If a bundle already exists, inspect it or choose `--output` with a fresh name rather than overwriting it. Copy the archive and sidecar with scp/rsync using the verified VM IP and SSH key. Extract into a new directory on the retained volume, not over an existing experiment.

Verify the archive checksum before extraction, then verify every manifest entry with Python's hashlib. Initialize a local Git repository in this extracted snapshot because the existing trainer records Git provenance. A synthetic snapshot commit is acceptable: keep `TRANSFER_MANIFEST.json` so the original base commit and exact working tree remain explicit. Do not create or push a remote branch just to run this experiment. For example, after extraction:

```sh
sha256sum -c transfer.tar.gz.sha256  # Run alongside the transferred archive before extraction.
# After extraction, in the experiment directory:
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('TRANSFER_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('All transferred files verified')
PY
git init
git add .
git -c user.name='Experiment snapshot' -c user.email='snapshot@localhost' commit -m 'Freeze architecture A/B experiment snapshot'
```

Install/use uv and Python 3.11 on the VM, then `uv sync --frozen`. Confirm the installed PyTorch wheel supports CUDA and the driver. If the locked wheel cannot use the selected image, resolve this before freezing the protocol; record the exact wheel, dependency versions, driver, and any environment-only change. Do not silently upgrade dependencies during a run. Capture `nvidia-smi`, `uv pip freeze`, `torch.__version__`, `torch.version.cuda`, device name, and backend flags in `experiments/ablations/architecture_sites/results/environment.txt`. Run the tests on the VM, then:

```sh
uv run python -m experiments.ablations.architecture_sites.run preflight
```

Preflight requires exactly one RTX A6000, verifies dataset hashes and the panel, and creates `experiments/ablations/architecture_sites/results/protocol.json`. Later commands reject changes to the source/config hashes, data identity, panel, or recorded PyTorch/CUDA versions. Preserve this receipt across spot recovery. CUDA itself cannot be validated on the Mac; this VM preflight is required.

## Benchmark, then train

Run commands inside tmux or another existing persistent SSH session, with unbuffered Python output and logs retained. A persistent terminal does not survive spot removal; checkpoints do. Benchmark the first 100 real updates of each candidate, with the final 10,000-step LR schedule already configured:

```sh
uv run python -u -m experiments.ablations.architecture_sites.run train separated --until 100
uv run python -u -m experiments.ablations.architecture_sites.run train coincident --until 100
uv run python -m experiments.ablations.architecture_sites.run benchmark
```

Read `benchmark.json`. Its estimate uses updates 11–100 and excludes evaluation, checkpoint I/O, setup, and interrupted/replayed work. Reserve time and money for those costs; the final confirmation grid uses over ten times as many rows as a selection grid. Measure one selection evaluation at the first 1,000-step checkpoint to refine the estimate. The first 100 updates are part of the 10,000 total, not throwaway runs. Do not adjust learning settings based on these early losses.

After checking the budget, progress both models through the same milestones. The train command automatically resumes from `ckpt.pt` and interprets `--until` as an absolute update count. If already at or beyond a milestone it does no training. Execute this loop with shell error handling; on a failure, investigate before rerunning:

```sh
set -euo pipefail
for step in 1000 2000 5000 8000 10000; do
  for variant in separated coincident; do
    uv run python -u -m experiments.ablations.architecture_sites.run train "$variant" --until "$step"
    uv run python -u -m experiments.ablations.architecture_sites.run evaluate "$variant" --step "$step"
    # Copy and verify this completed milestone off-VM before proceeding.
  done
done
uv run python -m experiments.ablations.architecture_sites.run summarize
```

Do not leave that backup comment as the only implementation of backup. Either execute milestones individually and transfer after each, or insert an actual transfer command using the chosen destination. Evaluation reuses existing JSON only after checking checkpoint hash, panel/data hashes, step, split, mask seeds, and grid completeness. JSON and CSV are retained beside each checkpoint. Diagnostics remain enabled on one fixed batch. Non-finite training or trained-support evaluation results are failures to investigate, not permission to retune or silently restart.

The scripts make a practical predeclared choice: B must improve mean `(3,3)` NLL across 8k/10k by at least 0.005, be no worse at either checkpoint, and have mean training-distribution-weighted grid NLL no more than 0.005 worse. Otherwise prefer A. This tolerance is a decision rule, not statistical significance. Review the full curves, depth gains at fixed temporal counts, and measured cost. Do not describe a marginal one-seed result as an architectural law. If a run fails, report it; the summary expects two complete valid runs and does not manufacture a winner.

`summarize` writes `decision.json` and `curves.csv`. Only after that decision, evaluate the selected final checkpoint on all confirmation rows:

```sh
AB_SELECTED=$(uv run python -c 'import json; print(json.load(open("experiments/ablations/architecture_sites/results/decision.json"))["selected"])')
uv run python -u -m experiments.ablations.architecture_sites.run evaluate "$AB_SELECTED" --step 10000 --split confirmation
```

Do not use confirmation to reverse the choice and continue tuning. These are rows from the inherited split, not a claim of game-disjoint generalization. No additional seeds, third architecture, LR sweep, auxiliary loss, live-feedback implementation, or extra training is part of this allocation.

## Spot recovery, backup, and shutdown

The trainer atomically refreshes `ckpt.pt` every 100 updates and retains 0/100/1k/2k/5k/8k/10k snapshots. Checkpoints contain optimizer, scaler, random generators, sampler state, model configuration, data identity, and panel hash. Never resume from model weights alone. On interruption, reuse the retained OS volume in the same compatible environment, verify receipts, and rerun the same command. If restoring elsewhere, copy the full experiment receipt and output directories with the original relative paths. Keep one active trainer only; do not run concurrent writers in the same output directory.

Back up immutable retained checkpoints, evaluations, receipts, logs, and the source bundle to the Mac after each completed milestone. Copy the latest checkpoint only while training is stopped at a milestone, or verify its hash before and after transfer and retry if it changed. Do not back up partial `.tmp` files as checkpoints. Verify destination SHA-256 hashes. At minimum, keep final checkpoints, the decision, all raw grids, metrics, and protocol/environment receipts off-VM before cleanup. The runner deduplicates replayed update numbers in timing summaries; actual billed time still includes replay, evaluations, setup, and checkpoint I/O.

Track actual spend during execution. Before the cap is exhausted, stop at a durable checkpoint, back it up, and release compute. At normal completion also verify artifacts locally and release this experiment's compute through Verda. Verify provider-side billing/resource status; do not assume closing SSH or stopping Python stops VM charges. Preserve retained storage until backups are verified, then remove only resources created for this experiment when cleanup is authorized. Report any remaining storage charges/resources explicitly. Never touch unrelated account resources.

## Final report

Write `experiments/ablations/architecture_sites/REPORT.md` and return it with artifacts to the Mac. Include exact configurations, source/data/panel hashes, environment, update/character counts, total wall time and spend, interruptions, nine-cell learning curves at every milestone, late selection scores, confirmation results, clipping summaries, and recommendation. Plot NLL versus updates and measured training seconds using `curves.csv`; explain that summed update seconds exclude evaluation/I/O while cloud wall time includes them. Show `(3,0)` versus `(3,3)` to distinguish better absolute fitting from useful depth refinement.

Compare the completed local `experiments/sweeps/baseline_lr_selection/results/lr3e-4` run at matching checkpoints if its artifacts are available. Use the same selection panel, mask placements, and evaluator; label its MPS training backend and original architecture. Do not overwrite old reports, change its training, or use small cross-backend differences as decisive evidence. Treat this reference as secondary to the matched CUDA A/B comparison. State what remains uncertain; do not automatically change the default architecture or launch a longer run.
