# 20B recurrence-axis study: experiment plan

The 20B study is a **four-arm discovery run** of the default layout A: ordinary transformer, temporal-only, depth-only and hybrid. Each arm trains on 20B characters with a curriculum toward four-pass execution and a warmup–stable–decay (WSD) learning rate. Architecture and optimizer are held fixed at the serious profile. The question is whether the combined two-axis path gains an advantage over either axis alone as training scale grows.

The runner, configuration and launch procedure are in the [study README](../experiments/long_runs/20B_recurrence/README.md). This page records the protocol and the reasoning behind it. It was last updated on **2026-09-24**, after the temporal-gate preflight and the short RTX 4090 microbatch, concurrency, WSD and resume probes. Those probes narrow the launch configuration; they are not evidence about the 20B result.

## Research question

> Can two-axis recurrence, when allowed to specialize to iterative computation, gain a training-scale-dependent advantage over either recurrence axis by itself?

A hybrid model beating the transformer would be useful but unsurprising: the recurrent model gets more computation. The comparison of interest is

$$
L_{\text{hybrid}(3,3)} < \min\left(L_{\text{temporal}(3,0)},\ L_{\text{depth}(0,3)}\right)
$$

at identical data exposure and a matched distribution of executed passes. The strongest version of the result would be both gaps, $L_D - L_H$ and $L_T - L_H$, **growing from 4B → 10B → 18B → 20B**.

Earlier evidence points weakly in this direction. In the completed 1B hybrid checkpoint, temporal-only and depth-only execution of that checkpoint each improved over ordinary execution, and `(3,3)` was better again. In the earlier 10k continuation, the `(3,3)` advantage over `(1,0)` grew between 5k and 10k updates. Both are comparisons within a single checkpoint, not between separately trained models.

**Interpretation limits.** The arms are matched in data and in the distribution of passes, not in parameters or FLOPs: temporal, depth and hybrid execute different source and mixer work. A hybrid win supports this two-axis architecture under matched exposure and pass distributions, not a compute-independent advantage, so compute estimates stay attached to every NLL comparison. There is one training seed per arm. Checkpoints along a trajectory are correlated observations, and mask-placement variation is not training-seed variation. The evaluated axis and diagonal cells each have only one possible mask placement, so the study can tell which direction the gaps move, not whether they exceed seed noise.

## Protocol

Freeze after the launch gates pass.

| Setting | Value |
| --- | --- |
| Horizon | **20B target characters per arm** (20,000,059,200 actual) |
| Updates | **195,504** |
| Arms | Transformer, temporal, depth, hybrid |
| Architecture | Layout **A** (1/1/4/1/1) |
| Width / layers / heads | 512 / 8 / 8 |
| Context | 1,023 characters |
| Effective batch | 100 |
| Physical batch | **5 × 20 accumulation** |
| Objective | **Final-only** |
| Optimizer | AdamW, betas `(0.9, 0.95)`, weight decay `0.1`, grad clip `1.0` |
| Peak LR | **3e-4** |
| Dropout | `0` |
| Precision | BF16 |
| Update support | `{0, 1, 3}` |
| Maximum execution depth | **4 passes** |
| Hybrid diagonal mass | **0.8** |
| LR schedule | **WSD** |
| Seeds and data order | matched across arms |
| Temporal gate initialization | **0.10** (current default) |
| Hardware | **Four Runpod Community RTX 4090s**, one arm per GPU |

Optimizer settings match the existing serious runs (AdamW `3e-4`, `.9/.95`, weight decay `.1`, clip 1, BF16, final-only supervision). This study introduces no optimizer changes.

### Pass curriculum

The update-count distribution shifts toward four passes during training:

| Characters | Optimizer steps (approx.) | `P(U=0)` | `P(U=1)` | `P(U=3)` |
| --- | ---: | ---: | ---: | ---: |
| 0–1B | 0–9,776 | .10 | .80 | .10 |
| 1–4B | 9,776–39,101 | .05 | .60 | .35 |
| 4–10B | 39,101–97,752 | 0 | .40 | .60 |
| 10–20B | 97,752–195,504 | 0 | .20 | .80 |

- **Early:** `U=0` gives a little scaffolding while the ordinary transformer computation forms.
- **By 4B:** `U=0` is gone.
- **Final half:** 80% four-pass and 20% two-pass. The `U=1` share keeps every recurrent application under pressure to be useful, while training concentrates on the four-pass endpoint.

Per arm:

- **Temporal** uses `(U_T, U_D) = (U, 0)`.
- **Depth** uses `(0, U)`.
- **Hybrid** keeps the same distribution over `max(U_T, U_D)` and puts 80% of each nonzero bucket on the diagonal, as in the 5B protocol. In the late phase most four-pass hybrid examples are therefore `(3,3)`, while a minority are asymmetric, so the model never sees only lockstep states.

Phase boundaries are absolute optimizer steps frozen in the configuration.

### Learning rate: warmup–stable–decay

| Phase | Steps | LR |
| --- | ---: | --- |
| Warmup | 0–2,000 | `0 → 3e-4`, linear |
| Stable | 2,000–175,954 | `3e-4`, constant |
| Decay | 175,954–195,504 | `3e-4 → 3e-5`, linear |

Step 175,954 is about 18B characters, so the last 2B characters are the cooldown. The stable phase is a continuing training trunk. A short final decay produces the finished checkpoint without tying the whole trajectory to a predetermined cosine horizon ([Wen et al., 2024](https://arxiv.org/abs/2410.05192)). Huginn's released large-run configuration similarly used a 4,096-step warmup and a trapezoidal schedule. Its configured horizon was longer than the run reached, so its reported training was effectively warmup plus stable.

The schedule also avoids increasing recurrence depth while the LR is falling. The last curriculum transition is at 10B characters, followed by another 8B characters at peak LR, so the model has a long period to specialize to deeper computation.

**Indexing.** Optimizer-step indices are zero-based:

- `lr_decay_start=175954` is the first cooldown update.
- `lr_decay_iters=195504` counts optimizer updates, so the final update (index 195,503) uses the minimum LR exactly.
- Checkpoint step 195,504 records the completed run.

The trainer accepts `lr_schedule='cosine'`, `'constant'` or `'wsd'`. If it is omitted, the historical `decay_lr` behavior applies: `decay_lr=False` means constant LR from step zero. This study selects WSD explicitly.

**Keep the 18B pre-decay checkpoint permanently.** If the 20B result argues for continuing, a longer stable branch should start from it rather than from the annealed 20B model.

## Evaluation

Training stops at four passes; evaluation goes further. At the major checkpoints, each arm is evaluated on the training graph at:

| Test depth | Temporal | Depth | Hybrid |
| --- | --- | --- | --- |
| 1 pass | `(0,0)` | `(0,0)` | `(0,0)` |
| 2 passes | `(1,0)` | `(0,1)` | `(1,1)` |
| 4 passes | `(3,0)` | `(0,3)` | **`(3,3)`** |
| 8 passes | `(7,0)` | `(0,7)` | `(7,7)` |
| 16 passes | `(15,0)` | `(0,15)` | `(15,15)` |

The **four-pass cells are the primary comparison.** Eight and sixteen passes are extrapolation diagnostics, not acceptance criteria. They are evaluation-only; training support stays `{0,1,3}`. Huginn uses variable recurrent depth as an inference-time compute axis ([Geiping et al., 2025](https://arxiv.org/abs/2502.05171)), and later work shows that deeper execution can help, saturate or degrade depending on the learned dynamics. The study measures this rather than assuming it.

### Live-feedback NLL

The training graph runs every pass over the whole sequence in parallel, so pass *b* reads temporal memory that is only *b−1* hops deep. Deployed temporal feedback is sequential: each token reads memory from the fully computed preceding token.

At the five major checkpoints, the runner therefore also records teacher-forced live NLL on the 128-row selection panel:

- temporal at one core pass per token;
- hybrid at one and at four;
- depth at four, which must equal its `(0,3)` cell.

Read training-graph and live results side by side. The live temporal model uses one core pass per token against four for depth and hybrid, so compute estimates stay attached.

### Checkpoints

- **Major checkpoints — 1B, 4B, 10B, 18B and 20B:** these mark the curriculum boundaries and the start of the LR decay. They get the full 1/2/4/8/16-pass table, live NLL, and per-pass activation RMS and finite-value stress checks on a fixed batch for the 8- and 16-pass schedules.
- **Curve checkpoints — 2B, 8B, 14B and 16B:** only the primary four-pass cell and the `(0,0)` reference.

The standard study command evaluates every retained nonzero checkpoint. Full-unroll gradient diagnostics are not run at 8 or 16 passes. A non-finite optional extrapolation cell or stress check is recorded as a diagnostic failure; it does not invalidate the primary metrics or stop later arms. A failure in a required primary measurement is fatal.

The key plot is four-pass NLL against characters for all four independently trained models.

## Launch evidence from the probes

### Microbatch

The RTX 4090 probes used the late hybrid update distribution, effective batch 100, and 200 updates per condition:

| Physical batch × accumulation | Measured chars/s | Peak allocated VRAM | Clipped updates |
| --- | ---: | ---: | ---: |
| `5 × 20` | 152,625 | 3.06 GiB | 61.5% |
| `10 × 10` | 158,176 | 5.61 GiB | 64.0% |
| `20 × 5` | 155,207 | 10.73 GiB | 59.5% |
| `25 × 4` | 147,349 | 13.28 GiB | 64.0% |

`10 × 10` was only 3.6% faster than `5 × 20`, used about 1.8× the VRAM, and halves the independent recurrence-schedule draws per optimizer update. These are throughput probes, not a learning-quality comparison. `5 × 20` is frozen: it keeps 20 schedule draws per update and leaves the most memory headroom.

The high clipping rates reflect the start of training; in the completed 5B runs, no update was clipped after step 40,000. Future exploratory throughput conditions should stay at about 200 updates each unless a longer sanity test is needed.

### Concurrency

A CUDA-MPS concurrency probe at `5 × 20` measured aggregate training-only throughput of 151.4k, 170.9k and 161.4k chars/s for one, two and four processes on one 4090. End-to-end rates, including startup and evaluation, were 107.9k, 105.4k and 114.9k chars/s. Two processes were slightly below the single-process rate end to end, and four gained only about 6.5%. The production run uses **one arm per GPU**, with no MPS multiplexing.

### Temporal-gate initialization

The 250M preflight completed both arms at 2,444 updates on one RTX 3090. At the final checkpoint, initialization `.25` was ahead of `.10` by only about 0.001 NLL in each of `(0,0)`, `(1,1)` and `(3,3)`. This is a one-seed, short-run difference, not a decision-grade win. The study keeps the existing **`.10`** and records `.25` as a sensitivity result.

### Resume

The 1,000-update WSD-plus-curriculum sanity run verified the LR formula and curriculum boundaries, but its resume comparison did not match exactly after restarting at update 500. Sampler and RNG state matched; model and optimizer state did not (maximum model difference 0.01365).

That run used ordinary, nondeterministic CUDA kernels. The 100-update comparison diverged the same way without deterministic kernels (maximum model difference 0.001) and matched bitwise with them. The mismatch is therefore most likely kernel nondeterminism amplified over 500 updates, not missing resume state. Bitwise equality with production kernels is not an achievable gate, because two uninterrupted runs would also differ.

**Deterministic resume gate passed (2026-09-25):** the 1,000-update WSD/curriculum comparison with `--deterministic` matched the model, optimizer, scaler, recurrence sampler, RNG state and every compared post-resume training metric exactly. The LR formula error was zero. It ran on one Secure RTX 4090 with Torch 2.14.0+cu130. To avoid exporting local dataset files after an automatic review block, the Pod generated synthetic 32-token rows with the same row size; this verifies exact resume mechanics under the locked CUDA stack, not data quality or full-dataset throughput. `EXACT_RESUME_GATE_PASSED` is enabled. Production runs then resume with ordinary kernels, where a resumed run is statistically, not bitwise, equivalent to an uninterrupted one. The machine-readable result is in `experiments/benchmarks/recurrent_runtime/results/20260925-4090-wsd-sanity-1000-deterministic-synthetic/sanity.json`.

## Execution on Community RTX 4090s

### Cost and runtime

On 2026-09-24 the deploy form showed a Community RTX 4090 at **$0.34/GPU-hour, On-Demand**, with no Spot or interruptible option on that offer. The form warns that Community Cloud is unpredictable, so recovery safeguards stay in place, but the budget must not assume a Spot discount. The price was for an unpinned region and is a snapshot, not a reservation. Recheck each offer in the chosen region before deployment. Do not silently substitute Secure pricing or another GPU if four suitable offers are unavailable.

On 2026-09-24, every available Community 4090 host that was rented for the resume probe had a faulted GPU: `cuInit` returned 999, and `nvidia-smi` reported "GPU Recovery Action: Reboot". Check that CUDA initializes on each Pod before transferring data.

The deterministic 1,000-update `5 × 20` RTX 4090 check measured 0.845 seconds per training update (1.18 updates/s). At that rate, a 195,504-update recurrent arm projects to about **46 hours of training work**. This excludes evaluations, checkpoint I/O, provisioning, transfers and recovery, and the synthetic rows may not capture full-dataset I/O. Use 48 hours per arm for initial planning. Four Community cards at `$0.34/hour` for 48 hours cost about **$65** for GPU time; the configured storage estimate adds about **$3**, and a 10% margin gives a projected campaign budget of **$75.28**. Set an explicit total spend cap before launch; the monitor must stop and alert before exceeding it.

### Storage

Four independent Community RTX 4090 Pods, one arm per GPU. Give each Pod a **30 GB Pod Volume Disk** mounted at `/workspace`, and keep that arm's checkout, dataset copy and results there. The dataset is about 7.9 GB, and comparable result directories are about 3.6–3.8 GB, so 30 GB leaves room for the environment, logs and checkpoint-write overhead.

Allocate a separate **30 GB container disk** for the temporary upstream archive, package caches and preparation scratch, and clear disposable staging data after preflight.

- **Persistence:** the container disk is temporary. The Pod Volume Disk is per-Pod and survives stop and restart.
- **Zero-GPU recovery:** Runpod's [zero-GPU recovery mode](https://docs.runpod.io/pods/troubleshooting/zero-gpus) can expose the volume when the GPU is unavailable. The volume stays tied to its host and is not portable to another Pod.
- **Network volumes:** Community Pods cannot attach [Runpod Network Volumes](https://docs.runpod.io/storage/network-volumes), which are Secure-Cloud-only.

At current published rates, a 30 GB Pod Volume Disk costs about `$3/month` per running Pod and `$6/month` while stopped: about `$12/month` running or `$24/month` stopped for four. A 30 GB container disk costs about `$3/month` per running Pod and is not retained when stopped. These costs are small for a ~37-hour run. Recheck rates at deployment and remove unneeded storage after a verified transfer ([Runpod pricing](https://www.runpod.io/pricing)).

### Checkpointing and recovery

The runner writes:

- the latest recovery checkpoint every **1,000 updates**, without evaluation;
- checkpoints at the evaluation interval of **10,000 updates** (about 1 hour 52 minutes at the measured rate);
- the named curve checkpoints.

Only the named checkpoints are kept as separate snapshots; recovery saves replace `ckpt.pt`. The final step is evaluated even though it does not fall on the interval, and the separate named-checkpoint study evaluations are unchanged. The deterministic resume gate above has passed, so `train <arm> --resume` is enabled; it never falls back to a fresh start. With hourly polling, a failure could take up to about an hour to detect, and recovery then replays up to one checkpoint interval of work.

On GPU interruption, keep the original Pod and volume. If needed, restart it with zero GPUs to retrieve the latest checkpoint, then transfer and verify it before terminating the Pod or moving the arm to a replacement. The Pod Volume Disk alone does **not** protect against host or storage loss or Pod termination. Verify the zero-GPU retrieval and checkpoint-transfer path before launch, and choose and test an external checkpoint backup if recovery must survive host or storage loss.

### Hourly monitoring

After the resume gate passes, a monitoring model runs an **hourly checkup**. For each arm it inspects:

- provider and Pod state;
- the runner heartbeat and latest step;
- the log tail;
- durable checkpoint age and hash;
- protocol, environment and panel hashes;
- disk health and budget.

While an arm is healthy the monitor stays quiet. If a Pod or process has stopped unexpectedly:

1. Acquire a per-arm lock and ensure no worker for that arm is alive.
2. **If the Pod still has its GPU:** validate the newest checkpoint and the frozen protocol, restart **only that arm**, and verify that the step advances.
3. **If the GPU is unavailable:** preserve the old Pod, use zero-GPU recovery to reach its volume, then transfer and verify the checkpoint and frozen artifacts on a replacement before resuming that arm.
4. **If the old volume cannot be accessed:** alert. Never start fresh.

Retries are bounded by the approved spend cap. Missing or corrupt checkpoints, hash mismatches, protocol drift, repeated restart failure or budget exhaustion stop automatic retries and raise an alert.

### Completion

On completion:

1. Copy each arm's full result directory and logs to the local repository.
2. Compare SHA-256 hashes for checkpoints and for the protocol, environment and panel receipts.
3. Only then stop and delete the Pod and any storage that is no longer needed.

Keep the original Pod until any needed checkpoint recovery or transfer has been verified. This avoids GPU charges after a successful transfer; the Pod Volume Disk is not a substitute for an external backup if the Community host is lost.

## Out of scope: 8-pass training

This run does not add 8-pass (`U=7`) training support, even with small probability. If the 20B checkpoints show $L_1 > L_2 > L_4$ with four-pass gains still growing, that is a clean justification for adding `U=7` in the next experiment. If the four-pass gain saturates, the study has shown that more recurrence depth is not the immediate lever, without spending the extra compute.
