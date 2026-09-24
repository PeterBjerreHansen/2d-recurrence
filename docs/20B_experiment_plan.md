If I were freezing the 20B experiment today, I would make it a **four-arm, four-pass-specialized discovery run**, with WSD and a curriculum that removes `U=0` early.

Your existing evidence is enough that I would stop changing architecture and optimizer. The 1B pair already shows useful recurrence, and the recurrent checkpoint shows the combined `(3,3)` path outperforming either axis alone.  The point of 20B is now to see whether that separation becomes substantial with training scale.

This plan was updated on **2026-09-24** with the completed temporal-gate preflight and the short RTX 4090 microbatch, concurrency, WSD, and resume probes. Those probes narrow the launch configuration; they are not evidence that a 20B result is already established.

## What the recent runs changed

The RTX 4090 probes used the late hybrid update distribution, effective batch 100, and 200 updates per throughput condition. They favor keeping the reference microbatch rather than trading away schedule draws for a small throughput gain:

| Physical batch × accumulation | Measured chars/s | Peak allocated VRAM | Clipped updates |
| ----------------------------- | ---------------: | ------------------: | --------------: |
| `5 × 20`                      | 152,625          | 3.06 GiB            | 61.5%           |
| `10 × 10`                     | 158,176          | 5.61 GiB            | 64.0%           |
| `20 × 5`                      | 155,207          | 10.73 GiB           | 59.5%           |
| `25 × 4`                      | 147,349          | 13.28 GiB           | 64.0%           |

`10 × 10` was only **3.6% faster** than `5 × 20`, used about **1.8× the VRAM**, and halves the independent recurrence-schedule draws per optimizer update. These short runs are throughput probes, not a learning-quality comparison. Freeze `5 × 20` for the 20B arms; it retains 20 schedule draws per update and leaves the most memory headroom. Keep future exploratory throughput conditions to the lower end—**200 updates each**—unless a specific longer sanity test is needed.

A separate CUDA-MPS concurrency probe at `5 × 20` measured aggregate training-only throughput of 151.4k, 170.9k, and 161.4k chars/s for one, two, and four processes on one 4090. End-to-end rates, including startup/evaluation, were 107.9k, 105.4k, and 114.9k chars/s. The 2-process end-to-end rate was slightly below the single-process reference; four processes gained only about 6.5%. This does not justify co-locating several 20B arms on one card. Run **one arm per GPU**, with no MPS multiplexing in the production run.

The 250M temporal-gate preflight completed both arms at 2,444 updates on one RTX 3090. At the final checkpoint, initialization `.25` was numerically ahead of `.10` by only about 0.001 NLL in each of `(0,0)`, `(1,1)`, and `(3,3)`. This is a one-seed, short-run difference, not a decision-grade win. Keep the existing **`.10` gate initialization** for 20B and record `.25` as a sensitivity result; do not tune the 20B gate from this small delta.

The 1,000-update WSD-plus-curriculum sanity run verified the LR formula and curriculum boundaries, but its resume comparison did not match exactly after restarting at update 500: sampler and RNG state matched, while model and optimizer state did not (maximum model difference 0.01365). That run used ordinary nondeterministic CUDA kernels. The 100-update comparison diverged the same way without deterministic kernels (maximum model difference 0.001) and matched bitwise with them, so the mismatch is most likely kernel nondeterminism amplified over 500 updates, not missing resume state. Bitwise equality with production kernels is not an achievable gate, because two uninterrupted runs would also differ. **Remaining launch gate:** repeat the 1,000-update interrupted WSD/curriculum comparison with `--deterministic` and require exact model, optimizer, sampler, RNG, and training-metric equality before enabling automatic resume. Production runs then resume with ordinary kernels; a resumed run is statistically, not bitwise, equivalent to an uninterrupted one.

## Proposed run protocol (freeze after launch gates pass)

| Setting                 | Recommendation                       |
| ----------------------- | ------------------------------------ |
| Horizon                 | **20B target characters / arm**      |
| Updates                 | **195,504**                          |
| Arms                    | Transformer, Temporal, Depth, Hybrid |
| Architecture            | Current **A**                        |
| Width / layers / heads  | 512 / 8 / 8                          |
| Context                 | 1,023 chars                          |
| Effective batch         | 100                                  |
| Physical batch          | **5 × 20 accumulation**              |
| Objective               | **Final-only**                       |
| Optimizer               | AdamW                                |
| Peak LR                 | **3e-4**                             |
| Betas                   | `(0.9, 0.95)`                        |
| Weight decay            | `0.1`                                |
| Grad clip               | `1.0`                                |
| Dropout                 | `0`                                  |
| Precision               | BF16                                 |
| Recurrence support      | `{0,1,3}`                            |
| Maximum execution depth | **4 passes**                         |
| Hybrid diagonal mass    | **0.8**                              |
| LR schedule             | **WSD**                              |
| Seeds/data ordering     | matched across arms                  |
| Temporal gate init      | **0.10** (retain current default)    |
| Hardware                | **Four Runpod Community RTX 4090s**, one arm per GPU |
| Runtime offer           | Community RTX 4090 showed **$0.34/GPU-hour, On-Demand** in the deploy form on 2026-09-24; confirm region, capacity, and rate again at launch |

I would keep those settings identical to the serious profile wherever possible. Your current serious runs already use AdamW `3e-4`, `.9/.95`, WD `.1`, clip 1, BF16 and final-only supervision; this isn't the time to introduce optimizer innovations.

The live deploy form labels this as **Community Cloud** and lists its instance pricing as **On-Demand**, with no Spot/interruptible option on the selected offer. The form warns that Community Cloud is unpredictable, so keep recovery safeguards, but do not describe this specific offer as Spot or budget on an assumed Spot eviction discount. The displayed `$0.34` was for an unpinned region and is a snapshot, not a reservation or price guarantee. Recheck each offer in the chosen region before deployment; do not silently substitute Secure pricing or another GPU if four suitable offers are unavailable.

At the measured recurrent rate (`~1.49 updates/s` for the 200-update `5 × 20` hybrid probe), a 195,504-update recurrent arm projects to about **36.5 hours of training work**, excluding evaluations, checkpoint I/O, provisioning, transfers, and recovery. Use **~37 hours** as a first planning estimate, not a service guarantee. Four cards at `$0.34/hour` for 37 hours would be about **$50** before disk/storage and overhead if all four remained billed for the full interval. Set an explicit total spend cap before launch; the monitor must stop and alert before exceeding it.

### Pass curriculum

This is the main change I'd make:

| Characters | Optimizer steps, approx. | `P(U=0)` | `P(U=1)` | `P(U=3)` |
| ---------- | -----------------------: | -------: | -------: | -------: |
| 0–1B       |                  0–9,776 |  **.10** |  **.80** |  **.10** |
| 1–4B       |             9,776–39,101 |  **.05** |  **.60** |  **.35** |
| 4–10B      |            39,101–97,752 |    **0** |  **.40** |  **.60** |
| 10–20B     |           97,752–195,504 |    **0** |  **.20** |  **.80** |

That is a stronger specialization curriculum than I initially suggested.

Early on, `U=0` provides a little scaffolding while representations and the ordinary transformer computation are forming. By 4B, it is gone completely. The final **half of training is 80% four-pass and 20% two-pass**.

I like that balance. It's enough `U=1` that every recurrent application is pressured to be useful, but the model is overwhelmingly optimizing for the four-pass endpoint you actually care about.

For temporal:

$$
(U_T,U_D)=(U,0).
$$

For depth:

$$
(U_T,U_D)=(0,U).
$$

For hybrid, preserve your existing construction: the table has the **same distribution over \(\max(U_T,U_D)\)** and 80% of each nonzero bucket goes on the diagonal. Your 5B protocol already gets this matching right.

That means, for example, in the late 80%-`U=3` phase, most hybrid high-depth examples are `(3,3)`, while a minority are asymmetric. I think that's desirable: specialize heavily toward the true hybrid while preventing the model from seeing only lockstep states.

## LR: WSD

I'd now use a canonical warmup–stable–decay rather than cosine:

| Phase  |           Steps |                   LR |
| ------ | --------------: | -------------------: |
| Warmup |         0–2,000 |    `0 → 3e-4` linear |
| Stable |   2,000–175,954 |  **`3e-4` constant** |
| Decay  | 175,954–195,504 | `3e-4 → 3e-5` linear |

Step 175,954 is about **18B characters**, so the last 2B characters are the cooldown.

This is exactly the use case WSD was designed for: the stable phase is a continuing training trunk, while a short final decay produces the finished checkpoint without tying the whole optimization trajectory to a predetermined cosine horizon. ([arXiv][1])

Implementation uses zero-based optimizer-step indices. `lr_decay_start=175954`
is the first cooldown update. `lr_decay_iters=195504` counts optimizer updates,
so the final update at index 195,503 uses the minimum LR exactly; checkpoint
step 195,504 records the completed run.

It also fits recurrent-model precedent reasonably well. Huginn's released large-run configuration has a 4,096-step warmup and a trapezoidal schedule with stable training plus a terminal cooldown; because its configured horizon was much longer than the run actually reached, the reported training effectively spent its time in warmup/stable operation.

Most importantly for **your** experiment, this avoids a nasty interaction:

$$
\text{increasing recurrence depth}
\quad+\quad
\text{simultaneously decreasing LR}.
$$

Your final transition to 80% four-pass occurs at 10B, while LR stays at `3e-4` for another **8B characters**. So the model gets a long period to genuinely specialize to deeper computation.

I would save the **18B pre-decay checkpoint permanently**. If the result at 20B screams “keep going,” that is the checkpoint from which I would continue a 40B stable branch rather than resuming the already-annealed 20B model.

The trainer now accepts `lr_schedule='cosine'`, `'constant'`, or `'wsd'`; an
omitted schedule preserves the historical `decay_lr` behavior. In particular,
`decay_lr=False` still means constant LR from step zero, while this 20B study
selects WSD explicitly.

## Evaluation is where I'd be more ambitious than training

Train at no more than four passes, but evaluate substantially beyond that.

At major checkpoints I would evaluate:

| Test depth | Temporal | Depth    | Hybrid      |
| ---------- | -------- | -------- | ----------- |
| 1 pass     | `(0,0)`  | `(0,0)`  | `(0,0)`     |
| 2 passes   | `(1,0)`  | `(0,1)`  | `(1,1)`     |
| 4 passes   | `(3,0)`  | `(0,3)`  | **`(3,3)`** |
| 8 passes   | `(7,0)`  | `(0,7)`  | `(7,7)`     |
| 16 passes  | `(15,0)` | `(0,15)` | `(15,15)`   |

The **primary comparison is four passes**. Eight and sixteen are extrapolation diagnostics, not acceptance criteria.

Huginn explicitly exploits variable recurrent depth as an inference-time compute axis, while subsequent work also shows that deeper execution can help, saturate, or degrade depending on the learned recurrent dynamics. ([OpenReview][2]) So I'd measure this rather than assume it.

Your code already supports arbitrary recurrence counts at the schedule level, and you've previously used eight-pass `U=7` stress schedules.

### Live-feedback NLL

The training graph runs every pass over the whole sequence in parallel, so pass *b* reads temporal memory only *b−1* hops deep. Deployed temporal feedback is sequential: each token consumes memory from the fully computed preceding token. At the five major checkpoints the runner therefore also records teacher-forced live NLL on the 128-row selection panel: temporal at one core pass per token, hybrid at one and four, and depth at four (which must equal its `(0,3)` cell). Read the training-graph and live results side by side. The live temporal model uses one core pass per token against four for depth and hybrid, so compute estimates stay attached.

### Checkpoints

I'd definitely retain/evaluate around:

**1B → 4B → 10B → 18B → 20B**

because those have semantic meaning: curriculum boundaries and LR-decay boundary.

I'd additionally keep perhaps 2B, 8B, 14B, and 16B for curves, but they don't need the complete expensive evaluation suite.

The really valuable plot will be the four-pass NLL versus characters for all four independently trained models.

The 20B runner evaluates the full 1/2/4/8/16-pass table at the 1B, 4B, 10B,
18B, and 20B checkpoints. The additional 2B, 8B, 14B, and 16B checkpoints
evaluate only the primary four-pass comparison and `(0,0)` reference. The
standard study command evaluates every retained nonzero checkpoint. Sixteen-
pass cells are evaluation-only; they do not expand training support beyond
`{0,1,3}`. These cells use the training graph; live-feedback NLL is recorded
separately (see above). At major checkpoints the runner also records per-pass activation
RMS and finite-value checks on a fixed batch for the 8- and 16-pass stress
schedules. It does not run full-unroll gradient diagnostics at those depths.
Non-finite optional extrapolation cells and stress checks are recorded as
diagnostic failures, but do not invalidate the primary four-pass metrics or
stop later study arms. A failure in a required primary measurement remains
fatal.

## Community 4090 execution and hourly recovery

The intended layout is four independent Community RTX 4090 Pods, one arm per
GPU. Give each Pod a **30 GB Pod Volume Disk** mounted at `/workspace`; keep
that arm's checkout, local dataset copy, and results there. The dataset is
about 7.9 GB and comparable current result directories are about 3.6–3.8 GB,
so 30 GB leaves room for the environment, logs, and checkpoint-write overhead.
Allocate a separate **30 GB container disk** for the temporary upstream
archive, package caches, and preparation scratch; clear disposable staging
data after preflight. Container Disk is temporary; the Pod Volume Disk is
per-Pod and survives stop/restart. Runpod's [zero-GPU recovery mode](https://docs.runpod.io/pods/troubleshooting/zero-gpus)
can expose that volume when the GPU is unavailable, but the volume remains
tied to its host and is not portable to another Pod. Community Pods cannot
attach [Runpod Network Volumes](https://docs.runpod.io/storage/network-volumes),
which are Secure-Cloud-only.

On GPU interruption, keep the original Pod and volume. If needed, restart it
with zero GPUs to retrieve the latest checkpoint, then transfer and verify it
before terminating the Pod or moving the arm to a replacement. The Pod Volume
Disk alone does **not** protect against host/storage loss or Pod termination;
verify the zero-GPU retrieval and checkpoint-transfer path before launch, and
choose/test external checkpoint backup if recovery must survive host/storage
loss. The runner now writes its latest recovery checkpoint every **1,000
updates** without evaluation, as well as at evaluation intervals (`10,000`
updates) and named curve checkpoints. Only the named checkpoints are retained
as separate snapshots; recovery saves replace `ckpt.pt`. Set in-training
evaluation to every **10,000 updates**; at the measured rate this is roughly
1 hour 52 minutes between periodic evaluations. The final step is evaluated
even though it does not land on that interval. The separate named-checkpoint
study evaluations remain unchanged. Automatic resume in the study runner is
disabled pending the production-settings exact-resume gate. With hourly
polling, recovery could wait up to about an hour to be detected and then replay
up to one checkpoint interval of work.

At current published rates, a 30 GB Pod Volume Disk is about `$3/month` per
running Pod and `$6/month` while stopped; four cost about `$12/month` running
or `$24/month` stopped. A 30 GB container disk is about `$3/month` per running
Pod and is not retained when stopped. These storage costs are small for a
~37-hour run, but recheck rates at deployment and remove unneeded storage after
verified transfer. ([Runpod pricing](https://www.runpod.io/pricing))

After the exact-resume gate above passes, use a monitoring model on an
**hourly checkup**. For each arm it should inspect provider/Pod state, the
runner heartbeat and latest step, log tail, durable checkpoint age/hash,
protocol/environment/panel hashes, disk health, and budget. While an arm is
healthy it should remain quiet. If a Pod or process has stopped unexpectedly,
it should acquire a per-arm lock and ensure no worker for that arm is alive.
If the Pod still has its GPU, validate the newest checkpoint and frozen
protocol, then restart **only that arm** and verify that the step advances. If
the GPU is unavailable, preserve the old Pod, use zero-GPU recovery to access
its volume, then transfer and verify the checkpoint and frozen artifacts on a
replacement before resuming that arm. If the old volume cannot be accessed,
alert; never start fresh. Retries are bounded by the approved spend cap;
missing/corrupt checkpoints, hash mismatches, protocol drift, repeated restart
failure, or budget exhaustion must stop automatic retries and alert.

On completion, copy each arm's full result directory and logs to the local
repository, compare SHA-256 hashes for checkpoints and
protocol/environment/panel receipts, and only then stop/delete its Pod and
any no-longer-needed storage. Retain the original Pod until any needed
checkpoint recovery/transfer has been verified. This avoids continued GPU
charges after successful transfer; the Pod Volume Disk is not a substitute for
an external backup if the Community host is lost.

## What result am I actually looking for?

The primary question isn't merely:

$$
L_\text{hybrid}<L_\text{transformer}.
$$

A recurrent model gets more computation, so that's useful but unsurprising.

The interesting result is:

$$
\boxed{
L_{\text{hybrid}(3,3)}
<
\min(
L_{\text{temporal}(3,0)},
L_{\text{depth}(0,3)}
)
}
$$

with identical data exposure and a matched distribution of executed passes.

This does **not** match parameter counts or FLOPs: the temporal, depth, and
hybrid arms execute different source and mixer work. A win supports this
particular two-axis architecture under matched exposure and pass distributions,
not a compute-independent advantage. Keep compute estimates with the NLL
comparison. This is one training seed per arm; checkpoints are correlated
observations, and mask-placement variation is not evidence of robustness across
training seeds. The evaluated axis and diagonal cells each have only one
possible mask placement.

Even better would be if

$$
L_D-L_H
\quad\text{and}\quad
L_T-L_H
$$

**increase from 4B → 10B → 18B → 20B.**

That would be the pattern I'd find genuinely exciting: both forms of recurrence work independently, but joint temporal/depth recurrence acquires an increasing advantage as pretraining proceeds.

Your existing evidence already points weakly in that direction: in the completed 1B recurrent checkpoint, temporal-only and depth-only execution each improved over ordinary execution, while `(3,3)` was better again.  And your older 10k continuation showed the `(3,3)` advantage over `(1,0)` getting larger between 5k and 10k.

## One thing I would *not* do

I would **not add 8-pass training support to this run**, even probabilistically.

If the 20B checkpoint gives something like:

$$
L_1 > L_2 > L_4
$$

and four-pass gains are still growing with scale, *then* you have a very clean justification for the next experiment to add `U=7`.

If the four-pass gain saturates, you've saved a lot of compute and learned that increasing recurrence depth isn't the immediate lever.

So the experiment stays conceptually tight:

> **Can two-axis recurrence, when actually allowed to specialize to iterative computation, produce a training-scale-dependent advantage over either recurrence axis by itself?**

I think **20B, maximum four passes, anneal `U=0` to zero, late training 80% `U=3`, final-only supervision, and WSD** is the best-shot version of that experiment with the evidence you currently have.

[1]: https://arxiv.org/abs/2410.05192?utm_source=chatgpt.com "Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective"
[2]: https://openreview.net/pdf?id=8ZiElzQxf1&utm_source=chatgpt.com "Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. 2025. Scaling up test-time compute with latent reasoning: A recurrent depth approach. *Preprint*, arXiv:2502.05171."
