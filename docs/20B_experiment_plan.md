# 20B recurrence-axis study: experiment plan

> **Completed.** This is the pre-registered plan, kept as the reference the results are judged against. Results are in the [20B report](../experiments/long_runs/20B_recurrence/REPORT.md). In short, the hybrid tied temporal-only and beat depth-only, and neither gap grew with scale.

The 20B study is a **four-arm discovery run** of the default layout A: ordinary transformer, temporal-only, depth-only and hybrid. Each arm trains on 20B characters with a curriculum toward four-pass execution and a warmup–stable–decay (WSD) learning rate. Architecture and optimizer are held fixed at the serious profile. The question is whether the combined two-axis path gains an advantage over either axis alone as training scale grows.

The runner, configuration and launch procedure are in the [study README](../experiments/long_runs/20B_recurrence/README.md). This page records the protocol and the reasoning behind it. It was frozen on **2026-09-25**, before launch.

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
| Hardware | Four RTX 4090s, one arm per GPU (planned as Community; ran on Secure) |

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

## Launch decisions

Short probes on RTX 4090s fixed the remaining launch choices:
- **Batch:** a physical batch of `5 × 20`. `10 × 10` was only 3.6% faster, at 1.8× the memory.
- **Hardware:** one arm per GPU; sharing a GPU gave no useful gain.
- **Temporal gate:** initialised at `.10`. `.25` led by only about 0.001 in a 250M preflight.
- **Resume:** exact under deterministic kernels, over 1,000 updates.


## Out of scope: 8-pass training

This run does not add 8-pass (`U=7`) training support, even with small probability. If the 20B checkpoints show $L_1 > L_2 > L_4$ with four-pass gains still growing, that is a clean justification for adding `U=7` in the next experiment. If the four-pass gain saturates, the study has shown that more recurrence depth is not the immediate lever, without spending the extra compute.
