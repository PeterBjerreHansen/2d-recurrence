# Architecture A/B Verda Results

Historical record of the completed experiment. Paths below follow the current repository layout; recorded configurations, measurements, and raw artifact provenance retain their original meaning. See the experiment README for current commands.

Date: 2026-09-18  
Status: complete; compute released after local verification

## Outcome

The prescribed rule selects architecture A, `separated`. Architecture B, `coincident`, has slightly lower late `(3,3)` selection NLL, but its mean improvement is only `0.00174`, below the required `0.005` threshold. B is also about 1.96% slower in measured update work. No default architecture or training setting was changed.

The result is a one-seed matched comparison. It is evidence for this run and protocol, not a general claim that separated recurrence is universally better.

## Fixed protocol

Both runs used `chess_long_v1`, the full frozen dataset, the same seed `1337`, schedule seed `1729`, support `{0, 1, 3}`, recurrence probability matrix from the configs, 8 transformer blocks, width 512, 8 heads, context 1023, no bias, dropout 0, batch size 2, gradient accumulation 4, AdamW, learning rate `3e-4` with 100-step warmup and cosine decay to `3e-5` over 10,000 updates, weight decay `0.1`, betas `(0.9, 0.95)`, gradient clipping `1.0`, and the fixed 128-row selection panel. The selected final checkpoint was evaluated on the disjoint confirmation complement.

| Variant | Prelude | Core | Coda | Buffer | Source |
| --- | ---: | ---: | ---: | ---: | ---: |
| separated (A) | 1 | 4 | 1 | 1 | 1 |
| coincident (B) | 1 | 6 | 1 | 0 | 0 |

The source snapshot was commit `33b1e64de76c33afaa66352fb0162e58e671d399`, with the working-tree transfer bundle SHA-256 `2b0749f727d7eed04167df70e837f35a81d009838c56d8798a878e7de19b16f`. The dataset manifest SHA-256 is `a517b15768ea375e8e38d6408820cde4017e362f1c9b5f755f65b74a4ba6eee5`; the panel SHA-256 is `8860782ceee0acf48ea554174c59fda042e63d667ecc7cbdd7da840264f91cea`.

Each update processed 8,184 target characters. Each model processed 81,840,000 target characters; the two runs processed 163,680,000 in total.

## Environment and cloud execution

The runs used one Verda FIN-01 spot `1A6000.10V` with one NVIDIA RTX A6000, 48 GB VRAM, 10 CPU cores, and 60 GB RAM. The environment was Ubuntu `ubuntu-24.04-cuda-12.6`, Python 3.11.16, PyTorch `2.14.0+cu130`, CUDA 13.0, driver `580.126.09`, one visible CUDA device, float32 eager execution, trainer TF32 settings, and no compilation or autocasting. The complete environment receipt is in `experiments/ablations/architecture_sites/results/environment.txt`.

Two spot preemptions occurred during the coincident continuation. The retained OS volume and full optimizer/RNG/sampler checkpoints allowed exact resume; the completed milestones were copied off-VM after stopping and SHA-256 verified. The third allocation completed the final tail and confirmation evaluation. The benchmark measured:

| Variant | Mean seconds/update | Median seconds/update | 10k update-work seconds |
| --- | ---: | ---: | ---: |
| separated | 0.2138 | 0.2170 | 2,133.6 |
| coincident | 0.2183 | 0.2289 | 2,175.3 |

Benchmark estimates exclude setup, evaluation, checkpoint I/O, and spot replay. The measured training seconds above are also update work only; cloud wall time includes provisioning, replay, evaluations, transfers, and interruption gaps.

The orchestration wall time from first allocation creation at `2026-09-17T23:02:11.632Z` through verified VM release at `2026-09-18T01:39:15Z` was 9,423 seconds (2 h 37 m), including two spot interruptions, re-provisioning, evaluations, transfers, and waiting. Observed Verda account balance was `$10.75914` at allocation and `$9.94715` immediately before final VM release, an observed pre-cleanup spend of approximately `$0.81199`. The quoted active rate was `$0.3234/hour` including the 100 GiB OS volume. After all required artifacts were locally verified, both the VM and its 100 GiB volume were deleted. Verda confirms zero active instances and zero active volumes; the deleted volume is in the provider recovery trash window for 96 hours. The final provider receipt is in `experiments/ablations/architecture_sites/results/cloud.json`.

## Nine-cell selection learning curves

Values are panel NLL. Cell labels are `(u_t,u_d)`.

| Variant | Updates | 00 | 01 | 03 | 10 | 11 | 13 | 30 | 31 | 33 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| separated | 1,000 | 0.90250 | 0.88562 | 0.88921 | 0.82504 | 0.82352 | 0.82487 | 0.82040 | 0.81778 | 0.81861 |
| separated | 2,000 | 0.67857 | 0.64735 | 0.64964 | 0.63062 | 0.62324 | 0.62478 | 0.63168 | 0.62256 | 0.62438 |
| separated | 5,000 | 0.50457 | 0.48624 | 0.48684 | 0.48207 | 0.47587 | 0.47589 | 0.48202 | 0.47497 | 0.47568 |
| separated | 8,000 | 0.44461 | 0.42797 | 0.42798 | 0.42694 | 0.42035 | 0.42018 | 0.42697 | 0.41972 | 0.41992 |
| separated | 10,000 | 0.43084 | 0.41428 | 0.41433 | 0.41441 | 0.40725 | 0.40693 | 0.41441 | 0.40680 | 0.40661 |
| coincident | 1,000 | 0.90471 | 0.88616 | 0.89480 | 0.83143 | 0.82869 | 0.82996 | 0.82681 | 0.82276 | 0.82332 |
| coincident | 2,000 | 0.67664 | 0.64183 | 0.64336 | 0.62867 | 0.62084 | 0.62100 | 0.62854 | 0.61854 | 0.62006 |
| coincident | 5,000 | 0.50646 | 0.48343 | 0.48358 | 0.48392 | 0.47553 | 0.47523 | 0.48360 | 0.47461 | 0.47503 |
| coincident | 8,000 | 0.44773 | 0.42579 | 0.42522 | 0.42986 | 0.42011 | 0.41901 | 0.42940 | 0.41905 | 0.41858 |
| coincident | 10,000 | 0.43408 | 0.41135 | 0.41063 | 0.41606 | 0.40625 | 0.40488 | 0.41554 | 0.40511 | 0.40447 |

![Architecture A/B learning curves](results/architecture_ab_curves.png)

## Selection rule and late scores

| Variant | 8k `(3,3)` | 10k `(3,3)` | Mean late `(3,3)` | Mean weighted grid NLL | Depth gain 8k | Depth gain 10k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| separated | 0.419923 | 0.406605 | 0.413264 | 0.418317 | 0.007048 | 0.007806 |
| coincident | 0.418582 | 0.404467 | 0.411525 | 0.417882 | 0.010815 | 0.011075 |

B is no worse than A at either late `(3,3)` checkpoint and its weighted grid mean is only `0.000435` better, but its mean late gain is `0.001739`, not the required `0.005`. The predeclared rule therefore selects A. The `(3,0)` to `(3,3)` comparison shows that B has stronger depth refinement in absolute NLL, while A has the rule-selected architecture outcome.

## Confirmation evaluation

The selected separated 10,000-update checkpoint was evaluated on the 1,317-row confirmation complement:

| Cell | 00 | 01 | 03 | 10 | 11 | 13 | 30 | 31 | 33 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NLL | 0.42157 | 0.40547 | 0.40517 | 0.40624 | 0.39894 | 0.39852 | 0.40620 | 0.39864 | 0.39833 |

At `(3,3)`, confirmation accuracy was `0.852` and NLL was `0.39833`. Confirmation was used only to characterize the selected checkpoint, not to reverse the selection decision.

## Clipping and numerical health

Using unique optimizer updates, separated clipped 425/10,000 updates (`4.25%`), with minimum applied clip coefficient `0.0797` and maximum pre-clip gradient norm `12.547`. Coincident clipped 438/10,000 unique updates (`4.38%`), with minimum coefficient `0.0763` and maximum pre-clip norm `13.100`. The coincident raw metrics file has 10,173 training rows because 173 updates were replayed after spot recovery; the deduplicated summary above counts each optimizer update once. No non-finite failure occurred.

## Preserved local MPS reference

The existing `experiments/long_runs/baseline/results/lr3e-4` artifacts were not modified. They use MPS and the original recurrent shape (2 prelude, 4 core, 1 coda), so they are secondary and not an exact architecture match. They use the same selection panel and evaluator. At `(3,3)` selection NLL:

| Updates | Local MPS reference | CUDA separated | CUDA coincident |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.81554 | 0.81861 | 0.82332 |
| 2,000 | 0.62215 | 0.62438 | 0.62006 |
| 5,000 | 0.47743 | 0.47568 | 0.47503 |
| 8,000 | 0.42174 | 0.41992 | 0.41858 |
| 10,000 | 0.40710 | 0.40661 | 0.40447 |

The local MPS 10,000-update confirmation `(3,3)` NLL is `0.39849`, versus `0.39833` for selected CUDA separated. This small cross-backend difference is not treated as decisive evidence. The MPS run and its original report remain intact.

## Artifacts

The raw checkpoints, metrics, all selection grids, selected confirmation grid, benchmark, decision, protocol, environment receipt, transfer bundle, and cloud receipt are in [`experiments/ablations/architecture_sites/results`](results). The final decision is [`decision.json`](results/decision.json), and the plotted curves are [`architecture_ab_curves.png`](results/architecture_ab_curves.png).
