# 20B recurrence-axis study: report

Four arms of the same 8-layer, width-512 character-level chess model, each trained on 20B characters: an ordinary transformer, a temporal-only model, a depth-only (looped) model, and the hybrid that trains both axes at once. The protocol is in the [experiment plan](../../../docs/20B_experiment_plan.md); the run setup is in this directory's [README](README.md).

## Summary

- **Pre-registered result: the hybrid ties temporal and beats depth, and neither gap grows.**
  - On 8,192 held-out confirmation rows, hybrid `(3,3)` and temporal `(3,0)` both score 0.2136. The difference is −0.00006, with a 95% interval of [−0.0002, +0.0001].
  - Depth `(0,3)` is 0.0024 behind.
  - All three recurrent models beat the transformer by 0.012–0.014 NLL.
  - The plan's strongest outcome, both gaps growing with scale, did not happen: from 4B to 20B both gaps shrink.
- **As trained, temporal fails in live execution.** Its token-by-token NLL is 0.48, twice the transformer's, although its training-graph NLL ties for best. About 95% of that penalty is one thing: live temporal loses track of **move numbers**. Its move choice suffers only mildly. The hybrid and depth models run live at their training-graph quality.
- **The failure is a train/live mismatch, and a tiny fine-tune removes it.** Retraining only the temporal mixer (about 6% of parameters) on settled memory, for 200 small updates (3.3M characters, 0.016% of the training data), brings temporal's live NLL to 0.2176, 0.0087 better than the transformer. Redoing the decay phase with warm-start batches also fixes it, with almost no training-graph cost (see [Alignment during the decay](#alignment-during-the-decay)).
- **Best deployable model: the hybrid with four core iterations per token** (live 0.2137). It beats depth, which applies the same number of transformer blocks per token, by 0.0012 and the transformer by 0.0127.
  - **At the transformer's block count** (one core iteration, 8 block applications per token), aligned temporal is best: 0.0034 ahead of the aligned hybrid and 0.0087 ahead of the transformer. The temporal mixer adds about 12% per-token latency on top of those blocks.
- **The gains are in move choice and grow over the game.** The hybrid's advantage over the transformer rises from 0.001 NLL in the first ten plies to 0.028 at plies 60–79.
- **Single seed.** The intervals cover which rows were evaluated, not training-seed variation.

![Four-pass training-graph NLL and live NLL against training characters](report_figures/nll_trajectories.png)

*Selection panel (128 rows), at every retained checkpoint. The confirmation results below use separate, larger row sets.*

## Setup

| | |
| --- | --- |
| Model | ChessGPT backbone: 8 layers, width 512, 8 heads, 1,023-character context; layout A (prelude 1, T-buffer 1, core 4, T-source 1, coda 1) |
| Data | 20,000,059,200 characters per arm (195,504 updates × batch 100), Lichess 6GB blocks, same data order for every arm |
| Optimiser | AdamW 3e-4, β = (0.9, 0.95), weight decay 0.1, clip 1.0, BF16; WSD schedule: warmup 2,000, constant to 175,954, linear decay to 3e-5 |
| Recurrence | update support {0, 1, 3}; curriculum toward four passes (80% four-pass in the second half); final-pass loss only |
| Arms | temporal `(U, 0)`, depth `(0, U)`, hybrid (80% diagonal), transformer |
| Hardware | one Secure RTX 4090 per arm on Runpod |

**What is matched:** data, data order, optimiser, and the distribution of executed passes. **What is not:** compute and parameters. Averaged over the curriculum, a recurrent training step applies about 19–22 transformer blocks per token against the transformer's 8, so the recurrent arms use roughly 2.4–2.7 times its training compute. At inference, one core iteration per token applies 8 transformer blocks, the same number as the transformer, and four iterations apply 20. The mixers add work on top of those blocks. Measured same-host live latency (Apple MPS, batch 1, same architecture):

| | ms per token | vs transformer |
| --- | --- | --- |
| Transformer | 2.35 | — |
| Temporal, 1 iteration | 2.62 | +12% |
| Hybrid, 1 iteration | 2.65 | +13% |
| Depth, 4 iterations | 5.81 | ×2.5 |
| Hybrid, 4 iterations | 5.83 | ×2.5 |

Comparisons below are therefore stated as **matched block applications**, not equal cost. The latencies come from the [5B dry-run cost benchmark](../../evaluation_battery/results/5B-dry-run/arm_cost_mps.json), whose checkpoints share this architecture.

**Evaluation data:** [evaluation battery](../../evaluation_battery/README.md) on the 20B panel's confirmation split.
- **Training graph:** 8,192 rows (8.4M targets).
- **Live execution:** the first 400 of those rows (409k targets, 74k move starts).
- **Intervals:** 95% row-bootstrap, 2,000 replicates.
- **Trajectories:** the selection-panel checkpoint reports produced during training.

## Pre-registered result

**Final checkpoint, confirmation rows** (training graph):

| | NLL | vs hybrid (3,3) |
| --- | --- | --- |
| Transformer | 0.22761 | +0.01405 [+0.01390, +0.01419] |
| Depth (0,3) | 0.21599 | +0.00242 [+0.00230, +0.00254] |
| Temporal (3,0) | 0.21363 | +0.00006 [−0.00005, +0.00018] |
| **Hybrid (3,3)** | **0.21357** | — |

**Trajectory, selection panel:**

| Checkpoint | Transformer | Temporal (3,0) | Depth (0,3) | Hybrid (3,3) | Temporal − hybrid | Depth − hybrid |
| --- | --- | --- | --- | --- | --- | --- |
| 1B | 0.3196 | 0.3049 | 0.3144 | 0.3058 | −0.0009 | +0.0086 |
| 4B | 0.2687 | 0.2507 | 0.2554 | 0.2497 | +0.0010 | +0.0057 |
| 10B | 0.2504 | 0.2329 | 0.2367 | 0.2323 | +0.0006 | +0.0044 |
| 18B (decay starts) | 0.2420 | 0.2253 | 0.2283 | 0.2253 | 0.0000 | +0.0030 |
| 20B | 0.2272 | 0.2122 | 0.2156 | 0.2125 | −0.0003 | +0.0031 |

- **Hybrid vs temporal:** within ±0.0015 at every checkpoint, ending in an exact tie.
- **Hybrid vs depth:** the hybrid leads throughout. The lead shrinks from 0.009 at 1B to 0.003 from 18B on.
- **The decay phase** improves every arm by 0.013–0.015 and leaves the ranking unchanged.
- **One hybrid does not replace the single-axis models at their own settings.** Run at `(3,0)`, the hybrid scores 0.0065 worse than the dedicated temporal model. At `(0,3)`, it's 0.0055 worse than depth. It matches the best model only at its own `(3,3)`.

## Live execution

The training graph runs all positions in parallel with a fixed number of passes. Live execution runs token by token, with real temporal feedback from every earlier token. Only live execution can generate text.

**Confirmation rows, as trained:**

| | Training graph (primary) | Live | Block applications per token |
| --- | --- | --- | --- |
| Transformer | 0.2264 | 0.2264 | 8 |
| Temporal, 1 iteration | 0.2126 | **0.4814** | 8 |
| Hybrid, 1 iteration | — | 0.2252 | 8 |
| Hybrid, 2 iterations | — | 0.2144 | 12 |
| Hybrid, 4 iterations | 0.2125 | 0.2137 | 20 |
| Depth, 4 iterations | 0.2148 | 0.2148 | 20 |

The live rows are a 400-row subset, so their training-graph values differ slightly from the 8,192-row table.

- **Depth** is exact by construction: with a KV cache per depth, live execution equals its `(0,3)` cell.
- **The hybrid** stays within 0.0012 of its training graph with four iterations. With two, it is already better than depth with four. With one iteration, the transformer's block count (about 13% more latency), it beats the transformer by 0.0012 [0.0003, 0.0020].
- **Temporal** matched its training graph early (live 0.2534 against 0.2507 at 4B on the selection panel). It then diverged: 0.33 at 10B and 18B, and 0.48 after the decay.

**What temporal gets wrong live:** almost all of the excess is on move-number characters. These are predictable from the move count, and they make up 25% of the targets.

| Live NLL by character class | Transformer | Temporal as trained | Temporal aligned |
| --- | --- | --- | --- |
| Move number | 0.0000 | **0.9708** | 0.0000 |
| First character of a move | 0.7582 | 0.8173 | 0.7370 |
| Rest of the move | 0.2267 | 0.2323 | 0.2138 |

As trained, live temporal puts 99.0% of its probability on legal moves, slightly above the transformer's 98.99%. The broken reader destroys its move counter, but not its sense of what is legal.

## Why temporal fails live

Scratch diagnostics on the selection panel (not part of the frozen protocol):

- **Training never shows the settled memory.** Every training trajectory starts from a memory-free pass and makes at most three temporal updates. Live execution reads the memory once it has settled, which is 20–28% away (relative L2).
- **The memory alternates between two states, and training saw only one.** Successive updates point in opposite directions, so odd and even passes land in different states. After step 39,101, the temporal arm trained only on 1 and 3 updates, and it learned to read only the odd state:

  | Temporal updates | 1 | 2 | 3 | 4 |
  | --- | --- | --- | --- | --- |
  | Temporal 4B | 0.253 | 0.267 | 0.251 | 0.267 |
  | **Temporal 20B** | **0.216** | **1.44** | **0.212** | **1.41** |
  | Hybrid 14B | 0.242 | 0.294 | 0.239 | 0.291 |

  The settled memory lies between the two states, and the 20B reader misreads it. A random push of the same size does no harm, so the failure is specific to that state, not general fragility.
- **The decay phase made the reader more selective.** The gap to the settled memory stayed the same, but the live penalty grew from +0.12 at 14B to +0.27 at 20B.
- **The hybrid resists it.** Its memory settles closer to the trained state, and its mixed temporal and depth schedules keep its reader broad.

## Post-hoc live alignment

**Procedure** ([`align.py`](../../ablations/live_warm_start/align.py), fixed before the confirmation evaluation):
- **Trainable:** only the temporal mixer; every transformer block stays frozen.
- **Each step:** 16 training rows (two microbatches of 8). The memory is settled with 64 gradient-free passes using the current mixer. Then one pass reads that memory with gradients, trained on the next-character loss.
- **Optimiser:** 200 updates, AdamW, learning rate 3e-4 with a 20-update warmup.
- **Hybrid:** the same procedure, alternating one and four core iterations across microbatches to cover both deployed settings.
- **Extra data:** 3,273,600 characters, 0.016% of the 20B budget.

**Confirmation results:**

| | As trained | Aligned | Change [95% CI] |
| --- | --- | --- | --- |
| Temporal, live | 0.4814 | 0.2176 | −0.2637 [−0.2670, −0.2603] |
| Temporal, training graph (3,0) | 0.2136 | 0.2174 | +0.0038 [+0.0037, +0.0039] |
| Hybrid, live, 1 iteration | 0.2252 | 0.2211 | −0.0041 [−0.0047, −0.0036] |
| Hybrid, live, 4 iterations | 0.2137 | 0.2138 | +0.0001 [−0.0001, +0.0003] |
| Hybrid, training graph (3,3) | 0.2136 | 0.2140 | +0.0005 [+0.0004, +0.0005] |

- **Temporal:** aligned, its live NLL is only 0.0012 above its own training graph, and the settled memory converges within about six passes. The cost is 0.0038 in the training graph, because the reader moved from the odd state to the settled one.
- **Hybrid:** alignment helps one-iteration execution and leaves four iterations unchanged.

The repair fixes the reader. A mixer trained on the original model's frozen settled memory reads that memory at 0.214 and runs live at 0.219. Training all parameters instead was no better.

### Alignment during the decay

The same idea can be applied during training instead of after it. [`aligned_decay.py`](../../ablations/live_warm_start/aligned_decay.py) resumes temporal from the retained pre-decay checkpoint (step 175,954) and redoes the 19,550 decay updates.

- **The change:** in 5 of every 20 microbatches, the memory is first settled without gradients (up to 64 passes, stopping at a tolerance; about 28 on average). The sampled schedule then trains from that memory instead of from none.
- **Everything else is the same:** data, schedules and learning rate match the original decay, so no characters are added.
- **Cost:** updates averaged 1.44 s, against 0.66 s in the original decay. The runs were on different hosts (EPYC 7282 against Ryzen 9 7950X), and these models are CPU-bound on EPYC hosts, so this is not a clean measure of the overhead. From the pass counts, the warm-start batches should add roughly 60% at equal hardware.

| Temporal, confirmation rows | Original decay | Post-hoc aligned | Aligned decay |
| --- | --- | --- | --- |
| Training graph (3,0) | 0.2136 | 0.2174 | 0.2143 |
| Training graph (7,0) | 0.2197 | 0.2185 | 0.2145 |
| Live, 1 iteration | 0.4814 | 0.2176 | 0.2142 |

On the live rows, the aligned decay is 0.0034 [0.0030, 0.0039] better than the post-hoc aligned model, and within 0.0008 of its own training graph. On the selection panel, its live NLL improves steadily during the decay (0.227 at 180k, 0.214 at 195.5k), where the original decay worsened.

Even update counts are still misread in the training graph (0.61–0.64 at 2 and 4 updates, against 1.40–1.43 before). Live execution does not use those states. This is one seed and one variant; the report's main comparisons use the post-hoc aligned model.

## Deployable comparison

Live execution on the confirmation rows, after alignment:

| Model | Block applications per token | Live NLL | vs transformer [95% CI] | Legal-move probability | Greedy move legal | Greedy move = move played |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | 8 | 0.2264 | — | 0.9899 | 0.9946 | 0.599 |
| Temporal, aligned | 8 | 0.2176 | −0.0087 [−0.0095, −0.0080] | 0.9929 | 0.9966 | 0.612 |
| Hybrid, aligned, 1 iteration | 8 | 0.2211 | −0.0053 [−0.0060, −0.0046] | 0.9901 | 0.9952 | 0.607 |
| Depth, 4 iterations | 20 | 0.2148 | −0.0115 [−0.0122, −0.0109] | 0.9950 | 0.9976 | 0.615 |
| **Hybrid, 4 iterations** | 20 | **0.2137** | **−0.0127 [−0.0135, −0.0120]** | **0.9953** | **0.9979** | **0.615** |
| *Temporal as trained* | 8 | *0.4814* | *+0.2550* | *0.9905* | *0.9952* | *0.583* |

- **At 20 block applications,** the hybrid beats depth by 0.0012 [0.0006, 0.0018]. The gap is small but consistent.
- **At 8,** aligned temporal beats the aligned hybrid by 0.0034 [0.0027, 0.0041]. At the transformer's block count, the dedicated temporal model is the better choice, once aligned. Both pay about 12–13% more latency than the transformer for their mixers.
- **The legal-move metrics follow the NLL ranking.** The two 20-application models put the most probability on legal moves and most often predict the move actually played.

## Where the gains come from

The battery splits each target by what it predicts: move numbers, the first character of a move (the move choice), the rest of the move, the check marker slot and delimiters.

| Training graph, confirmation rows | Move first | Move body | Check slot |
| --- | --- | --- | --- |
| Hybrid − transformer | −0.0349 | −0.0198 | −0.0014 |
| Hybrid − depth | −0.0058 | −0.0034 | −0.0003 |
| Hybrid − temporal | +0.0010 | −0.0006 | −0.0002 |

- **The gains are in move choice.** Almost everything comes from predicting which move is played. Check markers, which depend only on the board, barely move.
- **Against temporal,** the hybrid is slightly worse on the first character of the move and slightly better on the rest. The two cancel.
- **The advantage grows over the game.** Hybrid − transformer is −0.001 at plies 0–9, −0.012 at plies 20–39 and −0.028 at plies 60–79. Hybrid − depth follows the same pattern, from −0.0002 to −0.0047. Recurrence helps most where the game history is long.

## Reference: Karvonen's model

Karvonen's released 8-layer Lichess model uses the same architecture, the same data file and split, and three times the data (600,000 updates, 61.4B characters). Its own recorded best validation loss is 0.2166.

- **Checkpoint:** `lichess_8layers_ckpt_no_optimizer.pt` from [adamkarvonen/chess_llms](https://huggingface.co/adamkarvonen/chess_llms), W&B run `lichess_all_elos_8layers`, SHA-256 `84076e42e006c8edeabcb593e0b67469bdea17e1beabe78a459c2dd1f22c0668`.
- **Conversion:** [`karvonen_reference.py`](../../evaluation_battery/karvonen_reference.py) downloads the release, checks its SHA-256, strips the `torch.compile` prefix and adds our dataset metadata. The weights are unchanged. Its docstring gives the evaluation command.
- **Cross-check:** rescoring the released weights directly on the 400 live rows reproduces the reference's per-position losses exactly, and so does the converter's output.

It was evaluated on the same confirmation rows as the battery ([karvonen-reference](../../evaluation_battery/results/karvonen-reference)), so the differences below are paired. The model minus Karvonen, with 95% row-bootstrap intervals:

| | Training graph, 8,192 rows | Live, 400 rows |
| --- | --- | --- |
| Karvonen 8-layer | 0.2164 | 0.2155 |
| Transformer (ours) | +0.0112 [+0.0111, +0.0114] | +0.0109 [+0.0102, +0.0116] |
| Temporal (3,0), as trained | −0.0028 [−0.0029, −0.0026] | (live fails) |
| Temporal, aligned | +0.0010 [+0.0009, +0.0012] | +0.0021 [+0.0014, +0.0029] |
| Depth (0,3) / live 4 iterations | −0.0004 [−0.0006, −0.0003] | −0.0007 [−0.0013, −0.0001] |
| Hybrid (3,3) / live 4 iterations | −0.0028 [−0.0030, −0.0027] | −0.0018 [−0.0025, −0.0012] |

- **Beats Karvonen live:** on a third of his data, only the four-iteration models, hybrid and, narrowly, depth, at about 2.5 times the per-token latency.
- **Aligned temporal:** 0.0021 worse than Karvonen live.

## Limitations

- **One training seed per arm.** The intervals cover row sampling, not seed variation.
- **Matched in data and pass distribution, not in compute.**
- **The alignment was designed after the live failure was seen on the selection panel.** No confirmation row was used to design it. Some of those rows had been evaluated before, but only with other models.
- **Evaluated on subsets:** 8,192 of 82,675 confirmation rows for the training graph, and 400 for live execution.
- **Teacher-forced metrics only.** None of this measures playing strength.

## Implications for future runs

- **Measure deployed execution during training:** live NLL, NLL at long update chains, the gap between neighbouring update counts, and per-character-class NLL. Temporal's failure showed first and most clearly on move numbers.
- **Keep the update support gap-free and the mixture broad** throughout training. Deepen by moving the centre, never by concentrating the mass.
- **Add warm-start batches in the decay phase.** Tested once for temporal (see above): it fixed live execution without the post-hoc model's training-graph cost.

## Appendix: cost

- **Spend:** $134.73 for training, about $4.60 for the evaluation pod (alignment and battery), and about $7 for the aligned-decay run.
