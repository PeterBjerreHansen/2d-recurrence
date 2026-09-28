# Board-state probing: report

Linear probes for the board and the side to move at every residual-stream site of the four 20B arms, Karvonen's released 8-layer model and a random-init control, with causal checks. Setup and commands are in the [README](README.md).

## Summary

- **The pipeline reproduces Karvonen.** In his setting, his model's board probe peaks at L6 with 0.980 of squares correct (paper: 0.991 with about ten times more probe games). The random-init model scores 0.754 (paper: 0.750). His 99% applies only to each game's first 365 characters. Over whole 1,023-character rows the same kind of probe gets 0.87–0.91.
- **Temporal memory brings the board in early.** Right after the temporal mixer (`Tmix`, one block applied), temporal and hybrid decode the board at 0.85. The transformer needs five blocks to get there, and the depth model's first iteration tracks the transformer.
- **The memory carries the board already updated with the latest move,** and the model uses it to choose its move. The previous character's late layers apply the move. Swapping the memories of the last 64 characters for another game's moves both the board read at L6 and the next move toward that other game.
- **Temporal peaks slightly higher and earlier than our transformer:** 0.960 at L6 against 0.955 at L8. All four 20B arms stay below Karvonen's 3×-longer-trained transformer (0.979). Recurrence gives an earlier board, not a much better one.
- **The move decision is readable earlier in the temporal model too.** The memory carries information about the upcoming move from the previous character. Final readouts are equal across arms. In the looped arms, core iterations after the first (hybrid) or second (depth) add almost nothing to board or move readouts.
- **The probes are causal where the model reads them.**
  - Pushing the side-to-move direction at L2–L3 flips the model between predicting a move number and a move in 99–100% of held-out cases in the transformer and depth arms. A random direction of the same norm does nothing.
  - Deleting a piece from the probed board in the late layers makes the new move legal on the modified board 64–72% of the time across our arms, against 10–19% for random directions. Temporal gives 72%, our transformer 65%, Karvonen's model 78%.
  - Edits where the memory delivers the board have no effect, because earlier characters still carry the true board.
- **Single training seed, one probe split.** Differences of about 0.005 in probe accuracy should not be over-read.

![Board probe accuracy against transformer blocks applied](results/figures/probe_curves.png)

*Held-out probe accuracy at every site. Looped arms show every core iteration, so the x-axis is compute. `Tmix` is drawn at the same x as L1. Figure: `figures.py`.*

## Replicating Karvonen

Karvonen's probe code (`chess_llm_interpretability`) truncates each probe game to its first 365 characters and probes only `.` characters within them. Our first pass probed whole rows, which contain long games and later games in a row, and gave about 0.94 at best. `board_karvonen` applies his restriction.

| Model | L4 | L5 | L6 | L7 | L8 |
| --- | --- | --- | --- | --- | --- |
| Karvonen's model | 0.814 | 0.872 | **0.979** | 0.972 | 0.962 |
| Our transformer | 0.803 | 0.847 | 0.930 | 0.945 | **0.955** |
| Random init | 0.753 | 0.753 | 0.754 | 0.755 | 0.755 |

The per-square majority baseline is 0.667. The mine/theirs encoding is not the issue: at `.` characters White is always to move, so mine is white.

At white-to-move points, errors concentrate on the opponent's pieces. In an L7 probe of our transformer, White's pieces are recalled at 0.93–0.97 and Black's at 0.80–0.85, mostly black pawns predicted empty. An MLP probe at the same site reaches 0.961 over whole rows, against 0.940 for the linear probe. Part of the board is present but not linearly readable.

## Layer curves

Board accuracy in Karvonen's setting, and on changed squares over whole rows (squares whose class differs from the starting position):

| Arm | Early site | Karvonen setting at early site | Best (site) | Changed squares, best |
| --- | --- | --- | --- | --- |
| Transformer | L2 | 0.742 | 0.955 (L8) | 0.803 |
| Temporal | `Tmix` | **0.852** | **0.960 (L6)** | **0.825** |
| Depth | L2 | 0.756 | 0.946 (L8) | 0.810 |
| Hybrid | `Tmix` | **0.850** | 0.950 (L7) | 0.815 |
| Karvonen | L2 | 0.756 | 0.979 (L6) | 0.849 |

- **The early jump is the temporal read.** Temporal and hybrid gain 0.10 between L1 and `Tmix`, with no block applied in between.
- **Looping doesn't sharpen the board.** In depth and hybrid, board accuracy moves between 0.90 and 0.94 inside the loop and reaches its best only at L7 or L8. The hybrid's core stays about 0.02 below the depth core.
- **Side to move** is decoded perfectly from L2 in every trained arm (random init: 0.53). The two space kinds are the same input token, so this is computed, not read off the input.

## The memory carries the board

**Memory swap** (`swap.py`, 200 pairs of white-to-move dots at the same ply from different held-out games). For the last *k* characters of game A, the memory each character reads is replaced with game B's memory at the same offset. This is applied on every settling pass. On squares where A and B differ, the table shows how often the `board_karvonen` probe reads A and how often B. It also shows the next-character probability on first characters that begin a legal move only in A, or only in B.

| Temporal | `Tmix` A / B | L2 A / B | L6 A / B | L8 A / B | P(only-A first char) | P(only-B first char) |
| --- | --- | --- | --- | --- | --- | --- |
| No swap | 0.70 / 0.23 | 0.68 / 0.23 | 0.93 / 0.04 | 0.92 / 0.05 | 0.279 | 0.001 |
| k = 1 | 0.21 / 0.72 | 0.56 / 0.34 | 0.93 / 0.04 | 0.92 / 0.05 | 0.275 | 0.001 |
| k = 4 | 0.21 / 0.72 | 0.52 / 0.38 | 0.93 / 0.04 | 0.91 / 0.05 | 0.241 | 0.028 |
| k = 16 | 0.21 / 0.72 | 0.39 / 0.51 | 0.80 / 0.12 | 0.79 / 0.13 | 0.168 | 0.065 |
| k = 64 | 0.21 / 0.72 | 0.31 / 0.60 | 0.59 / 0.28 | 0.59 / 0.29 | 0.109 | 0.134 |

| Hybrid | `Tmix` A / B | L2 A / B | L6@4 A / B | L8 A / B | P(only-A first char) | P(only-B first char) |
| --- | --- | --- | --- | --- | --- | --- |
| No swap | 0.70 / 0.22 | 0.66 / 0.25 | 0.86 / 0.10 | 0.91 / 0.06 | 0.280 | 0.001 |
| k = 1 | 0.24 / 0.69 | 0.47 / 0.43 | 0.86 / 0.10 | 0.91 / 0.06 | 0.277 | 0.001 |
| k = 16 | 0.24 / 0.69 | 0.36 / 0.55 | 0.79 / 0.13 | 0.85 / 0.09 | 0.225 | 0.040 |
| k = 64 | 0.24 / 0.69 | 0.31 / 0.59 | 0.63 / 0.23 | 0.67 / 0.20 | 0.195 | 0.086 |

- **Controls:** self-swaps (B = A) reproduce the unswapped run for every *k*.
- **Narrow windows:** swapping one character's memory changes the board at `Tmix` but not the move. Attention reads A's board from neighbouring characters' memories.
- **Wide windows:** these push the board through to L8 and shift the move toward B. The temporal model decides its move partly from the board in its memory, not only from the text.
- **Hybrid vs temporal:** the hybrid depends on its memory less. Its core iterations partly restore A: 0.59 at L6@1, 0.63 at L6@4, 0.68 at L7 for *k* = 64.

**The update is already applied** (`update.py`). At pre-move points, probes for the board before and after the most recent move are scored on the squares that move changed:

| Site | Transformer after / before | Temporal after / before |
| --- | --- | --- |
| L1 | 0.40 / 0.47 | 0.39 / 0.47 |
| `Tmix` | — | **0.66** / 0.54 |
| L2 | 0.45 / 0.49 | 0.63 / 0.54 |
| L4 | 0.63 / 0.58 | 0.59 / 0.53 |
| L6 | 0.66 / 0.50 | 0.65 / 0.45 |

The transformer switches from the old board to the new one at L4. The temporal memory arrives already updated: the previous character's late layers applied the move. This is state *carried*, not state rebuilt at every character.

## Where the move decision forms

`decision.py`: at pre-move points, linear probes for the from-square of the move actually played, and the logit lens (final norm and unembedding) scored on the move's first character. The played move is a human's; the model's top-1 agreement with it is about 0.6.

| Arm | From-square: early | mid | late | Logit lens: early | mid | late |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | 0.106 (L2) | 0.368 (L6) | 0.429 (L8) | 0.000 (L2) | 0.354 (L6) | 0.714 (L8) |
| Temporal | **0.213 (`Tmix`)** | **0.400 (L6)** | 0.432 (L8) | **0.223 (`Tmix`)** | 0.454 (L6) | 0.721 (L8) |
| Depth | 0.110 (L2) | 0.409 (L6@4) | 0.428 (L8) | 0.000 (L2) | 0.509 (L6@4) | 0.724 (L8) |
| Hybrid | 0.179 (`Tmix`) | 0.415 (L6@4) | 0.428 (L8) | 0.088 (`Tmix`) | 0.506 (L6@4) | 0.725 (L8) |
| Karvonen | 0.108 (L2) | 0.389 (L6) | 0.416 (L8) | 0.000 (L2) | 0.379 (L6) | 0.726 (L8) |

- **Temporal's memory carries the upcoming move as well as the board.** The previous character's late layers already represent the move about to be written. This is a first handle on "considered moves".
- **Looped arms:**
  - Hybrid: most of the move is readable after the first core iteration (0.402 at L6@1).
  - Depth: needs the second iteration (0.349 at L6@1, 0.402 at L6@2).
  - Later iterations add almost nothing.
- **Final readouts are equal across arms.** Recurrence makes the decision available earlier in depth, not better at the output.

## Causal checks

**Side to move** (`steer.py turn`, 200 held-out spaces, ×2 the probe-margin scale). The table shows the fraction whose predicted next-character type (digit or letter) flips.

| Arm | Best sites (flip rate) | Late sites | Random direction |
| --- | --- | --- | --- |
| Transformer | L2 0.99, L3 1.00 | L6–L8 ≤ 0.03 | ≤ 0.08 |
| Temporal | L3 0.79, L2 0.51 | `Tmix` 0.00, L7–L8 0.00 | 0.00 |
| Depth | L3@1 0.99, L4@1 0.99, L2 0.96 | L6@4 0.50, L7 0.51 | ≤ 0.05 |
| Hybrid | L2 0.59 | L6@4 0.54, L7 0.51; L3@1 and L6@1 0.00 | ≤ 0.05 |

A side-to-move edit at `Tmix` has no effect in either temporal-memory arm, although the probe reads side to move perfectly there. The model recomputes it from the text at L2–L3. Late sites decode it but no longer use it; a flip rate near 0.50 means only one direction flips. The hybrid's L3@1 is unexplained: L2 edits work there but L3@1 edits don't, unlike the depth arm.

**Piece removal** (`steer.py board`, Karvonen's intervention, 150 held-out positions). Every position's original greedy move uses the deleted piece, so the unedited rate is 0.

| Arm | Edit | Legal on modified board | Legal on original | Random direction |
| --- | --- | --- | --- | --- |
| Transformer | L6, ×4 | 0.600 | 0.947 | 0.147 |
| Transformer | L4–L7, ×2 | 0.653 | 0.907 | 0.187 |
| Transformer | L4–L7, ×4 | 0.760 | 0.653 | 0.280 |
| Karvonen | L6, ×2 | 0.507 | 0.967 | 0.113 |
| Karvonen | L4–L7, ×2 | 0.780 | 0.813 | 0.193 |
| Temporal | L6, ×4 | 0.567 | 0.947 | 0.107 |
| Temporal | L4–L7, ×2 | **0.720** | 0.827 | 0.113 |
| Temporal | `Tmix` + L2, ×4 | 0.233 | 0.953 | 0.060 |
| Depth | L6@4, ×4 | 0.540 | 0.993 | 0.147 |
| Depth | L6@1, ×4 | 0.000 | 1.000 | 0.000 |
| Depth | L5 at all four iterations, ×4 | 0.640 | 0.907 | 0.147 |
| Depth | L4@4–L6@4 + L7, ×2 | 0.667 | 0.907 | 0.120 |
| Hybrid (100 positions) | `Tmix` + L2, ×4 | 0.020 | 1.000 | 0.000 |
| Hybrid | L5 at all four iterations, ×4 | 0.570 | 0.910 | 0.080 |
| Hybrid | L4@4–L6@4 + L7, ×2 | 0.640 | 0.850 | 0.100 |

Karvonen reports 0.904 for his model with edits at L4–L7, using five moves sampled at temperature 1 rather than one greedy move. Edits at L2–L4 or L8 alone have no effect in either transformer. Large edits cost legality on the original board, a side effect Karvonen also reports.

- **Every arm's board direction is causal in its late layers.** The best four-site window gives 0.64–0.72 on our 20B arms, against 0.10–0.19 for random directions. Temporal is highest (0.720), in line with its probe accuracy.
- **Edits where the memory delivers the board do almost nothing** (`Tmix` + L2: temporal 0.233 at ×4, hybrid 0.020). This matches the one-character memory swap: the edit starts at the probed character, and every earlier character still carries the true board, which attention reads.
- **In looped arms only late iterations matter.** Editing L6@1 in the depth arm has no effect, because later iterations overwrite it. Editing the last iteration or every iteration works.

## Limitations

- **Probe data:** probes are trained on 800 rows and tested on 200; the Karvonen setting uses 26.7k training points, about a tenth of his.
- **One seed per arm.** The temporal-over-transformer peak difference (0.005) is within what a second seed could reverse.
- **The temporal arm is the live-aligned export,** not the as-trained model, whose fixed point is broken (see the 20B report).
- **Move readouts use the human move actually played,** not the model's own choice.
- **Steering uses greedy decoding** and one edit rule. Scales are set per example from the probe margin, following Karvonen's dynamic scale.
