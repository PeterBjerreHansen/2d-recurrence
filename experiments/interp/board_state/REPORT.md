# Board-state probing: report

What the four 20B arms represent about the board and the side to move, at which layer and at which character, and whether they use it. Setup, data and commands are in the [README](README.md).

## Summary

- **The probes are validated.** On Karvonen's released model, our pipeline gets 0.981 of squares right at L6 in his setting (paper: 0.991, with about ten times more probe games). A random-init model scores 0.760 (paper: 0.750).
- **Board tracking depends on the character.** Every model assembles the full board only at the two characters where a move is chosen: the `.` before White's move and the space before Black's. At other characters the board stays partial even in late layers. The two spaces are the same input token, yet only the one followed by a decision gets the full board.
- **The temporal memory delivers the board early, but what it holds depends on the character that wrote it.** The memory is sharpest on the latest move right after that move is written (0.95). It holds the fullest board just before a decision: the digit before the `.` writes older changes at 0.86 (hybrid 0.90), where the same layer of the arms without memory holds 0.69–0.72.
- **The temporal mixer passes older changes poorly at every character,** losing 0.10–0.17 in the temporal model and up to 0.22 in the hybrid. It passes the latest move well only while the move is fresh. At decision characters the layers after the mixer rebuild the board: temporal completes it by L5, the transformer and the hybrid by L7.
- **The temporal model uses the board in its memory to choose its move** (memory swap, tested at the `.` only).
- **Recurrence gives an earlier board, not a better one.** Peak board accuracy in Karvonen's setting: temporal 0.961, transformer 0.958, hybrid 0.953, depth 0.948, and Karvonen's 3×-longer-trained transformer 0.981.
- **Looped cores stop improving early.** Depth needs its second pass and the hybrid only its first; later passes add little to board or move readouts.
- **The probe directions are causal in the late layers at decision characters.** Early edits have little effect because the model re-derives what they change.
- **Scope:** the memory swap and the steering tests cover decision characters only. Each arm has one training seed.

## Reading the results: the move cycle

Rows are PGN text such as `;1.e4 e5 2.Nf3 Nc6 3.Bb5`. Each character plays a role in a repeating cycle, and the model's task at each one differs:

| Role | Example | Next character | A move is chosen next |
| --- | --- | --- | --- |
| Space after Black's move | ` ` before `3` | move-number digit | no |
| Move-number digit | `3` | digit or `.` | no |
| `.` | `.` after `3` | White's first move character | **yes** |
| First character of a move | `B` | rest of the move | partly chosen |
| Inside a move | `b` | rest of the move | no |
| Last character of a move | `5` | space, or `+`/`#` for a check | no |
| Space after White's move | ` ` before `a6` | Black's first move character | **yes** |

A move counts as applied at its last character, because predicting `+` requires the new position. The board is represented relative to the side to move (Karvonen's mine/theirs). Every probe therefore uses the relative encoding, or fits one probe per side to move.

The analyses cover different characters:

| Analysis | Characters |
| --- | --- |
| Karvonen replication; layer profiles (`board_karvonen`) | `.` of each row's first game, first 365 characters |
| Layer profiles (`board`, whole rows) | the two decision characters |
| Board across the cycle (`cycle.py`) | every character |
| Mixer, Karvonen's setting (`mixer.py`) | `.` |
| Latest-move timing (`update.py`); move readouts (`decision.py`) | the two decision characters |
| Memory swap (`swap.py`) | `.` |
| Steering (`steer.py`) | side to move: both spaces; piece removal: decision characters, and each edit continues through the move being written |

## 1. Validation against Karvonen

Karvonen's probe code (`chess_llm_interpretability`) truncates each probe game to its first 365 characters and probes only its `.` characters. `board_karvonen` applies the same restriction.

| All-square accuracy | L4 | L5 | L6 | L7 | L8 |
| --- | --- | --- | --- | --- | --- |
| Karvonen's model | 0.822 | 0.877 | **0.981** | 0.974 | 0.966 |
| Our transformer | 0.807 | 0.850 | 0.932 | 0.946 | **0.958** |
| Random init | 0.758 | 0.758 | 0.759 | 0.759 | 0.760 |

The per-square majority baseline is 0.667. Over whole rows, which include long games and later games in a row, the best probe of each arm reads 0.90–0.93. Side to move is decoded perfectly from L2 in every trained arm (random init: 0.54). This is computed rather than read off the input, since both spaces are the same token.

## 2. The board across the move cycle

`cycle.py` fits a board probe per character role (at least 4,000 optimizer steps each, 600 training rows, 100 test rows). Each cell reads **squares the latest move changed / squares changed earlier in the game**. Never-changed squares read 0.89–0.97 everywhere. The rows for Black's moves match White's to within 0.01, and check markers are omitted (about 600 test characters each).

**Temporal.** "Memory" is the previous character's L7, which is what the mixer reads.

| Role | Memory | Mixer | L2 | L5 | L7 |
| --- | --- | --- | --- | --- | --- |
| Space after Black's move | 0.95 / 0.77 | 0.92 / 0.61 | 0.85 / 0.66 | 0.80 / 0.67 | 0.77 / 0.67 |
| Move-number digit | 0.62 / 0.65 | 0.45 / 0.55 | 0.52 / 0.59 | 0.66 / 0.73 | 0.68 / 0.75 |
| **`.`** | 0.86 / 0.86 | 0.59 / 0.70 | 0.69 / 0.70 | **0.87 / 0.89** | 0.86 / 0.90 |
| First character of White's move | 0.86 / 0.90 | 0.64 / 0.76 | 0.68 / 0.74 | 0.81 / 0.86 | 0.80 / 0.84 |
| Inside White's move | 0.79 / 0.82 | 0.56 / 0.65 | 0.61 / 0.66 | 0.74 / 0.79 | 0.73 / 0.77 |
| Last character of White's move | 0.81 / 0.77 | 0.85 / 0.61 | 0.84 / 0.64 | 0.98 / 0.76 | 0.97 / 0.77 |
| **Space after White's move** | 0.95 / 0.76 | 0.91 / 0.60 | 0.82 / 0.67 | **0.87 / 0.89** | 0.86 / 0.90 |

**Transformer.** "Previous L7" is the same readout as the temporal model's memory column, but the transformer has no memory connection to read it through.

| Role | Previous L7 | L2 | L5 | L7 |
| --- | --- | --- | --- | --- |
| Space after Black's move | 0.96 / 0.80 | 0.46 / 0.55 | 0.64 / 0.60 | 0.68 / 0.73 |
| Move-number digit | 0.56 / 0.69 | 0.46 / 0.56 | 0.47 / 0.59 | 0.50 / 0.67 |
| **`.`** | 0.53 / 0.69 | 0.46 / 0.55 | **0.83 / 0.70** | 0.85 / 0.88 |
| First character of White's move | 0.85 / 0.88 | 0.46 / 0.55 | 0.65 / 0.69 | 0.79 / 0.85 |
| Inside White's move | 0.80 / 0.84 | 0.44 / 0.55 | 0.59 / 0.65 | 0.75 / 0.80 |
| Last character of White's move | 0.82 / 0.79 | 0.56 / 0.55 | 0.98 / 0.63 | 0.98 / 0.79 |
| **Space after White's move** | 0.96 / 0.79 | 0.49 / 0.55 | **0.85 / 0.70** | 0.85 / 0.88 |

**All four arms at the `.`.** For the looped arms, L5 is read in the last pass.

| Arm | Memory or previous L7 | Mixer | L2 | L5 | L7 |
| --- | --- | --- | --- | --- | --- |
| Transformer | 0.53 / 0.69 | — | 0.46 / 0.55 | 0.83 / 0.70 | 0.85 / 0.88 |
| Temporal | 0.86 / 0.86 | 0.59 / 0.70 | 0.69 / 0.70 | **0.87 / 0.89** | 0.86 / 0.90 |
| Depth | 0.57 / 0.72 | — | 0.47 / 0.55 | **0.86 / 0.87** | 0.83 / 0.89 |
| Hybrid | 0.86 / 0.90 | 0.60 / 0.69 | 0.67 / 0.68 | 0.80 / 0.84 | 0.85 / 0.90 |

**Older changes at L7 at the two spaces**, which are the same input token:

| Arm | Space before a move number | Space before Black chooses |
| --- | --- | --- |
| Transformer | 0.73 | 0.88 |
| Temporal | 0.67 | 0.90 |
| Depth | 0.73 | 0.89 |
| Hybrid | 0.75 | 0.90 |

- **The full board is built only when a decision follows.**
  - At the two decision characters, older changes reach 0.88–0.90 at L7 in every arm.
  - At a move's first character they are still 0.83–0.85 at L7; at the other characters they stay at 0.67–0.84.
  - The two spaces are the same input token, yet in every arm only the one followed by a decision gets the full board.
- **Temporal completes the board at L5 at decision characters; the transformer and hybrid only at L7.** At the `.`, older changes read 0.89 at temporal's L5 against 0.70 for the transformer and 0.84 for the hybrid's last pass. Depth reaches 0.87 at L5, but only in its fourth pass.
- **Every arm applies a move at its last character by L5** (0.98–0.99 on the moved squares). The temporal model also holds older changes better there than the transformer: 0.76 against 0.63.
- **What the memory holds depends on the character that wrote it:**
  - Right after a move, it is sharp on that move (0.95) and loose on older changes (0.76–0.77).
  - Arriving at the `.`, it reads 0.86 / 0.86 (hybrid 0.86 / 0.90). That memory is the L7 of the digit before the `.`. In the two arms without memory the same readout is 0.53 / 0.69 (transformer) and 0.57 / 0.72 (depth). The memory arms load the board into their memory one character before the decision.
  - Arriving at a move-number digit, it reads 0.62 / 0.65.
- **The temporal model's early advantage holds at every character, but its size varies.** At L2 temporal reads 0.52–0.85 / 0.59–0.74 across roles, the transformer 0.44–0.57 / 0.55–0.63.

## 3. The temporal memory and the mixer

The mixer blends the memory with the current character's embedding, using per-dimension gates. What it passes depends on the character (section 2):

- **Older changes lose 0.10–0.17 at every role in the temporal table** (0.04–0.07 at check markers). The hybrid's memory holds more but loses more: 0.90 → 0.69 at the `.`, and 0.84 → 0.64 at the space before Black chooses.
- **The latest move survives only while it is fresh.** It drops 0.03–0.04 at the spaces right after a move is written, but 0.22–0.27 at the `.`, the move's first character and inside the move.
- **The gates barely vary by role.** For temporal, memory gate means are 0.11–0.20 (highest inside moves) and current-character gate means 0.10–0.15; for the hybrid, 0.05–0.11 and 0.11–0.20. The selectivity is per dimension, not per character.

Cross-check at the `.` in Karvonen's setting (`mixer.py`):

| Readout | Memory | Mixer | L2 |
| --- | --- | --- | --- |
| Temporal: all squares / changed squares | 0.939 / 0.893 | 0.873 / 0.759 | 0.846 / 0.722 |
| Hybrid: all squares / changed squares | 0.959 / 0.926 | 0.867 / 0.754 | 0.839 / 0.708 |

**Sizes of the mixer's inputs,** over all positions of 40 rows:

- **Memory term against current-character term** (each gate × its projected input): the memory term is a median 0.64× the current-character term for temporal (10th–90th percentile 0.32–0.98), and 0.50× for the hybrid (0.25–0.74).
- **Trained gates:** mean gates are 0.16 (memory) and 0.12 (current character) for temporal, and 0.08 and 0.15 for the hybrid. They start at 0.10 and 0.90.

## 4. Layer profiles at decision characters

![Board probe accuracy against transformer blocks applied](report_figures/probe_curves.png)

*Left: Karvonen's setting (`.` only). Right: changed squares at both decision characters over whole rows. Looped arms show every core pass, so the x-axis is compute. The mixer output is drawn at the same x as L1.*

| Arm | Early site | Karvonen setting at early site | Best, Karvonen setting (site) | Best, changed squares (site) |
| --- | --- | --- | --- | --- |
| Transformer | L2 | 0.762 | 0.958 (L8) | 0.826 (L7) |
| Temporal | mixer | **0.873** | **0.961 (L6)** | **0.852 (L6)** |
| Depth | L2 | 0.760 | 0.948 (L8) | 0.835 (L6, pass 4) |
| Hybrid | mixer | **0.867** | 0.953 (L7) | 0.844 (L7) |
| Karvonen | L2 | 0.760 | 0.981 (L6) | 0.872 (L6) |

The transformer reaches 0.850 only at L5. The temporal-over-transformer peak difference in Karvonen's setting (0.003) is noise-level. On changed squares the difference is 0.026.

## 5. When the latest move is applied

`update.py` fits two probes per side to move: one for the board after the latest move, one for the board before it. Each is scored on the squares that move changed, shown as after / before:

| Site | Transformer at `.` | Transformer after White's move | Temporal at `.` | Temporal after White's move |
| --- | --- | --- | --- | --- |
| L1 | 0.40 / 0.47 | 0.41 / 0.47 | 0.39 / 0.47 | 0.40 / 0.47 |
| Mixer | — | — | 0.57 / 0.44 | **0.91** / 0.66 |
| L2 | 0.47 / 0.51 | 0.49 / 0.52 | 0.68 / 0.56 | 0.82 / 0.64 |
| L3 | 0.69 / 0.59 | 0.69 / 0.59 | 0.75 / 0.58 | 0.80 / 0.63 |
| L4 | 0.83 / 0.66 | 0.87 / 0.67 | 0.74 / 0.57 | 0.78 / 0.61 |
| L6 | 0.87 / 0.55 | 0.88 / 0.57 | 0.89 / 0.44 | 0.89 / 0.48 |

- **The transformer re-applies the latest move at L3–L4 at each decision character.**
- **The temporal memory arrives with the move already applied.** The effect is clear one character after the move (0.91 vs 0.66) and weaker at the `.`, three characters later (0.57 vs 0.44).

## 6. Does the model use its memory?

`swap.py`, 200 pairs of `.` characters at the same ply from different held-out games. For the last *k* characters of game A, the memory each character reads is replaced with game B's memory at the same offset, on every settling pass.

- **Board columns:** on squares where A and B differ, how often the probe reads A's piece and how often B's.
- **Next-move columns:** next-character probability on first characters that begin a legal move only in A, or only in B.

| Temporal | Mixer A / B | L2 A / B | L6 A / B | L8 A / B | Next move only in A | Next move only in B |
| --- | --- | --- | --- | --- | --- | --- |
| No swap | 0.75 / 0.18 | 0.70 / 0.22 | 0.93 / 0.04 | 0.93 / 0.05 | 0.279 | 0.001 |
| k = 1 | 0.18 / 0.75 | 0.55 / 0.35 | 0.93 / 0.04 | 0.93 / 0.05 | 0.275 | 0.001 |
| k = 4 | 0.18 / 0.75 | 0.51 / 0.39 | 0.93 / 0.04 | 0.91 / 0.05 | 0.241 | 0.028 |
| k = 16 | 0.18 / 0.75 | 0.38 / 0.52 | 0.80 / 0.12 | 0.79 / 0.13 | 0.168 | 0.065 |
| k = 64 | 0.18 / 0.75 | 0.30 / 0.61 | 0.59 / 0.28 | 0.59 / 0.29 | 0.109 | 0.134 |

| Hybrid | Mixer A / B | L2 A / B | L6, pass 4, A / B | L8 A / B | Next move only in A | Next move only in B |
| --- | --- | --- | --- | --- | --- | --- |
| No swap | 0.74 / 0.18 | 0.68 / 0.23 | 0.86 / 0.10 | 0.91 / 0.06 | 0.280 | 0.001 |
| k = 1 | 0.20 / 0.73 | 0.45 / 0.45 | 0.86 / 0.10 | 0.91 / 0.06 | 0.277 | 0.001 |
| k = 16 | 0.20 / 0.73 | 0.34 / 0.57 | 0.79 / 0.13 | 0.85 / 0.09 | 0.225 | 0.040 |
| k = 64 | 0.20 / 0.73 | 0.29 / 0.62 | 0.64 / 0.23 | 0.67 / 0.21 | 0.195 | 0.086 |

- **Control:** self-swaps (B = A) match the unswapped run to within 0.0005 at every site and window.
- **Narrow windows:** swapping one character's memory changes the board at the mixer but not the move. Attention reads A's board from neighbouring characters' memories.
- **Wide windows:** these carry B's board through to L8 and shift the move toward B.
- **The hybrid relies on its memory less.** For *k* = 64 its passes partly restore A: 0.59 after pass 1, 0.64 after pass 4, 0.68 at L7.

## 7. Where the move decision forms

`decision.py`, at the two decision characters: probes for the from-square of the move actually played (one per side to move), and the logit lens (final norm and unembedding) scored on the move's first character. The target is a human's move, so the ceiling is how often the model agrees with it.

| Arm | From-square: early | middle | L8 | Logit lens: early | middle | L8 |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | 0.114 (L2) | 0.427 (L6) | 0.492 | 0.000 (L2) | 0.354 (L6) | 0.714 |
| Temporal | **0.260 (mixer)** | 0.464 (L6) | 0.497 | **0.223 (mixer)** | 0.454 (L6) | 0.721 |
| Depth | 0.118 (L2) | 0.479 (L6, pass 4) | 0.499 | 0.000 (L2) | 0.509 (L6, pass 4) | 0.724 |
| Hybrid | 0.220 (mixer) | 0.474 (L6, pass 4) | 0.497 | 0.088 (mixer) | 0.506 (L6, pass 4) | 0.725 |
| Karvonen | 0.117 (L2) | 0.472 (L6) | 0.518 | 0.000 (L2) | 0.379 (L6) | 0.726 |

- **The temporal memory carries information about the upcoming move.** At the mixer, the from-square reads as well as at the transformer's L4 (0.263).
- **Final readouts are nearly equal across our arms.** Karvonen's model is highest.

## 8. Looped cores

| After pass | Depth: board | Depth: from-square | Hybrid: board | Hybrid: from-square |
| --- | --- | --- | --- | --- |
| 1 | 0.896 | 0.399 | **0.912** | **0.463** |
| 2 | **0.944** | **0.466** | 0.921 | 0.474 |
| 4 | 0.947 | 0.479 | 0.922 | 0.474 |

*Board: Karvonen's setting at L6. From-square: at the decision characters.*

- **Pass 1 → 2:** the depth model needs its second pass for both readouts; the hybrid has them after its first.
- **Passes 3 and 4 add almost nothing to either readout.**
- **Why the hybrid barely updates:** its passes restart mostly from L2. Entering each pass, the depth mixer's previous-pass term is 0.48–0.50× its fresh-start term in the hybrid, and 0.79–0.85× in the depth model.

## 9. Causal checks

Edits add `k × c × d` at the probed character and at every character generated after it, on every settling pass:

- `d` is the probe direction.
- `c` sets the probe's logit margin to 5.
- `k` is the multiplier.

Each condition is paired with a random direction of the same norm.

**Side to move** (×2; 200 held-out spaces, 100 for the hybrid). Each cell is the fraction of spaces where the predicted next-character type (digit or letter) flips:

| Arm | Early sites | Late sites | Random direction |
| --- | --- | --- | --- |
| Transformer | L2 0.99, L3 1.00 | L6 0.03, L7 0.00, L8 0.00 | ≤ 0.06 |
| Temporal | mixer 0.00, L2 0.51, L3 0.84 | L7 0.01, L8 0.00 | 0.00 |
| Depth | L2 0.96, L3 pass 1 1.00, L4 pass 1 0.99 | L6 pass 4 0.50, L7 0.50 | ≤ 0.06 |
| Hybrid | L2 0.56; L3 and L6 in pass 1: 0.00 | L6 pass 4 0.52, L7 0.57 | ≤ 0.04 |

- **Side to move is computed from the text at L2–L3.** Edits at the mixer do nothing, although the probe reads side to move perfectly there.
- **A flip rate near 0.50 means only one direction flips.**
- **Hybrid first-pass edits have no effect even at ×16** (40 spaces), although random pushes there do change the output. Each pass restarts mostly from the unedited L2 (section 8).

**Piece removal** (Karvonen's intervention; 150 held-out decision characters, 100 for the hybrid). Each position's original greedy move uses the deleted piece, so the unedited success rate is 0.

| Arm | Edit | Legal on modified board | Legal on original board | Random direction |
| --- | --- | --- | --- | --- |
| Transformer | L6, ×4 | 0.620 | 0.953 | 0.100 |
| Transformer | L4–L7, ×2 | 0.640 | 0.900 | 0.167 |
| Transformer | L4–L7, ×4 | 0.833 | 0.720 | 0.227 |
| Transformer | L3, ×4 | 0.040 | 1.000 | 0.020 |
| Transformer | L8, ×4 | 0.033 | 1.000 | 0.107 |
| Karvonen | L6, ×4 | 0.727 | 0.940 | 0.167 |
| Karvonen | L4–L7, ×2 | 0.767 | 0.853 | 0.153 |
| Temporal | L6, ×4 | 0.573 | 0.940 | 0.127 |
| Temporal | L4–L7, ×2 | 0.693 | 0.833 | 0.093 |
| Temporal | mixer + L2, ×4 | 0.127 | 0.980 | 0.027 |
| Depth | L6 pass 4, ×4 | 0.493 | 0.980 | 0.127 |
| Depth | L6 pass 1, ×4 | 0.000 | 1.000 | 0.000 |
| Depth | L5 in all passes, ×4 | 0.613 | 0.907 | 0.113 |
| Depth | L4–L6 pass 4 + L7, ×2 | 0.673 | 0.940 | 0.127 |
| Hybrid | mixer + L2, ×4 | 0.010 | 1.000 | 0.000 |
| Hybrid | L5 in all passes, ×4 | 0.530 | 0.860 | 0.100 |
| Hybrid | L4–L6 pass 4 + L7, ×2 | 0.670 | 0.810 | 0.060 |

- **The late board directions are causal in every arm.** The best four-site window at ×2 gives 0.64–0.69 on our arms and 0.77 on Karvonen's model, against 0.06–0.17 for random directions. Karvonen reports 0.904 from five moves sampled at temperature 1; ours is one greedy move.
- **Early edits, L8 edits and first-pass edits in the depth model do little.**
  - The mixer + L2 result matches the one-character memory swap: earlier characters still carry the true board.
  - First-pass depth edits are overwritten by later passes.
- **Large edits cost legality on the original board.**
- **These edits don't separate use at the decision character from use by later characters.** They continue through the move being written and flow into the memory.

## Open questions

- **Other characters:** the memory swap and steering were run only at decision characters. We don't know whether the partial boards at other characters are used.
- **How L2–L5 rebuild the board at a decision:** re-reading the text, or attending to past move endings, where the latest-move information is sharp.
- **What the mixer's blur makes room for:** whether the mixer output holds alternative moves as well as the one chosen.

## Limitations

- **Probe data:** probes are trained on 800 rows (600 for `cycle.py`) and tested on 200 (100). Karvonen's setting uses 26.7k training points, about a tenth of his.
- **Seeds:** one training seed per arm.
- **The temporal arm is the live-aligned export,** not the as-trained model, whose fixed point is broken (see the [20B report](../../long_runs/20B_recurrence/REPORT.md)).
- **Move readouts use the human's move,** not the model's own choice.
- **Steering** uses greedy decoding and one edit rule, with scales set per example from the probe margin, following Karvonen's dynamic scale.
