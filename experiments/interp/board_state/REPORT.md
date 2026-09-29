# Board-state probing: report

What the four 20B arms represent about the board and the side to move, at which layer and at which character, and whether they use it. Setup, data and commands are in the [README](README.md).

## Summary

- **The probes are validated.** On Karvonen's released model, our pipeline gets 0.981 of squares right at L6 in his setting (paper: 0.991, with about ten times more probe games). A random-init model scores 0.760 (paper: 0.750).
- **Board tracking depends on the character.** Every model assembles the full board at the two characters where a move is chosen: the `.` before White's move and the space before Black's. The memory models also assemble it one character earlier, at the last digit of the move number. At other characters the board stays partial even in late layers. The two spaces are the same input token, yet only the one followed by a decision gets the full board.
- **The temporal memory delivers the board early, but what it holds depends on the character that wrote it.** The memory is sharpest on the latest move right after that move is written (0.95). Before a decision, the last digit of the move number builds the full board and hands it to the `.`: the memory arriving there reads 0.86 / 0.86 (hybrid 0.86 / 0.90). In the arms without memory, that digit holds 0.53–0.57 / 0.69–0.72.
- **The temporal mixer passes older changes poorly at every character,** losing 0.09–0.17 in the temporal model and up to 0.22 in the hybrid. It passes the latest move well only while the move is fresh. At decision characters the layers after the mixer rebuild the board: temporal completes it by L5, the transformer and the hybrid by L7.
- **The mixer's input-dependent gates are indispensable.** With the gates fixed at their averages the memory never settles and the model breaks. They re-express the memory for the current character, and they don't track how settled the memory is.
- **The loss happens in the mixer's memory term itself,** not through interference from the current character.
  - Much of it is a nonlinear re-encoding that an MLP probe reads back, but not all.
  - Training the whole network partly on settled memory (the aligned-decay model) doesn't change it.
  - Doubling the memory term makes NLL much worse (0.221 → 0.295) without passing more board.
- **The temporal model gets the board from its memory, not by attending to the previous character.** Blocking that attention leaves temporal's board unchanged. In the transformer it drops the board at White's first move character from 0.65 / 0.69 to 0.28 / 0.41.
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
| Earlier move-number digit (`1` in `12.`) | 0.74 / 0.67 | 0.52 / 0.58 | 0.51 / 0.60 | 0.51 / 0.70 | 0.52 / 0.71 |
| **Last move-number digit** (before the `.`) | 0.60 / 0.69 | 0.45 / 0.55 | 0.58 / 0.62 | **0.85 / 0.83** | **0.86 / 0.86** |
| **`.`** | 0.86 / 0.86 | 0.59 / 0.70 | 0.69 / 0.70 | **0.87 / 0.89** | 0.86 / 0.90 |
| First character of White's move | 0.86 / 0.90 | 0.64 / 0.76 | 0.68 / 0.74 | 0.81 / 0.86 | 0.80 / 0.84 |
| Inside White's move | 0.79 / 0.82 | 0.56 / 0.65 | 0.61 / 0.66 | 0.74 / 0.79 | 0.73 / 0.77 |
| Last character of White's move | 0.81 / 0.77 | 0.85 / 0.61 | 0.84 / 0.64 | 0.98 / 0.76 | 0.97 / 0.77 |
| **Space after White's move** | 0.95 / 0.76 | 0.91 / 0.60 | 0.82 / 0.67 | **0.87 / 0.89** | 0.86 / 0.90 |

**Transformer.** "Previous L7" is the same readout as the temporal model's memory column, but the transformer has no memory connection to read it through.

| Role | Previous L7 | L2 | L5 | L7 |
| --- | --- | --- | --- | --- |
| Space after Black's move | 0.96 / 0.80 | 0.46 / 0.55 | 0.64 / 0.60 | 0.68 / 0.73 |
| Earlier move-number digit (`1` in `12.`) | 0.63 / 0.73 | 0.47 / 0.57 | 0.49 / 0.62 | 0.52 / 0.70 |
| Last move-number digit (before the `.`) | 0.58 / 0.69 | 0.46 / 0.55 | 0.48 / 0.58 | 0.53 / 0.69 |
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

- **The full board is built where a move is chosen, and in the memory arms one character earlier.**
  - At the two decision characters, older changes reach 0.88–0.90 at L7 in every arm.
  - At the last digit of the move number they reach 0.86 (temporal) and 0.90 (hybrid) at L7, against 0.69–0.72 in the arms without memory.
  - At a move's first character they are still 0.83–0.85 at L7; at the other characters they stay at 0.67–0.84.
  - The two spaces are the same input token, yet in every arm only the one followed by a decision gets the full board.
- **Temporal completes the board at L5 at decision characters; the transformer and hybrid only at L7.** At the `.`, older changes read 0.89 at temporal's L5 against 0.70 for the transformer and 0.84 for the hybrid's last pass. Depth reaches 0.87 at L5, but only in its fourth pass.
- **Every arm applies a move at its last character by L5** (0.98–0.99 on the moved squares). The temporal model also holds older changes better there than the transformer: 0.76 against 0.63.
- **What the memory holds depends on the character that wrote it:**
  - Right after a move, it is sharp on that move (0.95) and loose on older changes (0.76–0.77).
  - Arriving at a move-number digit, it reads 0.60–0.74 / 0.67–0.69.
  - Arriving at the `.`, it reads 0.86 / 0.86. This is the same activation as the last digit's L7 in the table.
- **The memory models build the board twice before White decides.**
  - The last digit builds it (L5 0.85 / 0.83, from an incoming memory of 0.60 / 0.69) and hands it to the `.`.
  - The mixer drops part of it (0.59 / 0.70), and the `.` builds it again (L5 0.87 / 0.89).
  - Earlier digits don't build it (L7 0.52 / 0.71).
  - Nor does the last digit in the arms without memory (L7 0.53 / 0.69 transformer, 0.57 / 0.72 depth); those arms build the board only at the `.`.
- **The temporal model's early advantage holds at every character, but its size varies.** At L2 temporal reads 0.51–0.85 / 0.59–0.74 across roles, the transformer 0.44–0.57 / 0.55–0.63.

## 3. The temporal memory and the mixer

The mixer blends the memory with the current character's embedding, using per-dimension gates. What it passes depends on the character (section 2):

- **Older changes lose 0.09–0.17 at every role in the temporal table** (0.04–0.07 at check markers). The hybrid's memory holds more but loses more: 0.90 → 0.69 at the `.`, and 0.84 → 0.64 at the space before Black chooses.
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

### Why the mixer passes only part of the board

Four explanations, tested in turn: the board survives but isn't linearly readable; it is lost; the network was trained to treat the memory as unreliable; or the model gets the board by attention instead.

**Decomposition** (`mixer_terms.py`, temporal). The mixer output is the sum of a memory term (`gate_m ⊙ W_m·norm(memory)`) and a current-character term. Each cell is latest move / older changes:

| Role | Memory | Memory term | Current term | Mixer output | Memory, MLP probe | Mixer output, MLP probe |
| --- | --- | --- | --- | --- | --- | --- |
| Last move-number digit | 0.60 / 0.69 | 0.46 / 0.56 | 0.39 / 0.53 | 0.44 / 0.55 | 0.67 / 0.76 | 0.60 / 0.68 |
| `.` | 0.86 / 0.86 | 0.58 / 0.70 | 0.44 / 0.60 | 0.59 / 0.70 | 0.89 / 0.88 | 0.82 / 0.82 |
| First character of White's move | 0.86 / 0.90 | 0.66 / 0.77 | 0.48 / 0.65 | 0.64 / 0.76 | 0.88 / 0.91 | 0.74 / 0.81 |
| Space after White's move | 0.95 / 0.76 | 0.92 / 0.60 | 0.82 / 0.56 | 0.91 / 0.60 | 0.96 / 0.80 | 0.94 / 0.72 |

- **The loss is in the memory term itself.** It reads about as well as the full mixer output, so the current character's term doesn't mask the board.
- **Much of the loss is a nonlinear re-encoding.** At the `.`, an MLP probe reads the mixer output at 0.82 / 0.82 against 0.59 / 0.70 for the linear probe. The memory itself reads 0.89 / 0.88 with an MLP, so some information is lost as well. The multiplicative, input-dependent gates are the likely cause.

**Training on settled memory** (`cycle.py`, the aligned-decay model). Its decay phase trained the whole network with 25% of batches reading settled memory (live NLL 0.2142, against 0.2176 for the aligned temporal model):

| Role | Aligned temporal: memory → mixer → L5 | Aligned-decay: memory → mixer → L5 |
| --- | --- | --- |
| `.` | 0.86 / 0.86 → 0.59 / 0.70 → 0.87 / 0.89 | 0.86 / 0.83 → 0.62 / 0.70 → 0.86 / 0.89 |
| First character of White's move | 0.86 / 0.90 → 0.64 / 0.76 → 0.81 / 0.86 | 0.85 / 0.90 → 0.60 / 0.76 → 0.80 / 0.86 |
| Space after White's move | 0.95 / 0.76 → 0.91 / 0.60 → 0.87 / 0.89 | 0.95 / 0.76 → 0.88 / 0.60 → 0.87 / 0.89 |

The mixer passes the same share of the board. Settled memory in 25% of decay-phase batches is not enough to change it. Training on settled memory throughout remains untested.

**Scaling the memory term** (`memory_scale.py`). The memory term is multiplied by 2 on every settling pass, and the model is read at its new fixed point:

| | NLL, scale 1 → 2 | `.`: mixer, scale 1 → 2 | `.`: L2 | `.`: L5 |
| --- | --- | --- | --- | --- |
| Aligned temporal | 0.2210 → 0.2951 | 0.59 / 0.70 → 0.62 / 0.73 | 0.69 / 0.70 → 0.67 / 0.72 | 0.87 / 0.89 → 0.84 / 0.87 |
| Aligned-decay | 0.2172 → 0.3050 | 0.62 / 0.70 → 0.65 / 0.72 | 0.67 / 0.71 → 0.66 / 0.71 | 0.86 / 0.89 → 0.83 / 0.88 |

A larger memory term carries hardly more board, and prediction gets much worse. The layers after the mixer depend on the memory term's learned size.

**Attention to the previous character** (`attention.py`). Attention from the focus characters to the character before them (or, as a control, two before) is blocked in every block after the mixer. The board at L5 is then read with probes fit on unablated activations:

| Role | Temporal: normal / previous blocked / two back blocked | Transformer: normal / previous blocked / two back blocked |
| --- | --- | --- |
| `.` | 0.87 / 0.89, 0.87 / 0.89, 0.84 / 0.89 | 0.83 / 0.70, 0.56 / 0.58, 0.83 / 0.69 |
| First character of White's move | 0.81 / 0.86, 0.81 / 0.86, 0.81 / 0.86 | 0.65 / 0.69, 0.28 / 0.41, 0.65 / 0.69 |
| First character of Black's move | 0.82 / 0.85, 0.76 / 0.86, 0.51 / 0.86 | 0.68 / 0.69, 0.43 / 0.61, 0.43 / 0.69 |

- **The temporal model doesn't rely on the previous character.** Blocking it leaves the board nearly unchanged, and its mean attention to the previous character is 0.01–0.10 per block.
- **The transformer depends on the previous character.** There is no memory to take that role.
- **Both models read the latest move at Black's first character from two back.** That position is the last character of White's move. Blocking it lowers latest-move squares (temporal 0.82 → 0.51, transformer 0.68 → 0.43) and leaves older changes unchanged.

**What this settles:**

- **The board is partly re-encoded and partly lost inside the mixer's memory term.** Attention to the decision character doesn't compensate for it.
- **A mixer that is merely too weak is ruled out:** scaling it up hurts.
- **Unreliable training memory is not supported,** within the limits of the aligned-decay run.
- **What remains:** the learned gate and projection, or the task not needing more. Separating these needs a model trained with a different mixer or with settled memory throughout.

### What the gating does

`gates.py`, three tests on the aligned temporal model; the aligned-decay model gives the same picture.

**The input-dependent gates are indispensable.** With each gate fixed at its per-dimension mean over 50 rows:

- The memory never settles: after 64 passes it still changes by 72% per pass (aligned-decay: 98%), against 0.1% normally.
- NLL rises from 0.221 to 3.74 (aligned-decay: 0.217 to 4.79), worse than uniform over the 32 characters (3.47).
- This shows the gates are needed, but not whether for selection or for keeping the recurrence stable.

**The gate conditions the memory on the current character.** Probes on the memory and on the memory term alone, at roles where the character varies. Board columns are latest move / older changes; character is the current character's identity:

| Role | Memory: board | Memory: character | Memory term: board | Memory term: character |
| --- | --- | --- | --- | --- |
| Earlier move-number digit | 0.73 / 0.67 | 1.00 | 0.53 / 0.58 | 1.00 |
| First character of White's move | 0.86 / 0.90 | 0.67 | 0.66 / 0.77 | 0.96 |
| Inside White's move | 0.79 / 0.82 | 0.82 | 0.57 / 0.66 | 0.96 |
| Last character of White's move | 0.81 / 0.77 | 0.94 | 0.82 / 0.62 | 0.99 |

- **The memory term carries the current character better than the memory does** (0.96–0.99 against 0.66–0.94). The gates bring it in from the current character.
- **So the memory term is the memory re-expressed for this character,** with less of the board linearly readable.
- A character-dependent code would explain why an MLP recovers more than a linear probe. It doesn't explain the `.`, where the character is always the same and the loss is as large.
- Whether the gate also strips the previous character's stale prediction can't be read from this, because the gate injects the true character.

**The gates don't track memory quality.** Mean memory gates for memories after 1, 2 and 3 updates, as in training, and for the settled memory:

| Model, role | 1 update | 2 updates | 3 updates | Settled |
| --- | --- | --- | --- | --- |
| As-trained temporal, `.` | 0.098 | 0.116 | 0.109 | 0.113 |
| As-trained temporal, first character of White's move | 0.176 | 0.176 | 0.189 | 0.187 |
| Aligned temporal, `.` | 0.114 | 0.136 | 0.133 | 0.134 |
| Aligned temporal, first character of White's move | 0.192 | 0.190 | 0.193 | 0.192 |

Across all tested roles and models the means move by at most 0.022, with no consistent direction. That includes the as-trained model, whose mixer only ever saw memories of 1–3 updates. The gate is not a reliability detector, at least on average; per-dimension patterns remain untested.

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
- **Why the hybrid barely updates between passes:**
  - Each pass after the first starts from `W_prev·norm(L6 of the previous pass) + W_fresh·norm(L2)`. L2 is the same in every pass and, in the hybrid, already contains the temporal memory.
  - The previous-pass term is a median 0.48–0.50× the size of the fresh-start term in the hybrid, and 0.79–0.85× in the depth model.
  - So each hybrid pass is pulled about 2:1 back toward the same L2 state. A size ratio says nothing about information by itself; the causal check is in section 9: a first-pass edit has no effect in the hybrid and a strong effect in the depth model.

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
- **Hybrid first-pass edits have no effect even at ×16** (40 spaces), although random pushes there do change the output. Each later pass weights the unedited L2, which already holds the correct side to move, about twice as much as the edited previous pass (section 8).

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
- **How the temporal model rebuilds older changes at a decision:** not from the previous character (section 3). Whether it reads them from earlier move endings, where each move's change is sharp, is untested.
- **What the mixer's blur makes room for:** whether the mixer output holds alternative moves as well as the one chosen.
- **Whether the board is fully present per character:** probes fit separately for each current character would show whether the memory term keeps the board in a character-dependent code.

## Limitations

- **Probe data:** probes are trained on 800 rows (600 for `cycle.py`) and tested on 200 (100). Karvonen's setting uses 26.7k training points, about a tenth of his.
- **Seeds:** one training seed per arm.
- **The temporal arm is the live-aligned export,** not the as-trained model, whose fixed point is broken (see the [20B report](../../long_runs/20B_recurrence/REPORT.md)).
- **Move readouts use the human's move,** not the model's own choice.
- **Steering** uses greedy decoding and one edit rule, with scales set per example from the probe margin, following Karvonen's dynamic scale.
