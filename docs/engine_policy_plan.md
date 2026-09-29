# Move-target training: plan

> **Status: proposal.** Nothing here has been run. It replaces the earlier engine-policy fine-tune plan.

## Why

The 20B study trained four architectures on next-character prediction over Lichess games ([20B report](../experiments/long_runs/20B_recurrence/REPORT.md)):
- recurrence beats the transformer, and the gains are in move choice and grow over the game;
- in the training graph the hybrid ties temporal-only, and depth-only trails;
- at deployment, aligned temporal matches the hybrid at far fewer block applications.

Next-character prediction on unfiltered games applies one pressure: imitate the player. It rewards tracking the board, and temporal recurrence does that well, but the targets contain no more calculation than the players did. That board state is learned implicitly from game text is established, here and by [Karvonen (2024)](https://arxiv.org/abs/2403.15498).

This plan keeps the input, game text, and changes the pressure. The model is trained to output a distribution over moves at each decision, against one of two targets:
- **legal moves**, uniform: requires the board and the rules;
- **engine-weighted moves**: requires the board and an evaluation, which calculation can improve.

**The question:** which pressure favours which recurrence axis? Temporal recurrence should gain most under the legal-move target, depth recurrence under the engine target, and the hybrid wherever both are needed.

**Closest prior work** (surveyed in [related work](chess_related_work.md)):
- **[Ruoss et al. (2024)](https://arxiv.org/abs/2402.04494)** distil Stockfish into search-free transformers that read the **board** (FEN). Without history they cannot see threefold repetition and wander between winning plans. Stage 2 is their setting with the game history as input instead, so the model has to reconstruct the board itself.
- **[Walker & Lyons (2026)](https://arxiv.org/abs/2605.30100)** show recurrence across tokens (linear RNNs) beats transformers at tracking the chess board from moves, at small scale. They test neither loops within a token nor a decision target.

The 20B checkpoints are lost; their results stand in the report. Every run below starts from scratch.

## Design

### Input and output

- **Input:** the character stream, unchanged (`;1.e4 e5 2.Nf3 …`). Every row begins at a game start, so every position is reachable from move 1.
- **Decision characters:** the character just before a move. For White it is the `.` after the move number; for Black, the space after White's move. Earlier probing found the board is built at these characters (board-state report, `experiments/interp/board_state/REPORT.md` on the `interp` branch).
- **Supervised decisions:** a decision is supervised when the next character begins a move. This excludes game ends and moves cut off by the row end.
- **Move head:** one linear layer from the final normalised residual to **1,968 moves**, the vocabulary ChessBench uses, so its labels map directly:
  - every from/to pair a queen or knight could make on an empty board (1,792);
  - promotions to queen, rook, bishop and knight for both colours (176);
  - castling is the king's move (`e1g1`), and en passant is the pawn's move (`e5d6`).

  It adds 1.0M parameters, about 4% of the model, identically in every arm.
- **No next-character loss.** Only decision characters produce an output; every other character is pure computation. A move is converted to text outside the model, from the game history, when needed.

The head asks for more than the character task did. `Nf3` doesn't say which knight moved, but the from-square does, so the head needs every piece's location. Tracking pieces for from/to moves is one of the problems [Merrill et al. (2024)](https://arxiv.org/abs/2404.08819) show fixed-depth transformers cannot solve exactly. Temporal feedback passes a nonlinear state from each character to the next, so it is not bound by that limit in principle.

### Targets

Let $L(s)$ be the legal moves in position $s$.

- **Legal:** $\pi(a \mid s) = 1/|L(s)|$ for $a \in L(s)$.
- **Engine:** $\pi(a \mid s) \propto \exp\!\big(A(s,a)/\tau\big)$ over $L(s)$, with $A(s,a) = Q(s,a) - \max_{a'} Q(s,a')$ and the Lichess conversion from centipawns
  $$Q = \tfrac12 + \tfrac12\left(\frac{2}{1 + e^{-0.00368208\,\mathrm{cp}}} - 1\right).$$
  Forced mates map to 1 or 0. $Q$ is a utility surrogate from one pinned engine, not a calibrated win probability. With $\tau = 0.02$, a move that loses 0.05 of $Q$ gets about 8% of the best move's weight.

The legal target is the $\tau \to \infty$ limit of the engine target, so both stages use one objective and change only the temperature.

**Loss:** cross-entropy between $\pi$ and the head's softmax, at supervised decisions only, on the final pass only (as in all our recurrent training). Illegal moves have zero target, so the loss also trains legality. The game continues with the move actually played, so the input distribution is the same as in pretraining.

### Conditions

| Condition | Target | Training |
| --- | --- | --- |
| **Legal** (stage 1) | uniform over legal moves | from scratch |
| **Engine** (stage 2) | engine-weighted, $\tau$ fixed in advance | continues each arm's stage-1 checkpoint, same budget for every arm |
| **Reference** | next character (human imitation) | from scratch, same protocol |

- **Stage 2 continues from stage 1** because engine annotation, not GPU time, is the cost. Training from scratch would need billions of annotated positions. A from-scratch engine arm for temporal and hybrid is an optional check that the curriculum is not driving the result.
- **Matched budget, not matched performance.** Every arm gets the same stage-1 budget. Checkpoints are kept, so a comparison at matched stage-1 performance can be run afterwards as a secondary analysis.
- **The reference condition** differs from the others in output as well as target. Comparing it with them is a comparison between tasks, and is reported as such.

### Arms and protocol

- **Arms:** transformer, temporal-only, depth-only, hybrid: the 20B model and layout.
- **Budget:** 5B characters per stage-1 and reference run, the [5B protocol's](../experiments/long_runs/5B_axis/README.md) horizon and batch. The 20B study showed the gaps shrink with scale, so the smaller horizon gives up little for a comparison between pressures.
- **Seeds:** three per arm and condition.
- **Fixes from the 20B study:**
  - gap-free update support with a broad mixture throughout;
  - warm-start batches in the decay phase;
  - live evaluation during training.
- **Evaluation split:** game-disjoint from the training rows. The data manifest asserts only that no exact row is shared, so the evaluation games are hashed against the training rows used and overlaps dropped. A confirmation set is reserved before any decision is made.

## Data

### Legal-move targets

Replay each row with python-chess and emit, for every supervised decision, its position and the indices of its legal moves.
- A 5B-character run holds roughly 750M decisions.
- Generate the targets in data-loader workers, or precompute them once as `uint16` indices with offsets.
- Choose between the two after the throughput benchmark.

### Engine annotation

- **Scope:** stage-2 training positions (about 2M, sized after the pilot) and the evaluation positions.
- **Engine:** a pinned Stockfish binary (not installed yet), MultiPV over every legal move, and a fixed node budget so labels are deterministic.
- **Two budgets per position.** Disagreement between the shallow and the deep evaluation marks positions where calculation changes the answer. The gap between the best and second-best move does not: a large gap can be a trivial recapture.
- **Storage:** legal moves and $Q$ values per position. $\pi$ is computed at training time, so $\tau$ can change without re-annotating.
- **Existing annotations:** [ChessBench](https://github.com/google-deepmind/searchless_chess) (Ruoss et al., 2024) scores every legal move with Stockfish 16, for positions from 10M Lichess games from February 2023.
  - The action-value set is 1.1 TB and keyed by position. It keeps no game IDs or move histories.
  - To use it: download that month's Lichess games, render them in our text format, replay them, and look up each position. Those games then become the stage-2 input text.
  - Check coverage and cost in the pilot. If the join works, most of the annotation cost disappears. The players' ratings and time controls may differ from our training data.

## Evaluation

All metrics are computed at every supervised decision, with live execution as the primary setting and the training graph alongside. Depth and hybrid are evaluated at J = 1, 2 and 4 core iterations. J beyond the trained maximum is reported separately, as extrapolation.

**Legal stage:**
- illegal mass, the probability the head puts on illegal moves;
- KL from the uniform legal target;
- exact-set accuracy: the $|L(s)|$ most probable moves are exactly the legal set;
- **out of distribution:** the same metrics on games of uniformly random legal moves, rendered as text. Following [Walker & Lyons (2026)](https://arxiv.org/abs/2605.30100), this separates arms that learned the rules from arms that learned common patterns, after in-distribution metrics saturate.

**Engine stage:**
- legal-conditional expected regret, $\sum_{a \in L} \tilde p(a)\,\big(\max_{a'} Q - Q(a)\big)$, where $\tilde p$ is the head's distribution renormalised over legal moves. Illegal mass is reported next to it, so an arm cannot look better by moving mass off legal moves;
- best-move agreement;
- KL from $\pi$.

**Strata:**
- ply bins;
- **history-dependent legality:** en passant available, and castling rights lost by earlier king or rook moves. These can't be read from where the pieces stand;
- in check, pinned pieces, promotions;
- **engine stage only:** shallow-versus-deep disagreement, and the forcing lines from `experiments/interp/board_state/forcing.py` on the `interp` branch (mate in 1, mate in 2 by check, matched controls).

**Learning speed:**
- board-probe accuracy and legal-stage metrics against characters seen, for the legal and reference conditions, with the existing probe tools (`experiments/interp/board_state/` on the `interp` branch);
- this tests whether move targets teach the board faster than next-character prediction.

**Reference condition:** its move probabilities come from scoring each legal move's text. They are reported at the model's own temperature; low-temperature decoding alone improves imitation models' play ([Zhang et al., 2024](https://arxiv.org/abs/2406.11741)).

**Optional:** fit a next-character head to a frozen legal-stage model, to test whether training on moves alone makes the text predictable.

## Predictions and decision rules

Fixed before the main runs. Practical margins are set after the pilot.

- **Legal stage:** temporal and hybrid beat depth and transformer, with larger gaps on history-dependent strata and late plies; J changes little.
- **Engine stage, depth helps:** hybrid J=4 has lower regret than hybrid J=1, and the gain is larger on high-disagreement positions than on low ones. Both are paired across positions, with the interval across the three seeds excluding zero.
- **Engine stage, complementarity:** the hybrid beats temporal-only on regret at matched block applications, and beats depth-only at J=4.
- **A null result** means no detected benefit at this model size, target and budget, not that chess doesn't need depth.

## Pilot

The surviving 5B checkpoints (temporal, depth and hybrid; the 5B transformer is lost) are in `experiments/long_runs/5B_axis/runpod_live_backups/` (local, not tracked).
- Attach the move head and fine-tune each briefly on the legal target, with the same rows and updates for every arm.
- These trunks were pretrained on next characters, so the pilot only checks feasibility.

**Pass conditions:**
1. the targets and loss are correct end to end (the tests below pass, and live equals the training graph);
2. the legal task doesn't saturate within the budget: illegal mass and exact-set accuracy still separate the arms;
3. legal-move generation keeps up with training throughput.

The Stockfish benchmark runs in parallel.

## Steps

1. **Targets and tests:**
   - the move vocabulary covers every legal move of the replayed games, including castling, en passant and every promotion;
   - decision characters match the SAN parse;
   - targets sum to one with zero mass on illegal moves;
   - the loss touches only supervised decisions, on the final pass;
   - the head's output at the live fixed point equals the training graph.
2. **Model and trainer:** add the optional move head and the decision loss to the model and trainer, and the target stream to the data loader.
3. **Pilot** on the 5B checkpoints; Stockfish benchmark; ChessBench join check.
4. **Freeze the protocol:** budgets, $\tau$, margins, evaluation split.
5. **Stage 1 and reference:** four arms × three seeds each.
6. **Annotation and stage 2:** four arms × three seeds.
7. **Report** next to the scripts.

## Cost

| Item | Estimate |
| --- | --- |
| Stage 1, 4 arms × 3 seeds, 5B characters | about $100: the 20B study cost $135 for four arms, and 5B is a quarter of that |
| Reference, 4 arms × 3 seeds | about $100 |
| Stage 2 fine-tunes | a few GPU-hours per arm |
| Engine annotation | pending the benchmark; the earlier estimate was about 100 core-hours per 1M positions, so tens of dollars for 2M positions at two budgets |
| Pilot | a few GPU-hours |

The head adds almost no compute: the trunk still runs on every character.

## Later extensions

- **Board-given control:** the same targets with the current position given as a FEN prefix. Temporal recurrence has less to reconstruct there, so its advantage should shrink while depth's remains.
- **Harder evaluations:** Lichess puzzles with their game histories, split by solution length; play against fixed Stockfish levels.
- **A value for every move:** predict $Q$ for all 1,968 moves at once, masked to the legal ones, instead of a softmax policy. [Ruoss et al. (2024)](https://arxiv.org/abs/2402.04494) found action values the strongest target; with a head over the whole vocabulary it costs one forward pass per position, not one per move.
