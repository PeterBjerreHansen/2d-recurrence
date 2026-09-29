# Move-target training: plan

> **Status: proposal; nothing has been run.** This document is the single source of truth for the `engine-policy` branch. Read [Pitfalls](#pitfalls) before implementing anything. Record every decision that changes the design under [Decisions](#decisions), with its reason; don't edit frozen items silently.

## The question

**Does the training target decide which recurrence axis pays off?**

The 20B study trained a transformer, a temporal-only, a depth-only and a hybrid model to predict the next character of Lichess games ([20B report](../experiments/long_runs/20B_recurrence/REPORT.md)).
- All three recurrent models beat the transformer, with the gains in move choice.
- The hybrid tied temporal-only, and depth added little.
- Deployed, aligned temporal matched the hybrid at far fewer block applications.

**Hypothesis:** imitating players rewards tracking the board, which temporal recurrence does well, but asks for no more calculation than the players did. Depth recurrence should pay off only when the target needs calculation.

**Test:** train the same models, on the same games, under three targets for the next move:

| Condition | Target | What it needs | Expected to favour |
| --- | --- | --- | --- |
| **Human** (reference) | the move played | the board, and a model of the player | temporal |
| **Legal** | which moves are legal | the board and the rules | temporal |
| **Engine** | the value of every legal move | the board and an evaluation, which calculation can improve | depth, or both axes together |

## Design at a glance

- **One token per ply**, in from/to (UCI) notation. The input vocabulary and the output layer are the same 1,968 moves, so the three conditions are one language model with three targets.
- **Two stages.**
  - Human and legal train from scratch.
  - Engine continues from **both**: the legal-stage checkpoints (the primary recipe) and the human-stage checkpoints (the control).
- **Four arms**, three seeds each, at the same token budget.
- **The central test** is an interaction: do extra core iterations (J = 1 → 4) help the engine-trained models more than the human-trained ones, on the same positions? A gain from loops in one model alone doesn't show the target caused it.

## Why these choices

- **Move tokens, not characters.**
  - The architectures are unchanged; this is a tokenizer choice, like word-level tokens.
  - It removes the character cycle (move numbers, dots, spaces), so every token is a decision and each temporal memory step is one ply.
  - It makes the output layer the same for every condition.
  - It is the tokenization of the closest prior work ([related work](chess_related_work.md)): Chess-World-Model, Allie, Toshniwal et al.
- **Legal before engine.**
  - Legality needs the board and the rules but no evaluation, so it isolates state tracking.
  - Continuing into the engine stage keeps annotation affordable: engine calls, not GPU time, are the cost.
- **Per-move values, not a policy.** [Ruoss et al. (2024)](https://arxiv.org/abs/2402.04494) found predicting a value for every move a stronger target than imitating the engine's best move. Our output layer scores all moves in one pass. The head and loss differ from theirs (see [Losses](#losses)), so their result motivates the design but doesn't validate it.
- **Random games in training.** Human positions follow familiar patterns. Uniformly random legal games force the rules to be learned rather than the patterns.

## Specification

### Tokens and model

- **Vocabulary:**
  - the 1,968 moves ChessBench uses: every from/to pair a queen or knight could make on an empty board (1,792), plus the four promotions for both colours (176);
  - castling is the king's move (`e1g1`), and en passant is the pawn's move (`e5d6`);
  - plus a game-start token and padding.
- **Arms:** transformer, temporal-only, depth-only and hybrid, with the 20B layout: 8 layers, width 512; prelude 1, T-buffer 1, core 4, T-source 1, coda 1.
- **Context:** 256 tokens. Every human game fits: a 1,023-character row holds at most 255 plies, since each ply takes at least four characters with its share of the move number. The conversion checks this for every game.
- **Output layer:**
  - untied from the input embedding;
  - one per condition; the engine stage starts a fresh one;
  - a 512 × 1,968 linear layer, identical in every arm.

### Losses

All losses apply at positions whose next ply exists in the same game, on the final pass only.

- **Human:** softmax cross-entropy with the move played.
- **Legal:** one sigmoid per move, binary cross-entropy with target 1 for legal and 0 for illegal. Average over the 1,968 moves per position, then over positions. Report the loss on legal and on illegal moves separately, against a constant-rate baseline: about 1.8% of moves are legal, so a constant predictor already scores about 0.089 nats.
- **Engine:** one sigmoid per move, binary cross-entropy against $Q$ as a soft label, on **legal moves only**. Average over legal moves per position, then over positions.
- **Weighting:** any weighting, such as balancing legal and illegal moves or weighting by rank, needs a pilot result showing it's necessary. It is frozen before the main runs.

### Engine values

- **Perspective:** $Q(s,a)$ is always from the perspective of the player making move $a$.
- **Conversion:** $Q = \sigma(0.00368208\,\mathrm{cp})$, with centipawns capped at ±1,500, so non-mate values lie in [0.004, 0.996].
- **Mates:**
  - mate in $n$ for the mover maps to $0.997 + 0.002/n$;
  - being mated in $n$ maps to $0.003 - 0.002/n$;
  - every mate outranks every non-mate value, and shorter mates are strictly better.
  - Whether this stops the model from wandering between winning plans, as Ruoss et al. report for their models, is a hypothesis to test.
- **The teacher sees the history.**
  - Each position is sent to Stockfish as the start position plus the moves played, so repetitions and the 50-move rule count.
  - Labels are cached by position together with the positions since the last capture or pawn move, never by FEN alone.
  - Labels reused from ChessBench are keyed by FEN, so they form a **board-only teacher**. If used, they are reported as a separate condition.
- **Annotation protocol:**
  - a pinned Stockfish binary (its SHA-256 recorded) with its bundled network, 1 thread, a fixed hash size, a new game between positions, and no tablebases;
  - **one search per legal move,** of the position after it, with a fixed node budget, and the score negated to the mover's perspective. A single all-moves search under one budget gives low-ranked moves shallower searches, so disagreement between budgets would partly measure unequal work;
  - **two budgets,** shallow and deep. Disagreement between them marks positions where calculation changes the answer;
  - store the score, its type (centipawn, mate, bound), the depth and the nodes;
  - re-annotating a sample in a different order must give identical labels.
- **A stronger, separate teacher** (more nodes) labels a small held-out set, to check that gains aren't just fitting the training teacher.

### Games

- **Human games:** `chess_8M_v1` (Karvonen's Lichess release), converted to move tokens by replaying the SAN.
  - Each row starts at a game start and holds about three games. The last one is cut by the row end; drop its partial ply and keep the prefix.
  - The training split holds about 1.5B plies: 5.5 characters and 62 plies per game on average, measured on 2,000 rows.
- **Random games:** uniformly random legal plies from the start position until the game ends or 255 plies are reached.
- **Mixture:** human and legal runs see the same stream, with 80% of tokens from human games and 20% from random games. The engine stage uses human games only.
- **Budget:** one pass over the human plies plus random plies at 20% of the total, about 1.9B tokens per run. Log four counters separately: human plies, random plies, supervised positions, and row tokens including padding.
- **Packing:**
  - rows of 256 tokens, each game starting with the game-start token;
  - games are never split across rows, and row tails are padded;
  - state carries across games within a row, as in the character data;
  - evaluation places each game first in its row;
  - the pilot tests whether a game's outputs change when it follows another game.
- **Split:** by a hash of the move sequence. Random test games are deduplicated against training.

### Training protocol

- **Optimiser:** the 20B study's AdamW settings, with the same small learning-rate check for each objective on the transformer, applied to every arm.
- **Update support:** gap-free, with a broad mixture throughout.
- **Warm-start batches in the decay phase: temporal-only.** The trainer implements them for temporal only (`train.py`), and the 20B hybrid ran live at its training-graph quality without them. This is a difference between training recipes, and is stated as one.
- **Live evaluation during training.**
- **Seeds:** three per arm and condition.
- **Checkpoints** on the 5B protocol's schedule, so readouts after each loop and probes can be added later.

## Evaluation

### Execution

- **Live execution is primary.** It uses the cached implementation, tested against the slow reference `inference.reference` ([inference contract](INFERENCE_CONTRACT.md)).
- **KV cache:** `depth_specialized` for depth-only and hybrid, as in the 20B battery.
- **Training-graph results** are reported alongside. Their gap to live execution is a measured property, not an error.
- **Loops:** depth and hybrid run at J = 1, 2 and 4. J > 4 is reported separately, as extrapolation.

### Test sets

- **Development sets** for the pilot, thresholds and margins, kept separate from the untouched final test sets.
- **Held-out human games.**
- **Held-out random games.** These are in the training distribution of the random generator: unfamiliar trajectories, not a distribution shift.
- **An unseen generator:** Stockfish self-play at a low skill level. This carries the distribution-shift claim.
- **Targeted rule sets:** en passant, castling rights lost by moves that were later undone, pins, and pairs of identical positions with different repetition histories. The last group checks the teacher too.

### Metrics

Each metric is used only where it means something.

| | Human | Legal | Engine |
| --- | --- | --- | --- |
| Legality: probability on legal moves (human), precision and recall at 0.5 (legal), exact-set accuracy | ✓ | ✓ | — the head never saw illegal moves |
| Errors on **almost-legal moves**: illegal only because of a pin, a check or a blocked line | ✓ | ✓ | — |
| Regret of the top-scoring legal move, $\max_{a'}Q - Q(\hat a)$ | ✓ | — every legal move scores alike in an ideal head | ✓ |
| Best-move agreement (near-ties count as agreement) and rank correlation with $Q$ allowing ties (Kendall's τ-b) | ✓ | — | ✓ |

- **Legality** is always judged by the rules at evaluation, never by a value head.
- **The human model's move quality** is taken from its most probable legal move. Imitation models play better at low temperature ([Zhang et al., 2024](https://arxiv.org/abs/2406.11741)).

**Strata:**
- ply;
- history-dependent legality;
- in check, promotions;
- shallow–deep disagreement (engine and human);
- forcing lines: mate in 1 and mate in 2 by check, with matched controls. Rebuilt from `experiments/interp/board_state/forcing.py` on the `interp` branch.

### Primary analyses

Declared now; margins and thresholds are set on the development sets before the main runs.

1. **Learning speed (stage 1):** tokens seen until each arm reaches a fixed exact-set accuracy, in the human and legal conditions, with board probes as a separate diagnostic. Legality on human games will likely saturate near 100% for every arm, as board state does in [Chess-World-Model](https://arxiv.org/abs/2605.30100). So speed, the almost-legal moves and the unseen generator carry the comparison, not final in-distribution accuracy.
2. **Target × loops (the central test):** for depth and hybrid, the regret reduction from J=1 to J=4 in the engine continuation minus the same reduction in the human-trained model, on the same annotated positions. Also whether it is larger on high-disagreement positions than on low ones. Both are tested as interactions, not as two separate significance tests.
3. **Arms in the engine stage,** at matched data:
   - hybrid vs temporal at J=1, which matches block applications;
   - hybrid vs depth at J=4, which also matches;
   - hybrid J=4 vs temporal, reported with measured latency, prefill and cache bytes. This is not matched compute.

**Statistics:**
- effects for each seed shown individually;
- intervals from bootstrapping games, with each game's positions resampled together;
- a result counts only if it exceeds the practical margin, not merely if its interval excludes zero.

**Diagnostic, not primary:** an engine stage continued from legal-stage checkpoints matched on exact-set accuracy, with the matching rule fixed in advance.

## What the design can and cannot show

**Can show:**
- how whole training recipes compare, at matched data;
- whether engine targets increase the benefit of loops relative to human targets;
- whether legality targets teach the board faster.

**Cannot show, without more evidence:**
- compute-optimality: temporal-only has no 20-block configuration;
- that loops perform **search** or **look-ahead**: that needs mechanism evidence;
- that "chess doesn't need depth": a null means no detected benefit at this size, target and budget;
- anything about history-dependent values from a board-only teacher;
- that the theory decides this: [Merrill et al. (2024)](https://arxiv.org/abs/2404.08819) is about exact tracking over arbitrarily long sequences, which motivates temporal recurrence but predicts nothing for 256 tokens.

## Pitfalls

For anyone implementing or analysing on this branch:

1. **The training graph is not live execution** for temporal and hybrid. Never change recurrence semantics to make them agree. Correctness means three things:
   - cached live execution equals the slow reference;
   - the training graph iterated to its fixed point equals live execution;
   - depth-only with `depth_specialized` caches equals its training cell $(0, J-1)$.
2. **Heads answer only their own question.** An engine head says nothing about legality; a legality head's ranking of legal moves says nothing about quality.
3. **$Q$ is from the mover's perspective.** Scores of the position after a move come from the opponent's side and must be negated. Test with winning positions for White to move and for Black to move.
4. **A small legality loss doesn't mean the rules are learned.** Compare with the constant baseline, and read precision, recall and the almost-legal errors.
5. **"Moves" means plies** everywhere in code and results. Token budgets count human plies, random plies, supervised positions and row tokens separately.
6. **No tuning or checkpoint selection on final test sets.**
7. **Recipes differ on purpose** (warm-start batches for temporal only). Don't equalise them silently, and don't describe the results as pure architecture effects.
8. **Repository hygiene:**
   - data, `.venv` and results live only in the main checkout;
   - never commit symlinks or anything under a `results` path; tracked `results` links destroyed the 20B checkpoints;
   - stage explicit paths, never `git add -A`.

## Pilot

Small from-scratch runs of all four arms, human and legal, one seed, plus a second seed of one arm to estimate seed variance. Then a short engine stage on a few hundred thousand annotated positions, continued from both the legal and the human checkpoints. The old 5B checkpoints are character models and can't be reused.

**Pass criteria** (none requires an arm to win):
1. all tests pass;
2. label and training throughput are adequate, with their cost measured;
3. development sets have room to improve. If every arm sits at the ceiling, use harder sets or a smaller scale, both declared in advance;
4. learning curves and seed variance are measured;
5. the packing-order effect is measured;
6. teacher labels are reproducible, and Stockfish throughput at both budgets is known.

Also check the ChessBench join (coverage and cost) and Chess-World-Model's code, which its paper links.

## Steps

1. **Tokenizer, conversion, teacher and tests:**
   - SAN → tokens → SAN round trip;
   - the vocabulary covers every ply in the data and in random games;
   - legality targets match python-chess;
   - losses touch only valid positions, on the final pass;
   - the three execution checks in [Pitfalls](#pitfalls);
   - $Q$ perspective and the mate mapping;
   - reproducible annotation;
   - the repetition-history pairs get different labels where they should.
2. **Model and trainer:**
   - token vocabulary and context length from the config;
   - untied output layers;
   - the three losses;
   - the random-game generator and the packed loader;
   - the four counters.
3. **Pilot**, then fix every open decision below.
4. **Human and legal runs:** 4 arms × 3 seeds each.
5. **Annotation, then both engine continuations:** 4 arms × 3 seeds each.
6. **Report** next to the scripts.

## Decisions

**Open until the pilot:**
- batch size and update count;
- learning rate for each objective;
- the random-game fraction (20% by default);
- loss weighting;
- node budgets;
- number of stage-2 positions;
- practical margins;
- the exact-set threshold for learning speed.

**Log:** (date — decision — reason)

## Cost

Estimate from the character runs, to be re-measured in the pilot:
- **Human and legal runs:** about $3–6 each at about 1.9B tokens, so $70–150 for all 24. The larger output layer, the sigmoid losses, padding, legal-move generation and warm-start passes all add to that.
- **Engine continuations:** short.
- **Annotation:** about 35 searches per position at two budgets. The pilot sets the budget and the number of positions.

Storing legal-move lists for every training position would take about 105 GB as `uint16` indices, so they are generated in the loader unless the pilot shows that's too slow.

## Later extensions

- **The Chess-World-Model benchmark** with its own interface (board target, auxiliary fields, protocol) for a direct comparison with its published baselines.
- **Board-given control:** the same targets with the current position as input. It tests whether engine-stage advantages come from reconstructing the board.
- **A look-ahead target:** predict the engine's principal variation, if loops show no benefit.
- **Harder evaluations:** Lichess puzzles with their game histories, and play against fixed Stockfish levels. Tactical accuracy and playing strength can come apart ([Miłosz et al., 2026](https://arxiv.org/abs/2608.27757)).
