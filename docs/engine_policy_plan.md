# Move-target training: plan

> **Status: proposal.** Nothing here has been run. It replaces the earlier engine-policy fine-tune plan.

## Why

The 20B study trained four architectures to predict the next character of Lichess games ([20B report](../experiments/long_runs/20B_recurrence/REPORT.md)):
- recurrence beats the transformer, and the gains are in move choice and grow over the game;
- in the training graph the hybrid ties temporal-only, and depth-only trails;
- at deployment, aligned temporal matches the hybrid at far fewer block applications.

Imitating the player applies one pressure. It rewards tracking the board, which temporal recurrence does well, but the targets contain no more calculation than the players did. That board state is learned implicitly from game sequences is established, here and by [Karvonen (2024)](https://arxiv.org/abs/2403.15498).

This plan keeps the model and the input games and changes the pressure. It trains the same language model over a vocabulary of moves, with three different targets for the next move:

| Condition | Target | Requires |
| --- | --- | --- |
| **Human** (reference) | the move played | the board, and imitating the player |
| **Legal** (stage 1) | which moves are legal | the board and the rules |
| **Engine** (stage 2) | the value of every move | the board and an evaluation, which calculation can improve |

**The question:** which pressure favours which recurrence axis? Temporal recurrence should gain most under the legal target, depth recurrence under the engine target, and the hybrid wherever both are needed.

**Closest prior work** (surveyed in [related work](chess_related_work.md)):
- **[Ruoss et al. (2024)](https://arxiv.org/abs/2402.04494)** distil Stockfish into search-free transformers that read the **board**. Predicting a value for every move was their strongest target. Without the history, their models can't see threefold repetition and wander between winning plans. Stage 2 is their setting with the game history as input, so the model has to reconstruct the board itself.
- **[Walker & Lyons (2026)](https://arxiv.org/abs/2605.30100)** show linear RNNs beat transformers at tracking the board from move tokens at small scale. They test neither loops within a token nor a decision target.
- **[Merrill et al. (2024)](https://arxiv.org/abs/2404.08819)** prove fixed-depth transformers cannot track chess state exactly in from/to notation, the notation used here. Temporal feedback passes a nonlinear state from each token to the next, so it is not bound by that limit in principle.

The 20B checkpoints are lost; their results stand in the report. Every run below starts from scratch.

## Design

### Tokens

- **One token per move,** in from/to notation (UCI). The vocabulary is the 1,968 moves ChessBench uses:
  - every from/to pair a queen or knight could make on an empty board (1,792);
  - promotions to queen, rook, bishop and knight for both colours (176);
  - castling is the king's move (`e1g1`), and en passant is the pawn's move (`e5d6`).
- **Special tokens:** a game-start token and padding, 1,970 input tokens in all.
- **This is a tokenizer choice.** The architectures are unchanged; only the unit of the sequence is a move instead of a character, as a word-level tokenizer would use words.
- **Every token is a decision:** the output after a move token is a prediction for the next move. The character cycle of the earlier models (move numbers, dots, spaces) is gone. Each temporal memory step is exactly one move.

### Model

- **Arms:** transformer, temporal-only, depth-only and hybrid, with the 20B model's layout: 8 layers, width 512, prelude 1, T-buffer 1, core 4, T-source 1, coda 1.
- **Context:** 256 tokens. Our human games are at most 185 moves long (see below), and random games are truncated to fit.
- **Output layer:**
  - untied from the input embedding;
  - a 512 × 1,968 linear layer, about 1M parameters, identical in every arm;
  - each condition trains its own output layer, and stage 2 starts a fresh one.

### Targets and losses

Losses apply at every position whose next move exists in the same game, on the final pass only, as in all our recurrent training.

- **Human:** softmax over the 1,968 moves, cross-entropy with the move played. Ordinary next-token prediction.
- **Legal:** one sigmoid per move, binary cross-entropy with target 1 for legal moves and 0 otherwise, averaged over the vocabulary. The loss can reach zero.
- **Engine:** one sigmoid per move, binary cross-entropy against $Q(s,a)$ as a soft label, on legal moves only. Illegal moves get no loss here; legality is stage 1's job.
  - $Q$ uses the Lichess conversion $Q = \sigma(0.00368208\,\mathrm{cp})$, so the head's output before the sigmoid is a linear estimate of the centipawn score.
  - Binary cross-entropy against $Q$ is the two-bin case of the binned value targets in [Ruoss et al. (2024)](https://arxiv.org/abs/2402.04494): spreading $Q$ over bins at 0 and 1 gives exactly $(1-Q, Q)$. It is classification, not regression.
  - **Mates:** centipawns are capped at ±1,500, so non-mate values lie in [0.004, 0.996]. A mate in $n$ for the side to move maps to $1 - 0.001\min(n, 3)$ (0.999, 0.998, 0.997), and being mated to the mirror image. Values stay finite, every mate outranks every non-mate score, and shorter mates are worth more. That last point addresses the wandering between winning plans that Ruoss et al. report.
  - **The chosen move:** the legal move with the highest predicted value. Legality at evaluation comes from the rules.

### Conditions and stages

- **Human and legal** train from scratch on the same games, budget and schedule.
- **Engine** continues each arm's legal-stage checkpoint with a fresh output layer, the same budget for every arm, on human games only. Continuing is what makes stage 2 affordable: engine annotation, not GPU time, is the cost.
- **Matched budget, not matched performance.** Checkpoints are kept, so a comparison at matched legal-stage performance can be run afterwards as a secondary analysis.
- **Optional:** also continue the human-condition checkpoints on the engine target. This tests whether pretraining on legality or on human moves makes value learning easier.

### Games

- **Human games:** the Lichess games of the existing dataset (`chess_8M_v1`, from Karvonen's release), converted to move tokens by replaying the SAN.
  - Each 1,024-character row starts at a game start and holds about three games. The last one is cut off by the row end; the cut-off move is dropped, and the prefix kept.
  - Measured on 2,000 rows: 5.5 characters per move, 62 moves per game on average, 185 at most.
  - The training split holds about 1.5B moves.
- **Random games:** uniformly random legal moves from the start position until the game ends or 255 moves are reached. They are generated on the fly, so the supply is unlimited.
- **Mixture:** 20% of training tokens come from random games, in the human and legal conditions alike, so the conditions see the same input distribution. The fraction is checked in the pilot. On random games, next-move prediction has the uniform legal distribution as its expected target, which gives the human condition some legality signal there; that is the Othello-GPT setting, and is stated as such.
- **Packing:** games are packed into rows of 256 tokens, each starting with the game-start token. Games are never split across rows, and row tails are padded. State carries across game boundaries inside a row, as in the character data.
- **Split:** by a hash of the move sequence, so identical games stay on one side. Random test games use separate seeds.

### Protocol

- **Optimiser:** the 20B study's AdamW settings.
- **Update support:** gap-free, with a broad mixture throughout, plus warm-start batches in the decay phase. These are the fixes from the 20B study.
- **Live evaluation during training:** temporal's failure showed only in live execution.
- **Budget:** one pass over the human training moves, about 1.5B tokens with the random games, per human or legal run. Batch size and update count are fixed after the pilot.
- **Seeds:** three per arm and condition, more if the pilot shows large seed variance. Move tokens make runs cheap enough for that.
- **Checkpoints:** kept on the 5B protocol's schedule, so per-loop readouts and probes can be added later without retraining.

## Data pipeline

- **Conversion:** replay each game with python-chess and write move tokens, the legal-move set of every position, and the metadata needed for strata (en passant available, castling rights, in check).
- **Legal-move targets:** about 35 moves per position over 1.5B positions. Precompute them as `uint16` index lists, or generate them in data-loader workers. Choose after benchmarking python-chess against the training throughput; a compiled move generator is the fallback.
- **Engine annotation:**
  - **Scope:** stage-2 training positions (sized after the pilot; a few million) and the evaluation positions.
  - **Engine:** a pinned Stockfish binary (not installed yet), MultiPV over every legal move, and a fixed node budget, so labels are deterministic.
  - **Two budgets per position.** Disagreement between the shallow and the deep evaluation marks positions where calculation changes the answer. The gap between the best and second-best move does not: a large gap can be a trivial recapture.
  - **Storage:** legal moves and centipawn or mate scores per position.
  - **Existing annotations:** [ChessBench](https://github.com/google-deepmind/searchless_chess) scores every legal move with Stockfish 16, for positions from 10M Lichess games from February 2023. Its 1.1 TB action-value set is keyed by position, without game IDs or histories. Using it means downloading those games, tokenising them and looking up each position. Check coverage and cost before annotating ourselves.

## Evaluation

**Execution:** live, token by token, as the primary setting, with the training graph alongside. Depth and hybrid are evaluated at J = 1, 2 and 4 core iterations. J beyond the trained maximum is reported separately, as extrapolation.

**Test sets:**
- held-out human games;
- held-out random games;
- an unseen generator: Stockfish self-play at a low skill level. The positions are plausible but unlike both training sources, so it tests whether the rules were learned rather than the patterns of either source.

**Legal condition:**
- per-move precision and recall of legality at threshold 0.5;
- exact-set accuracy: the $|L(s)|$ highest-scoring moves are exactly the legal set;
- binary cross-entropy.

**Engine condition:**
- regret of the chosen move, $\max_{a'} Q(s,a') - Q(s,\hat a)$;
- best-move agreement;
- rank correlation between predicted values and $Q$ over the legal moves;
- binary cross-entropy.

**Across conditions:** every condition outputs a score for all 1,968 moves, so two metrics apply to all three:
- exact-set accuracy (ranked by softmax probability in the human condition);
- regret of the top-scoring legal move.

This shows how much legality and move quality each pressure produces as a by-product. The human condition's move probabilities are its own; low-temperature decoding alone improves imitation models' play ([Zhang et al., 2024](https://arxiv.org/abs/2406.11741)), so move quality is reported for its most probable legal move.

**Strata:**
- move number;
- **history-dependent legality:** en passant available, and castling rights lost by earlier king or rook moves. These can't be read from where the pieces stand;
- in check, pinned pieces, promotions;
- **engine condition:** shallow-versus-deep disagreement, and forcing lines. The mate-in-1 and mate-in-2-by-check selection with matched controls from `experiments/interp/board_state/forcing.py` on the `interp` branch is rebuilt for move tokens.

**Learning speed:**
- linear board probes at every move token, against tokens seen, for the human and legal conditions;
- this tests whether the legal target teaches the board faster than imitation.

## Predictions and decision rules

Fixed before the main runs. Practical margins are set after the pilot.

- **Legal stage:** temporal and hybrid beat depth and transformer, with larger gaps on history-dependent strata, late moves and random games. J changes little.
- **Engine stage, depth helps:** hybrid J=4 has lower regret than hybrid J=1, and the gain is larger on high-disagreement positions than on low ones. Both are paired across positions, with the interval across seeds excluding zero.
- **Engine stage, complementarity:** the hybrid beats temporal-only on regret at matched block applications, and beats depth-only at J=4.
- **A null result** means no detected benefit at this model size, target and budget, not that chess doesn't need depth.

## Pilot

Small from-scratch runs of all four arms, one seed, on a small fraction of the budget, in the human and legal conditions. The old 5B checkpoints are character models and can't be reused.

**Pass conditions:**
1. **Correctness:** the tests below pass, and live execution equals the training graph.
2. **No saturation:** the legal task doesn't saturate within the budget. Exact-set accuracy still separates the arms on at least one test set.
3. **Throughput:** legal-move generation keeps up with training.
4. **The mixture:** 20% random games don't cost the human condition much on human test games.

Also measured:
- **Stockfish throughput** at the two node budgets;
- **the ChessBench join:** coverage and cost;
- **a short engine stage** on a few hundred thousand positions. Check best-move agreement against the equal weighting of moves in the value loss; if it lags, weight moves by rank.

## Steps

1. **Tokenizer, conversion and tests:**
   - replaying each game's SAN and writing it back from the tokens gives the same game;
   - the vocabulary covers every legal move in the data and in random games, including castling, en passant and every promotion;
   - legality targets match python-chess;
   - losses touch only positions with a next move in the same game, on the final pass;
   - live output equals the training graph.
2. **Model and trainer:**
   - token vocabulary and context length from the config;
   - untied output layer;
   - the three losses;
   - the random-game generator and the packed loader.
3. **Pilot.**
4. **Freeze the protocol:** budgets, mixture, margins, test sets.
5. **Human and legal conditions:** four arms × three seeds each.
6. **Annotation and engine stage:** four arms × three seeds.
7. **Report** next to the scripts.

## Cost

| Item | Estimate |
| --- | --- |
| Pilot | a few GPU-hours |
| Human and legal conditions, 4 arms × 3 seeds each | about $3–5 per run, $70–120 for all 24. A 5B-character run cost about $8.5, and compute per token is about the same for move tokens, so 1.5B tokens is about 30% of that. Shorter rows may use the GPU less efficiently, hence the upper end |
| Engine stage | a few GPU-hours per arm |
| Engine annotation | pending the benchmark; the earlier estimate was about 100 core-hours per 1M positions |

## Later extensions

- **The Chess-World-Model benchmark:** its input is also from/to move tokens, so our four arms fit its board-state target and its published baselines directly. That gives a "state readout" condition at 3–8M parameters, cheaply. First check that its data and code are released.
- **Board-given control:** the same targets with the current position given as input tokens. Temporal recurrence has less to reconstruct there, so its advantage should shrink while depth's remains.
- **A look-ahead target:** predicting the engine's next moves in the principal variation supervises look-ahead directly. Worth adding if the engine stage shows no benefit from loops.
- **Harder evaluations:**
  - Lichess puzzles, which record the game they come from, so histories are recoverable;
  - play against fixed Stockfish levels. Tactical accuracy and playing strength can come apart ([Miłosz et al., 2026](https://arxiv.org/abs/2608.27757)).
