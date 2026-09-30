# Move-target training: plan

> **Status:** steps 1–3 are implemented: the dataset builders and loader (`moves/build_stage1.py`, `moves/build_leela.py`, `moves/rows.py`) and training on them (`data_format='moves'`). The pilot runner is `experiments/move_pilot/pilot.py`. The pilot datasets are being built; no pilot has run.
>
> This document is the single source of truth for the `engine-policy` branch. Read [Pitfalls](#pitfalls) before implementing anything. Record every decision that changes the design under [Decisions](#decisions), with its reason; don't edit frozen items silently.

## The question

**Does the training target decide which recurrence axis pays off?**

The 20B study trained a transformer, a temporal-only, a depth-only and a hybrid model to predict the next character of Lichess games ([20B report](../experiments/long_runs/20B_recurrence/REPORT.md)):
- all three recurrent models beat the transformer, with the gains in move choice;
- the hybrid tied temporal-only, and depth added little;
- deployed, aligned temporal matched the hybrid at far fewer block applications.

**Hypothesis:** imitating players rewards tracking the board, which temporal recurrence does well, but asks for no more calculation than the players did. Depth recurrence should pay off only when the target needs calculation.

**Test:** train the same models under three targets for the next move:

| Condition | Target | What it needs | Expected to favour |
| --- | --- | --- | --- |
| **Human** (reference) | the move played | the board, and a model of the player | temporal |
| **Legal** | which moves are legal | the board and the rules | temporal |
| **Engine** | Leela's search distribution over the legal moves | the board and the result of a search, which calculation can approximate | depth, or both axes together |

## Design at a glance

- **One token per ply**, in from/to (UCI) notation. The input vocabulary and the output layer are the same 1,968 moves, so the three conditions are one language model with three targets.
- **Two stages.**
  - **Stage 1:** human and legal conditions, trained from scratch on a fixed mixture of Lichess games and random games.
  - **Stage 2:** the engine condition, on Leela Chess Zero's self-play data. It continues from **both** the legal-stage checkpoints (the primary recipe) and the human-stage checkpoints (the control), with a fresh output layer.
- **Fixed datasets.** All chess computation (replay, legal moves, packing, conversion) happens once, when the datasets are built. Training only reads fixed arrays, in one fixed order for every run.
- **Four arms,** three seeds each, at the same token budget.
- **The central test is an interaction:** do extra core iterations (J = 1 → 4) help the engine-trained models more than the human-trained ones, on the same positions? A gain from loops in one model alone doesn't show the target caused it.

## Why these choices

- **Move tokens, not characters.**
  - The architectures are unchanged; this is a tokenizer choice, like word-level tokens.
  - It removes the character cycle (move numbers, dots, spaces), so every token is a decision and each temporal memory step is one ply.
  - It makes the output layer the same for every condition.
  - It is the tokenization of the closest prior work ([related work](chess_related_work.md)): Chess-World-Model, Allie, Toshniwal et al.
- **Legal before engine.** Legality needs the board and the rules but no evaluation, so it isolates state tracking.
- **Random games in stage 1.** Human positions follow familiar patterns. Uniformly random legal games force the rules to be learned rather than the patterns.
- **Leela's data for the engine stage.**
  - It is dense and strong: every legal move gets a share of a few hundred search visits by a top network, and it's free under the Open Database License.
  - The alternatives were worse. Our own Stockfish labels would be weak at the scale we can afford. ChessBench has no game histories: its records are shuffled, even within a position, as verified on its test file. PAWN's values for every move come without search.
  - **What we give up: per-move values.** Leela stores probabilities for every move, but values only for the position, the best move and the played move. Per-move values exist only on a small Stockfish-labelled evaluation panel.
- **Fixed datasets.** Every run sees identical data, and no chess code runs during training.

## Specification

### Tokens and model

- **Vocabulary:**
  - the 1,968 moves ChessBench uses: every from/to pair a queen or knight could make on an empty board (1,792), plus the four promotions for both colours (176);
  - castling is the king's move (`e1g1`), and en passant is the pawn's move (`e5d6`);
  - plus a game-start token and padding.
- **Arms:** transformer, temporal-only, depth-only and hybrid, with the 20B layout: 8 layers, width 512; prelude 1, T-buffer 1, core 4, T-source 1, coda 1.
- **Context:** 256 tokens.
  - Every Lichess game fits: a 1,023-character row holds at most 255 plies.
  - Random games are capped at 255 plies.
  - Leela games longer than 255 plies are truncated (5.4% of games, 2.7% of positions).
- **Output layer:**
  - untied from the input embedding;
  - one per condition; stage 2 starts a fresh one;
  - a 512 × 1,968 linear layer, identical in every arm.

### Datasets

**One format for every dataset:** fixed rows of packed games, plus a sparse move list for each position.

| Array | Contents | Type |
| --- | --- | --- |
| `tokens` | rows of 257 tokens: game start, the game's plies, the next game, …, padding | uint16 |
| `move_counts` | for each of a row's 256 positions, how many moves its target lists (0: no target) | uint8 |
| `moves` | the listed move ids, row after row, position after position | uint16 |
| `row_offsets` | where each row's moves start | int64 |
| `weights` (stage 2) | Leela's probability for each listed move | float16 |
| `values` (stage 2) | per position: Leela's position value (win minus loss, draw probability) and visit count | float16 |

**Reading and targets:**
- Reading row *i* is pure slicing, with no chess code at training time.
- The inputs are the first 256 tokens, and position *t*'s next-ply target is token *t*+1. The human condition needs nothing more.
- The legal condition reads the move lists as legal sets; the engine condition reads them with `weights`.
- A position has a target when its next ply exists in the same game.

**Packing:**
- Games start with the game-start token, never cross rows, and are packed greedily; row tails are padded.
- State carries across games within a row.
- Human and random games may share a row.
- Evaluation files hold **one game per row**, so no game reads another game's state.

**Splits:** every source is split by a hash of each game's plies into train, dev and test, before packing.

**Order:**
- Every run reads the same training rows in the same order; seeds vary only the initial weights.
- A stage that runs longer than its dataset reads later passes in a new fixed permutation of the same rows.

#### Stage 1: Lichess and random games, with legal moves

1. **Lichess games:** `chess_8M_v1` (Karvonen's release), replayed into ply tokens with each row's cut-off final ply dropped. The training split holds about 1.5B plies.
2. **Random games:** uniformly random legal games from the start position, until the game ends or 255 plies. They are generated once from a recorded seed, in the amount needed for **20% of training positions**. Random dev and test games use separate seeds and are deduplicated against training.
3. **Shuffle and pack** all training games, human and random, in one fixed order.
4. **Legal moves:** each position's legal moves, computed once with python-chess.
5. **Evaluation files:** human dev and test, random dev and test.
6. **Sizes:**
   - a **0.5B-position first version** for the pilot (~30 GB, mostly the legal-move lists);
   - then the **full ~1.9B** (one pass over the Lichess games plus 20% random games; ~120 GB).

#### Stage 2: Leela data

1. **Select** a fixed, recorded range of Leela test80 archives (hourly; about 50,800 games and 5.6M positions each). Record their names and hashes.
2. **Convert and verify:**
   - replay every game from the start position;
   - skip Chess960 games (about 4%) and any game that fails a check;
   - translate castling;
   - truncate at 255 plies;
   - keep each position's legal moves with Leela's probabilities, the position value and the visit count.
3. **Train on the probabilities raw,** as Leela's own training does. Visit counts are stored, so filtering can be added later if play shows it is needed.
4. **Split, shuffle and pack** as in stage 1.
5. **Evaluation files:** Leela dev and test.
6. **Sizes:** **100M positions for the pilot** (~18 archives, ~13 GB), **400M for the main runs** (~72 archives, ~52 GB).

#### The evaluation panel

A few thousand positions from Leela test and from human test, labelled by our Stockfish teacher with a value for every legal move, at two budgets. It is the only source of per-move values, and it scores regret for every condition. See [Evaluation teacher](#evaluation-teacher).

### Losses

All losses apply at positions with a target, on the final pass only.

- **Human:** softmax cross-entropy with the move played.
- **Legal:** one sigmoid per move, binary cross-entropy with target 1 for legal and 0 for illegal.
  - Average over the 1,968 moves per position, then over positions.
  - Report the loss on legal and on illegal moves separately, against a constant-rate baseline: about 1.8% of moves are legal, so a constant predictor already scores about 0.089 nats.
- **Engine:** softmax over the legal moves only, with cross-entropy against Leela's probabilities.
- **Loss weighting** needs a pilot result showing it helps, and is frozen before the main runs.

### Evaluation teacher

Stockfish labels the evaluation panel, with values from the mover's perspective:
- **Conversion:** $Q = \sigma(0.00368208\,\mathrm{cp})$, with centipawns capped at ±1,500, so non-mate values lie in [0.004, 0.996].
- **Mates:** mate in $n$ for the mover maps to $0.997 + 0.002/n$, and being mated in $n$ to $0.003 - 0.002/n$. Every mate outranks every non-mate value, and shorter mates are strictly better.
- **History:** each position is sent with the moves played, so repetitions and the 50-move rule count. A move after which the opponent could claim a draw is worth at most 0.5.
- **Protocol:**
  - a pinned binary, its SHA-256 recorded;
  - 1 thread, a fixed hash size, no tablebases;
  - **one search per legal move**, of the position after it, with a new game before each search;
  - two node budgets, shallow and deep; disagreement between them marks positions where calculation changes the answer;
  - store the score, its type, whether it is exact or a bound, the depth and the nodes;
  - re-annotating in a different order must give identical labels.

### Training protocol

- **Optimiser:** the 20B study's AdamW settings, with the same small learning-rate check for each objective on the transformer, applied to every arm.
- **Update support:** gap-free and broad throughout; deepen by moving the centre, never by concentrating the mass (the 20B report's recommendation). The concrete schedule is set in the pilot.
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
- **Human, random and Leela test files.** Random games are in the random generator's training distribution: unfamiliar trajectories, not a distribution shift.
- **An unseen generator:** Stockfish self-play at a low skill level. This carries the distribution-shift claim for legality.
- **The evaluation panel,** from Leela test and human test positions.
- **Targeted rule sets:** en passant, castling rights lost by moves that were later undone, pins.

### Metrics

Each metric is used only where it means something.

| | Human | Legal | Engine |
| --- | --- | --- | --- |
| Legality: probability on legal moves and legality of the top move (human); precision and recall at 0.5 and exact-set accuracy (legal) | ✓ | ✓ | — the head only saw legal moves |
| Errors on **almost-legal moves**: illegal only because of a pin, a check or a blocked line | ✓ | ✓ | — |
| Regret of the top-scoring legal move on the panel, $\max_{a'}Q - Q(\hat a)$ | ✓ | — every legal move scores alike in an ideal head | ✓ |
| Best-move agreement on the panel (near-ties count) | ✓ | — | ✓ |
| Top-move agreement with Leela's best move; KL from Leela's distribution (a diagnostic) | — | — | ✓ |

- **Legality** is always judged by the rules at evaluation, never by a head that didn't learn it.
- **Move quality** is taken from each model's top-scoring legal move, never by sampling. Imitation models play better at low temperature ([Zhang et al., 2024](https://arxiv.org/abs/2406.11741)), and an engine-trained model keeps Leela's one-visit floor.

**Strata:**
- ply;
- history-dependent legality;
- in check, promotions;
- shallow–deep disagreement on the panel;
- forcing lines: mate in 1 and mate in 2 by check, with matched controls. Rebuilt from `experiments/interp/board_state/forcing.py` on the `interp` branch.

### Primary analyses

Declared now; margins and thresholds are set on the development sets before the main runs.

1. **Learning speed (stage 1):** tokens seen until each arm reaches a fixed threshold on its own condition's metric: exact-set accuracy for legal, next-ply loss for human. Arms are compared within a condition.
   - Exact-set accuracy is not read from human heads: they score unpopular legal moves low whatever they know about the board.
   - Whether legality targets teach the board faster than imitation is judged by board probes only.
   - Legality on human games will likely saturate near 100% for every arm, as board state does in [Chess-World-Model](https://arxiv.org/abs/2605.30100).
   - So speed, the almost-legal moves and the unseen generator carry the comparison, not final in-distribution accuracy.
2. **Target × loops (the central test):** for depth and hybrid, the regret reduction from J=1 to J=4 in the engine continuation, minus the same reduction in the human-trained model, on the same panel positions.
   - Also whether it is larger on high-disagreement positions than on low ones.
   - Both are tested as interactions, not as two separate significance tests, and reported separately for Leela and human positions.
   - **This compares recipes.** The engine model has had a second stage, a fresh head and a softmax over the legal moves; the human model has not. So the result is recipe × loops.
   - **Matched human continuation** (decided after the pilot): continue the human trunks on Lichess games for the same stage-2 budget, with a fresh head and a softmax over the legal moves. Then only the target and its games differ, and the claim can be about the target.
3. **Arms in the engine stage,** at matched data:
   - hybrid vs temporal at J=1, which matches block applications;
   - hybrid vs depth at J=4, which also matches;
   - hybrid J=4 vs temporal, reported with measured latency, prefill and cache bytes. This is not matched compute.

**Statistics:**
- **the estimate** is the mean over seeds of each seed's paired contrast (the same seed index and panel positions in both conditions);
- effects for each seed shown individually;
- intervals within a seed from bootstrapping games, with each game's positions resampled together. They are conditional on the trained checkpoints and say nothing about seed variation;
- a result counts only if all three seeds agree in sign and the mean exceeds the practical margin, not merely if an interval excludes zero.

**Diagnostic, not primary:** an engine stage continued from legal-stage checkpoints matched on exact-set accuracy, with the matching rule fixed in advance.

## What the design can and cannot show

**Can show:**
- how whole training recipes compare, at matched data;
- whether the engine recipe increases the benefit of loops relative to the human recipe; attributing it to the target needs the matched human continuation;
- whether legality targets teach the board faster, by board probes.

**Cannot show, without more evidence:**
- compute-optimality: temporal-only has no 20-block configuration;
- that loops perform **search** or **look-ahead**: that needs mechanism evidence;
- that "chess doesn't need depth": a null means no detected benefit at this size, target and budget;
- that agreeing with Leela is playing well: move quality is judged on the Stockfish panel;
- results for human positions from the engine stage alone: its inputs are Leela's games, so human-position results are a transfer measurement;
- that the theory decides this: [Merrill et al. (2024)](https://arxiv.org/abs/2404.08819) is about exact tracking over arbitrarily long sequences, which motivates temporal recurrence but predicts nothing for 256 tokens.

## Pitfalls

For anyone implementing or analysing on this branch:

1. **The training graph is not live execution** for temporal and hybrid. Never change recurrence semantics to make them agree. Correctness means three things:
   - cached live execution equals the slow reference;
   - the training graph iterated to its fixed point equals live execution;
   - depth-only with `depth_specialized` caches equals its training cell $(0, J-1)$.
2. **Heads answer only their own question.** An engine head says nothing about legality; a legality head's ranking of legal moves says nothing about quality.
3. **Leela's encoding has traps, all checked by replay in the converter:**
   - each rank's files are stored in reverse bit order;
   - Black's moves are rank-mirrored;
   - castling is written king-takes-own-rook (`e1h1`);
   - a plain pawn move to the last rank is a knight promotion;
   - archives also contain a `LICENSE` text file.
4. **Leela's probabilities are search effort, not values.** A move's share reflects exploration as well as quality. Never read them as calibrated values or sample a student at temperature 1 to judge it.
5. **$Q$ is from the mover's perspective** on the evaluation panel. Scores of the position after a move come from the opponent's side and must be read from the mover's. Test with winning positions for White to move and for Black to move.
6. **A small legality loss doesn't mean the rules are learned.** Compare with the constant baseline, and read precision, recall and the almost-legal errors.
7. **"Moves" means plies** everywhere in code and results. Report human plies, random plies, supervised positions and row tokens separately.
8. **No tuning or checkpoint selection on final test sets.**
9. **Recipes differ on purpose** (warm-start batches for temporal only). Don't equalise them silently, and don't describe the results as pure architecture effects.
10. **Repository hygiene:**
    - data, `.venv` and results live only in the main checkout;
    - never commit symlinks or anything under a `results` path; tracked `results` links destroyed the 20B checkpoints;
    - stage explicit paths, never `git add -A`.

## Pilot

- **Data:** build the 0.5B stage-1 dataset and the 100M-position Leela dataset.
- **Stage 1:** small from-scratch runs of all four arms, human and legal, one seed, plus a second seed of one arm to estimate seed variance.
- **Stage 2:** a short engine stage (about 50M Leela positions) continued from both the legal and the human checkpoints.
- **Runner:** `experiments/move_pilot/pilot.py` (`list`, `run <name>`), with the defaults logged under Decisions.
- The old 5B checkpoints are character models and can't be reused.

**Pass criteria** (none requires an arm to win):
1. all tests pass;
2. dataset builds and training throughput are adequate, with their cost measured;
3. development sets have room to improve. If every arm sits at the ceiling, use harder sets or a smaller scale, both declared in advance;
4. learning curves and seed variance are measured;
5. the packing-order effect is measured.

## Steps

1. ✅ **Tokenizer, parsing, teacher and tests.**
2. ✅ **Model and trainer:** vocabulary and context from the config, untied output layers, the human and legal losses, live evaluation, continued runs.
3. ✅ **Fixed datasets and one loader:**
   - the stage-1 builder (games, random games, packing, legal moves);
   - the Leela converter (verified replay, castling, truncation);
   - the engine loss;
   - one row-indexed loader for both stages.
4. **Pilot** without the evaluation panel, then fix the training decisions below.
5. **Evaluation panel** (deferred until a trained model plays chess): pin Stockfish; label a development panel and set node budgets and practical margins on it; check that labels reproduce. The test panel is labelled only for the frozen analysis.
6. **Stage 1:** 4 arms × 3 seeds, human and legal.
7. **Stage 2:** both continuations, 4 arms × 3 seeds each.
8. **Report** next to the scripts.

## Decisions

**Open until the pilot:**
- batch size and update count;
- the update-support schedule;
- learning rate for each objective;
- loss weighting;
- the Leela archive range;
- the learning-speed thresholds;
- whether to run the matched human continuation.

**Open until the development panel:**
- panel size and node budgets;
- practical margins.

**Log:** (date — decision — reason)

- **2026-09-30 — Rows and games** *(superseded by the fixed datasets, below)*.
  - A human game that doesn't fit starts the next row of the same batch; the one left at the end of a batch is dropped.
  - A random game is truncated to the space left in its row.
  - Reason: less padding without splitting games. Measured row fill: 0.85 on human games, 0.88 with 20% random rows.
- **2026-09-30 — Moves that end the game by rule are labelled without search.**
  - Mate is 0.999; a draw by rule (stalemate, insufficient material, the 75-move rule, fivefold repetition) is 0.5.
  - Reason: the engine has nothing to search there, and a mated position has no mate score to read from the mover's side.
- **2026-09-30 — A move after which the opponent could claim a draw is worth at most 0.5.**
  - This covers threefold repetition and the 50-move rule.
  - Reason: the opponent claims when that beats their own prospects. This is what makes the teacher's values depend on history.
- **2026-09-30 — A new game before every per-move search, not just every position.**
  - Reason: otherwise the hash from earlier moves' searches gives later moves more work, and labels depend on move order.
- **2026-09-30 — Label cache** *(its code was deleted with the old annotation script; reimplement with the panel)*.
  - Labels of positions up to 16 plies deep are cached by their history key: the FEN plus the positions since the last capture or pawn move.
  - Reason: openings repeat across games, and the key captures everything that can change a deterministic engine's labels.
- **2026-09-30 — Score bounds are kept, not rejected.**
  - Each label records whether the engine reported an exact score or a bound, flipped to the mover's side. The dataset manifest counts bounded labels per budget.
  - Reason: with a node limit the final score is normally exact; the counts show whether bounds matter before deciding to drop them.
- **2026-09-30 — Best-move agreement tolerance** *(its metric was deleted with the Stockfish-value target; reimplement with the panel)*.
  - The chosen move agrees if its $Q$ is within 0.01 of the best (`NEAR_TIE`).
  - Reason: the plan counts near-ties as agreement; 0.01 is one percentage point of win probability.
- **2026-09-30 — Evaluation during training.**
  - Dev and random games are evaluated one game per row, so no game reads another game's state.
  - Recurrent runs also decode the same dev games live with the cached implementation: temporal at J=1; depth and hybrid at J=1, 2, 4 with `depth_specialized` caches. Results are reported as `live_J{n}_val_*` next to the training-graph `val_*`.
  - Reason: live execution is primary, and temporal's 20B failure showed only live. Evaluating the same games makes the two paired.
- **2026-09-30 — First measurements** (MPS laptop, one process):
  - conversion: about 40,000 rows in 17 s with 8 workers, so about an hour for all 8.3M rows;
  - no parse failures;
  - longest game in that 40,000-row sample: 187 plies (119,000 games);
  - legal-move targets are built at about 28,000 positions/s per process.
- **2026-09-30 — Training batches built by worker processes, depending only on (seed, rank, batch index)** *(superseded by the fixed datasets; code deleted)*.
  - Measured with legal targets and 20% random rows: 30,000 supervised positions/s in-process, 114,000 with 4 workers, 230,000 with 8.
- **2026-09-30 — The engine stage uses Leela's test80 self-play data, as search-policy distillation.**
  - Reason: a dense target from a strong search, free and open. The alternatives were weaker; see [Why these choices](#why-these-choices).
  - Measured on one archive (`training-run1-test80-20240401-0017`, 1.1 GB):
    - 50,807 games (one per file), 5.6M positions;
    - 96% of games start at the standard position, the rest are Chess960;
    - every standard game replays with matching boards, legal sets and played moves;
    - a median of 467 visits per position;
    - the top move's probability has a median of 0.55; 42% of the probability sits on searched alternatives and 3.5% on the one-visit floor;
    - Leela's stored best move is the top-probability move in 99.3% of positions;
    - 5.4% of games exceed 255 plies.
  - Per-move values are dropped from the engine stage and come only from the Stockfish evaluation panel.
- **2026-09-30 — Fixed datasets replace on-the-fly batch building.**
  - All chess computation happens when datasets are built. Training reads fixed rows in one order for every arm and seed; seeds vary only initialisation.
  - Human and random games may share a row.
  - Sizes: stage 1 at 0.5B positions first, then the full ~1.9B; stage 2 at 100M positions for the pilot and 400M for the main runs.
  - Reason: identical data for every run, a simple loader with no CPU bottleneck, and the same format for every dataset.
- **2026-09-30 — How the fixed datasets are built and read** (step 3).
  - Builders turn each source into a stream of games with a move list per position, and one packer writes any such stream into rows. Stage 1 packs straight from memory. Leela games are first converted per archive into game stores, then shuffled and packed; the stores are deleted afterwards.
  - Games longer than the context are truncated, in both builders (at most 255 plies). Random games are generated up to that length.
  - Training order: pass 0 reads rows in stored order (shuffled at build time); pass *k* uses a fixed permutation seeded by *k*. Micro-batch *m* takes the next `batch_size` rows. So every run reads the same rows, and exact resume needs only the step count.
  - Evaluation during training reads the first rows of each evaluation split, one game per row. Live evaluation uses the first rows of the first evaluation split, so it is paired with the training-graph numbers.
  - Engine target options *(deleted, below)*: `policy_floor` dropped moves with at most ~one visit, and `policy_min_visits` dropped positions with smaller searches.
  - Reason: one format and one loader for both stages, with all chess code at build time.
- **2026-09-30 — The evaluation panel is deferred until a trained model plays chess.**
  - The pilot judges stage 1 by the legality metrics, and stage 2 by top-move agreement with Leela and KL divergence.
  - Regret, best-move agreement against Stockfish, and the shallow–deep strata wait for the panel. So does the central target × loops test, which is scored on it.
  - Reason: labelling a panel is not worth the effort before a model plays well enough for regret to mean something.
- **2026-09-30 — First fixed-dataset build on real data** (2M training positions, 2,000 games per evaluation file, 6 workers):
  - 85 s end to end, about 59 bytes per position on disk (so ~30 GB for the 0.5B version);
  - no parse failures and no truncated Lichess games; random games average 238 plies; rows are 82% full;
  - reading legal targets back: 25M positions/s in one process, against 30,000/s computed on the fly. The loader needs no workers.
- **2026-09-30 — The on-the-fly pipeline is deleted.**
  - Deleted: the on-the-fly loader and conversion (`moves/data.py`, `moves/prepare.py`); the trainer's options for it (`random_game_fraction`, `loader_workers`, `value_budget`); the Stockfish-value dataset and target (`moves/annotate.py`, `ValueTargets`); the archive `fetch` helper (a `curl` range replaces it).
  - Kept: `moves/teacher.py` and `moves/values.py`, the tested Stockfish labelling protocol, for the panel.
  - The only move format is the fixed dataset, so `data_format='moves'` now means it.
  - Reason: the fixed datasets are measured to be fast and affordable, so keeping the old pipeline "as a fallback" only added code and trainer branches. It stays in git history.
- **2026-09-30 — Leela conversion speed.**
  - Measured: about 2,100 positions/s per process, so about 1.7 hours per 100M positions on 8 cores.
  - Most of the time goes to decoding every listed move through python-chess. Lookup tables could make it two- to three-fold faster if the 400M build needs it.
- **2026-09-30 — Pre-commit review of step 3.**
  - **Leela targets are used raw; the target options are deleted.** Dropping one-visit moves took legal moves out of the softmax, so the model could choose them at no cost; dropping positions broke the loss pooling and could empty a batch; and dev metrics were measured against the modified targets. Leela trains on the raw targets itself, and the pilot couldn't judge the options without the panel.
  - **Learning speed uses each condition's own metric,** and board-learning claims rest on probes. Reading exact-set accuracy from a human head breaks Pitfall 2.
  - **The central test is recipe × loops** until a matched human continuation is run.
  - **Panel budgets and margins are set on a development panel,** not in the training pilot.
  - **Seed-level inference:** the mean of paired per-seed contrasts, with sign agreement across seeds.
  - Evaluation during training runs batch by batch; it had sent `eval_iters × batch_size` training rows through one forward pass. The Leela converter skips records with no probability mass or an out-of-range played move, and empty game stores load.
- **2026-09-30 — Pilot defaults** (`experiments/move_pilot/pilot.py`).
  - **Updates:** 400 rows of 256 tokens (about 100k tokens, as the 20B study's 100 × 1,023 characters), in 20 micro-batches of 20 rows. One pass over the stage-1 dataset per run.
  - **Optimiser:** the 20B settings (AdamW 3e-4 to 3e-5, WSD), with 3% warmup and the last 10% decaying. The learning-rate check runs the transformer at 1e-4, 3e-4 and 1e-3 for a fifth of a run, per objective; the engine check continues from the legal transformer.
  - **Update support {0, 1, 2, 3}:** the probabilities of max(U_T, U_D) = 0, 1, 2, 3 are (0.25, 0.35, 0.25, 0.15), then (0.15, 0.30, 0.30, 0.25) from 25% of the run, then (0.10, 0.25, 0.35, 0.30) from 50%. Hybrid puts 80% of each count on the diagonal, as in the 20B study. Stage 2 keeps the final mixture throughout.
  - **Temporal** gets warm-start batches (a quarter of each update's micro-batches) in the decay phase only.
  - **Seeds** change only the initial weights; the second seed runs legal hybrid.
  - Reason: the 20B recipe where it worked, and the 20B report's recommendations (a gap-free, broad support deepened by moving the centre; warm starts in the decay) where it didn't. `build_update_probability_matrix` now accepts any increasing support starting at zero.


## Cost

Estimated; to be re-measured in the pilot:
- **Stage-1 runs:** about $3–6 each at about 1.9B tokens, so $70–150 for all 24.
- **Stage-2 runs:** scale with the 400M-position dataset and its passes.
- **Dataset builds:**
  - stage-1 legal moves: about 18 core-hours for 1.9B positions;
  - Leela: ~2 hours of download for 100M positions, ~8 hours for 400M, at the ~3 MB/s we saw, plus conversion.
- **Storage:** about 120 GB for the full stage-1 dataset, and ~52 GB for 400M Leela positions.
- **Evaluation panel:** a few thousand positions at two Stockfish budgets.

## Later extensions

- **Per-move values at scale:** our own Stockfish labels on stage-2 positions, if the policy target proves too coarse.
- **The Chess-World-Model benchmark** with its own interface (board target, auxiliary fields, protocol), for a direct comparison with its published baselines.
- **Board-given control:** the same targets with the current position as input. It tests whether engine-stage advantages come from reconstructing the board.
- **A look-ahead target:** predict the engine's principal variation, if loops show no benefit.
- **Harder evaluations:** Lichess puzzles with their game histories, and play against fixed Stockfish levels. Tactical accuracy and playing strength can come apart ([Miłosz et al., 2026](https://arxiv.org/abs/2608.27757)).
