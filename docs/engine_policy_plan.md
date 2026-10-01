# Move-target training: plan

> **Status:** the code is implemented: dataset builders and loader (`moves/`), training on move datasets (`data_format='moves'`), and the pilot runner (`experiments/move_pilot/pilot.py`). The stage-1 pilot dataset is built; the Leela archives are downloading. No pilot has run.
>
> This is the single source of truth for the `engine-policy` branch. Read [Pitfalls](#pitfalls) before implementing anything, and record every design change under [Decisions](#decisions) with its reason.

## The question

**Does the training target decide which recurrence axis pays off?**

The 20B study trained a transformer, a temporal-only, a depth-only and a hybrid model to predict the next character of Lichess games ([20B report](../experiments/long_runs/20B_recurrence/REPORT.md)). All three recurrent models beat the transformer; the hybrid tied temporal-only, and depth added little. Imitating players asks for board tracking, which temporal recurrence does well, but for no more calculation than the players did.

**Hypothesis:** the kind of task decides the axis. A task that needs the board but no calculation favours temporal recurrence; a task that needs calculation favours depth, or both axes together. Teacher strength alone doesn't make a task deep: a model can also learn a search's outputs as a direct mapping. So this is the hypothesis under test, not an assumption.

**Test:** train the same four models on a shallow target, then continue them on a deep one.

| Stage | Target | What it needs | Expected to favour |
| --- | --- | --- | --- |
| **1. Legal** (shallow) | which moves are legal, one sigmoid per move | the board and the rules | temporal |
| **2. Engine** (deep) | Leela's search distribution over the legal moves | the board and the result of a search, which calculation can approximate | depth, or both axes |

The targets are the change that matters. Everything else is chosen for efficiency and needs no ablation:
- **One token per ply** (UCI from/to), so every token is a decision and each temporal step is one ply. It is the tokenization of the closest prior work ([related work](chess_related_work.md)).
- **Dense targets:** every position is supervised over all 1,968 moves, not only the one played.
- **Fixed datasets:** all chess computation happens once, at build time; every run reads the same rows in the same order.

## Design

- **Arms:** transformer, temporal-only, depth-only and hybrid, with the 20B layout: 8 layers, width 512; prelude 1, T-buffer 1, core 4, T-source 1, coda 1.
- **Stage 1 (legal):** from scratch, on Lichess games mixed with uniformly random games. Random games make the rules, not familiar patterns, carry the target.
- **Stage 2 (engine):** continues each stage-1 model on Leela Chess Zero's self-play games, with a fresh output layer.
- **Seeds:** three per arm and stage in the main runs; seeds change only the initial weights.
- **Controls wait for a signal** (see [Controls](#controls-once-there-is-a-signal)).

### Tokens and model

- **Vocabulary:** ChessBench's 1,968 moves (every from/to pair a queen or knight could make on an empty board, plus the promotions), a game-start token and padding. Castling is the king's move (`e1g1`); en passant is the pawn's move.
- **Context:** 256 tokens. Every Lichess game fits; random games are capped at 255 plies; Leela games longer than 255 plies are truncated (5.4% of games, 2.7% of positions).
- **Output layer:** untied from the input embedding, 512 × 1,968, identical in every arm; stage 2 starts a fresh one.

### Datasets

**One format for both stages:** fixed rows of packed games, with each position's legal moves listed.

| Array | Contents | Type |
| --- | --- | --- |
| `tokens` | rows of 257 tokens: game start, the game's plies, the next game, …, padding | uint16 |
| `legal_counts` | legal moves listed at each of a row's 256 positions (0: no target) | uint8 |
| `moves` | the listed move ids, row after row | uint16 |
| `rows` | where each row's moves start | int64 |
| `weights` (stage 2) | Leela's probability for each listed move | float16 |

- **Targets:** position *t*'s next ply is token *t*+1; the legal target reads the move lists; the engine target reads them with `weights`.
- **Packing:** games start with the game-start token, never cross rows, and are packed greedily. Evaluation files hold one game per row.
- **Splits:** by a hash of each game's plies into train, dev and test, before packing.
- **Order:** pass 0 reads the stored (build-time shuffled) order; later passes use a fixed permutation seeded by the pass number.

**Stage 1** (`moves/build_stage1.py`):
- Lichess games from `chess_8M_v1` (Karvonen's release), replayed into plies, plus **20% of positions** from uniformly random legal games generated from a recorded seed. Random dev and test games are deduplicated against training.
- Evaluation files: human dev and test, random dev and test.
- **Pilot version built:** 500M positions (400M human, 100M random) in 2.38M rows, 82% full, 30 GB; no parse failures; 55 minutes on 8 workers. The full version is ~1.9B positions (~120 GB).

**Stage 2** (`moves/build_leela.py`):
- A recorded range of Leela test80 archives, their names and hashes kept in the manifest. Every game is replayed and verified; Chess960 games (4%) and games that fail a check are skipped.
- **Targets are Leela's probabilities, raw,** as Leela's own training uses them. Leela's values and visit counts are not kept; the recorded archives can be re-read if an analysis needs them.
- Evaluation files: Leela dev and test.
- **Sizes:** 100M positions for the pilot (18 archives, 1.1–1.8 GB each), 400M for the main runs.

### Losses

All losses apply at positions with a target, on the final pass only.

- **Legal:** binary cross-entropy per move (1 legal, 0 illegal), averaged over the 1,968 moves, then over positions.
- **Engine:** softmax over the legal moves only, cross-entropy against Leela's probabilities.
- **Played** (implemented, used only by the controls): softmax cross-entropy with the next ply.

### Training protocol

- **Updates:** 400 rows of 256 tokens (~100k tokens, as the 20B study's 100 × 1,023 characters). The micro-batch size (50 by default, measured) only trades speed for memory; evaluation always covers 1,000 dev games, 100 of them live.
- **Optimiser:** AdamW as in the 20B study, warmup–stable–decay (3% warmup, the last 10% decaying to a tenth of the peak). Every arm trains at a peak learning rate of **1e-3** in both stages (see Decisions: short checks misjudged the recurrent arms).
- **Update support {0, 1, 2, 3}, gap-free and broad throughout; deepen by moving the centre, never by concentrating the mass** (20B report). The probabilities of max(U_T, U_D) = 0, 1, 2, 3 are (0.25, 0.35, 0.25, 0.15), then (0.15, 0.30, 0.30, 0.25) from 25% of the run, then (0.10, 0.25, 0.35, 0.30) from 50%. Hybrid puts 80% of each count on the diagonal. Stage 2 keeps the final mixture.
- **Temporal-only gets warm-start batches** (a quarter of each update's micro-batches) in the decay phase, as the 20B report recommends. This is a recipe difference, stated as one.
- **Evaluation every 200 updates,** including live decoding. Measured on MPS at 1–3% of training time; evaluations log their seconds, and GPU time spent evaluating stays under 5%.

## Evaluation

### Execution

- **Live execution is primary:** the cached implementation, tested against the slow reference ([inference contract](INFERENCE_CONTRACT.md)), with `depth_specialized` caches for depth-only and hybrid. Depth and hybrid run at J = 1, 2 and 4; J > 4 is extrapolation.
- **Training-graph results** are reported alongside, and paired with live results on the same dev games (`{split}_live_subset_*`). Their gap is a measured property, not an error.

### Test sets

- **Development sets** for the pilot, thresholds and margins; the **final test sets** stay untouched until the frozen analysis.
- Human, random and Leela test files. Random games are in the training distribution (unfamiliar trajectories, not a shift).
- **An unseen generator** (Stockfish self-play at a low skill level) for distribution shift in legality.
- **The evaluation panel** (step 5): a few thousand Leela and human test positions, with a Stockfish value for every legal move at a shallow and a deep budget. It is the only source of per-move values.
- **Targeted rule sets:** en passant, castling rights lost to earlier moves, pins.

### Metrics

| Stage | Metrics |
| --- | --- |
| **Legal** | legal and illegal loss against the best constant predictor; precision and recall at 0.5; **exact set** (the moves above 0.5 are exactly the legal set); separation (every legal move scores above every illegal one, a ranking diagnostic); errors on almost-legal moves (illegal only because of a pin, a check or a blocked line) |
| **Engine** | KL from Leela's distribution and top-move agreement with Leela (diagnostics); on the panel, the regret of the top-scoring legal move, $\max_{a'}Q - Q(\hat a)$, and best-move agreement (within 0.01 of the best $Q$) |

- **Legality** is judged by the rules, never by a head that didn't learn it. **Move quality** is the top-scoring legal move, never a sample.
- **Strata:** ply; history-dependent legality; in check; promotions; shallow–deep disagreement on the panel; forcing lines (mate in 1 and 2 by check, with matched controls, from `experiments/interp/board_state/forcing.py` on the `interp` branch).

### Evaluation teacher (for the panel)

Stockfish values from the mover's perspective (`moves/teacher.py`, `moves/values.py`, tested):
- $Q = \sigma(0.00368208\,\mathrm{cp})$, centipawns capped at ±1,500; mate in $n$ maps to $0.997 + 0.002/n$, being mated to $0.003 - 0.002/n$; moves that end the game by rule are labelled without search (mate 0.999, draws 0.5).
- Each position is sent with its history, so repetitions and the 50-move rule count; a move after which the opponent could claim a draw is worth at most 0.5.
- A pinned binary with its SHA-256; 1 thread, fixed hash, no tablebases; one search per legal move, with a new game before each, so labels don't depend on move order; score bounds are recorded, not rejected.

### Primary analyses

Margins and thresholds are set on the development sets before the main runs.

1. **Arms at the shallow end (stage 1):** tokens until each arm reaches a fixed exact-set accuracy, and the benefit of loops (J = 1 → 4) on legality. Expected: temporal at least as good as depth and hybrid, and little benefit from loops. Legality on human games will likely saturate near 100%, so learning speed, almost-legal moves and the unseen generator carry the comparison.
2. **Arms at the deep end (stage 2):** the benefit of loops on KL and, with the panel, on regret, also split by shallow–deep disagreement; hybrid vs temporal at J = 1 and hybrid vs depth at J = 4, which match block applications; hybrid J = 4 vs temporal with measured latency, which does not.
3. **The central comparison:** whether the ranking of the arms and the benefit of loops change from the shallow end to the deep end. Each stage is measured on its own metrics, so the comparison is of patterns (signs, rankings, relative loop gains), not a difference on one scale.

**Statistics:** each seed's effect is shown; the estimate is the mean of per-seed paired contrasts; intervals within a seed come from bootstrapping games; a result counts only if all three seeds agree in sign and the mean exceeds the practical margin.

## What the design can and cannot show

**Can show:** whether loops and axes pay off differently after a shallow target and after a deep one, in the same models.

**Cannot show, without more evidence:**
- that the target alone caused a difference: stage 2 also adds training and changes the games (Lichess and random to Leela). The same-rows control addresses this;
- compute-optimality: temporal-only has no 20-block configuration;
- that loops perform search or look-ahead: that needs mechanism evidence;
- that "chess doesn't need depth": a null means no detected benefit at this size, target and budget;
- that agreeing with Leela is playing well: move quality is judged on the panel.

## Controls, once there is a signal

- **Same-rows target control:** continue each legal model on the stage-2 Leela rows with the legal target, the same budget and a fresh head. Only the target differs from the engine stage. (The loader needs one change to allow the legal target on engine datasets.)
- **The played condition:** imitation of the next ply, as a reference with regret on the panel as a common scale. On the Leela rows, played vs the search distribution separates hard from soft supervision.
- **Packing sensitivity:** training rows carry earlier unrelated games in context, while evaluation shows each game alone. Score dev games with and without an unrelated game before them.
- More seeds where variance is large.

## Pitfalls

1. **The training graph is not live execution** for temporal and hybrid. Never change recurrence semantics to make them agree. Correctness means: cached live execution equals the slow reference; the training graph iterated to its fixed point equals live execution; depth-only with `depth_specialized` caches equals its training cell $(0, J-1)$.
2. **Heads answer only their own question.** An engine head says nothing about legality; a legality head's ranking of legal moves says nothing about quality.
3. **Leela's encoding has traps,** all checked by replay in the converter: each rank's files are in reverse bit order; Black's moves are rank-mirrored; castling is written king-takes-own-rook (`e1h1`); a plain pawn move to the last rank is a knight promotion; archives contain a `LICENSE` file.
4. **Leela's probabilities are search effort, not values.** Never read them as calibrated values, or judge a student by sampling at temperature 1.
5. **$Q$ is from the mover's perspective** on the panel; scores of the position after a move must be read from the mover's side.
6. **A small legality loss doesn't mean the rules are learned.** Compare with the constant baseline and read exact set, precision, recall and the almost-legal errors. The legal head starts far above the baseline (no output bias), so early curves mostly show it learning the offset.
7. **"Moves" means plies** in code and results; report human plies, random plies, supervised positions and row tokens separately.
8. **No tuning or checkpoint selection on final test sets.**
9. **Recipes differ on purpose** (warm starts for temporal only); don't equalise them silently or call the results pure architecture effects.

## Pilot

- **Runs** (`experiments/move_pilot/pilot.py list`): the learning-rate check (`lr_legal_*`, `lr_engine_*`); `legal_{arm}` for all four arms, one pass over the 500M-position dataset (5,944 updates) each, plus `legal_hybrid_seed2`; `engine_{arm}`, about 50M Leela positions continued from `legal_{arm}`; then `engine_long_{arm}`, two passes over the 100M-position Leela dataset, to look for a signal at the deep end before the main runs.
- **On one GPU:** `pilot.py queue <runs> --slots K` runs several side by side and resumes any run from its last checkpoint (every 200 updates) when started again; `pilot.py bench <runs> --micro-batches …` times them side by side first, to choose K and the micro-batch size.
- **Pass criteria** (none requires an arm to win): all tests pass; build and training costs are measured; the dev sets have room to improve (otherwise harder sets or a smaller scale, declared in advance); learning curves and seed variance are measured.

## Steps

1. ✅ Tokenizer, parsing, teacher.
2. ✅ Model and trainer: untied readout, sparse targets, live evaluation, continued runs.
3. ✅ Fixed datasets and one loader.
4. **Pilot:** runner ✅, stage-1 dataset ✅, Leela dataset (downloading); then run it and fix the open decisions.
5. **Evaluation panel** once a model plays chess: label a development panel, set node budgets and margins on it, check that labels reproduce; label the test panel only for the frozen analysis.
6. **Stage 1:** 4 arms × 3 seeds.
7. **Stage 2:** 4 arms × 3 seeds.
8. **Controls**, where the signal calls for them.
9. **Report** next to the scripts.

## Decisions

**Open until the pilot:** the engine stage's learning rate; update count and schedule; the Leela archive range; the learning-speed thresholds.
**Open until the development panel:** panel size and node budgets; practical margins.

**In force** (date — decision — reason):
- **2026-09-30 — Leela test80 self-play data for the deep target.** Dense and strong, free under the ODbL. Measured on one archive: 50,807 games, 5.6M positions; 96% standard starts; every standard game replays; median 467 visits per position; the top move's probability has a median of 0.55, with 42% of the probability on searched alternatives and 3.5% on the one-visit floor. Rejected: our own Stockfish labels (weak at affordable scale), ChessBench (no game histories; records shuffled, as checked on its test file), PAWN (per-move values without search).
- **2026-09-30 — Fixed datasets, one order for every run.** All chess code at build time, identical data for every run, no loader bottleneck (reading: 25M positions/s in one process).
- **2026-09-30 — Raw Leela targets.** Removing the one-visit floor took legal moves out of the softmax, and filtering by visits broke loss pooling; neither could be judged without the panel.
- **2026-09-30 — The evaluation panel waits until a model plays chess.** The pilot judges stage 1 by legality and stage 2 by agreement with Leela.
- **2026-09-30 — Stage 1 is the legal target only; the played condition is a control.** The question is shallow vs deep targets; comparability with the 20B run is not a goal.
- **2026-09-30 — Pilot defaults** as in [Training protocol](#training-protocol): the 20B recipe where it worked, and the 20B report's recommendations (a broad, gap-free support; warm starts in the decay) where it didn't.
- **2026-10-01 — Seed variance is large at pilot scale.** Two hybrid seeds at the same settings ended at 0.890 / 0.926 human and 0.512 / 0.648 random exact set: as large as the gaps between arms. Orderings need several seeds; random-game exact set is the noisiest measure.
- **2026-10-01 — Every arm at 1e-3; short learning-rate checks misjudge the recurrent arms.** Checks at a fifth of a run (1,189 updates), dev legal loss: transformer 0.0185 / 0.0086 / **0.0058** / 0.0068 at 1e-4 / 3e-4 / 1e-3 / 3e-3; temporal 0.0197 / **0.0195** / 0.0306 at 1e-4 / 3e-4 / 1e-3; depth 0.0191 / **0.0161** / 0.0270; hybrid 0.0384 at 3e-4, **0.0349** at 1e-3. But full stage-1 runs reversed the recurrent result: temporal ended at 0.931 / 0.637 (human / random exact set) at 1e-3 against 0.834 / 0.347 at 3e-4, and depth at 0.866 / 0.435 against 0.852 / 0.398. At a fifth of a run the recurrent arms have barely left the early plateau, so the check measures how fast they leave it, not where they end. The 3e-4 runs are kept as records (`_lr3e-4`).
- **2026-09-30 — Exact set is thresholded.** The earlier version picked the true number of top moves, so it measured ranking; that is now `separation`. The constant baseline uses one legal rate, not each position's count.

## Cost

- **Measured on a Verda A100 80 GB spot instance** ($0.89/h; EPYC 7643, 22 vCPUs), at the deepest update mixture, without evaluation: one hybrid run takes 0.44 s per update at micro-batch 50 (0.53 s at 20, 0.39 s at 400; 7 GB at 50). Four arms side by side take 0.98 (transformer) to 1.63 s (hybrid) per update each, about as long as running them one after another: the GPU, not the CPU, is the limit.
- **Stage-1 pilot run:** 5,944 updates, at most about 0.75 hours alone on an A100, so under $1 each on spot pricing; the five stage-1 runs about $3.
- **Builds:** stage 1 at 500M positions in 55 minutes on 8 workers; Leela conversion about 2,100 positions/s per process (1.7 hours per 100M on 8 cores); the Leela download runs at 0.4–3.7 MB/s.
- **Storage:** 30 GB for the pilot stage-1 dataset, ~120 GB for the full one; ~11 GB per 100M Leela positions (estimated).

## Later extensions

- **Per-move values at scale:** our own Stockfish labels on stage-2 positions, if the policy target proves too coarse.
- **The Chess-World-Model benchmark,** for a direct comparison with its published baselines.
- **Board-given control:** the same targets with the current position as input, to test whether stage-2 advantages come from reconstructing the board.
- **A look-ahead target:** the engine's principal variation, if loops show no benefit.
- **Harder evaluations:** Lichess puzzles with their histories, and play against fixed Stockfish levels ([Miłosz et al., 2026](https://arxiv.org/abs/2608.27757)).
