# Review of the move-target plan and chess literature survey

Reviewed 2026-09-29. Sources: [the move-target plan](engine_policy_plan.md) and [the chess literature survey](chess_related_work.md) on `engine-policy`, as revised in commit `30be03d`. The review is saved in the current checkout, which is now on `engine-policy`. The proposal and literature survey were not edited. This is a design review; no training results were reproduced and the literature checks concentrate on the closest precedents.

The plan asks a useful question: does supervision change the relative benefit of temporal and depth recurrence? Move tokens remove the PGN character cycle, the human condition provides a new reference under that tokenizer, and legality versus engine values is a useful distinction. Three independent seeds, live evaluation, teacher-budget disagreement, and a bounded interpretation of null results are good choices.

I would proceed with the pilot after correcting the execution tests and output metrics. Before the main experiment, specify the annotation protocol and the compute comparison. The present design can establish whether a legality-pretrained hybrid benefits from additional deployed loops. A stronger claim that engine supervision specifically induces depth computation needs additional controls.

## Changes needed before implementation

### 1. Test live inference against its oracle, not arbitrary training passes

Plan lines 152 and 169 require live execution to equal the training graph. This contradicts [the inference contract](INFERENCE_CONTRACT.md), especially lines 74–90. Cached live inference should equal the slow sequential reference within numerical tolerance. Temporal and hybrid training graphs approximate a different execution; their discrepancy is a measured property, not automatically an implementation failure. The documented depth-only identity also requires the appropriate `depth_specialized` cache policy.

Replace both tests with cached-live versus sequential-reference checks for all arms. Keep the applicable exact graph identities as separate tests. Declare the primary KV-cache policy and report training/live differences separately at every checkpoint. Otherwise the pilot can fail correct code or encourage changes to the intended recurrence semantics simply to satisfy an invalid test.

The warm-start recipe also needs an arm-specific definition. The current trainer permits warm-start batches only for temporal mode (`train.py`, lines 140–144). If hybrid receives them too, specify and implement the corresponding live-depth settling operator. Count settling passes in the training budget. If only temporal receives them, state that these are comparisons between complete architecture/training recipes.

### 2. The value head cannot also serve as a legality head

Plan lines 50, 58, 67 and 122–124 together create a contradiction: stage 2 discards the legality output layer, creates a fresh value layer, and never trains illegal actions at those positions, yet exact-set legality is measured by ranking that layer.

A legal move with Q=0.01 is supposed to score below an unconstrained illegal move with output 0.5. That can happen with a perfectly accurate legal-action value predictor. Stage-1 board features do not restore the discarded head or establish a legal/illegal ordering for the new one.

Keep separate legality and value readouts if legality retention matters. A retained legality head evaluated on the changed backbone can measure retention; a newly fitted probe with a fixed fitting protocol measures recoverability. State which question is intended. An auxiliary legality loss is another option, but it changes the engine condition to multitask training and should be declared as such. Keep the rules-based legal filter for value evaluation.

The reverse comparison also needs care. An ideal legality head assigns the same score to every legal move, so its within-legal ranking has no decision-quality meaning. Regret can be reported as an exploratory by-product, with a uniform-legal baseline and a fixed tie rule, but it is not a comparable trained policy target. Human policy exact-set accuracy likewise measures whether rare legal moves outrank illegal moves, not simply whether the model knows the rules.

### 3. History must be present in the teacher, not only the student

The proposed history advantage over ChessBench is conditional. Giving the student histories while supervising it with position-only labels does not teach different values for histories that reach the same FEN but have different repetition possibilities.

Specify whether Stockfish receives the reconstructed move stack, how draws and claims are handled, and how labels are keyed. The [python-chess engine interface](https://python-chess.readthedocs.io/en/latest/engine.html#chess.engine.Protocol.analyse) sends the board's move stack, so replayed boards can preserve the relevant history; recreating boards from FEN loses it. Cache history-sensitive labels using a sufficient history key rather than FEN alone.

The [ChessBench loader](https://raw.githubusercontent.com/google-deepmind/searchless_chess/main/src/data_loader.py) decodes action-value records as FEN, move and win probability. These are useful board-value targets, but cannot supply the proposed history-dependent differences or recover mate distance from the stored probability alone. Treat label reuse as a board-only teacher condition; reannotate repetition and mate-distance cases when those are part of the claim.

Add a small targeted test of equivalent current positions with different repetition histories, with teacher labels verified to differ when appropriate. En passant and castling strata are useful, but are already represented in FEN and do not test this extra history advantage.

### 4. Define the engine annotation procedure completely

Plan line 97 overstates determinism from a pinned binary and node count. Also fix the NNUE network, thread count, hash size and clearing policy, tablebases, engine strength, score perspective, and treatment of incomplete or bounded evaluations. Use one thread per annotation worker and verify identical labels when a sample is processed twice in a different order. Parallelize independent workers.

State whether the node budget is for the entire all-move MultiPV search or separately for each candidate. These define different teachers and different costs. Store completed depth, nodes and score bounds; require coverage of every legal root action. Shallow/deep disagreement can otherwise reflect incomplete annotation or unequal work across branching factors. Stockfish exposes these options in its [engine implementation](https://github.com/official-stockfish/Stockfish/blob/master/src/engine.cpp); python-chess exposes MultiPV, root-move restrictions and score perspective in its [engine API](https://python-chess.readthedocs.io/en/latest/engine.html).

Define every Q from the player choosing the root action's perspective. This is particularly important if annotating child boards, where side to move changes. Add known white-to-move and black-to-move winning positions to the conversion tests.

Use a stronger, separately specified evaluation teacher on a manageable held-out panel to test whether gains survive beyond fitting the training teacher. Define best-move agreement with an explicit tie or near-tie policy and use tie-aware rank correlation.

## Changes needed before drawing the main conclusion

### 5. The engine stage inherits differences from the legality stage

Plan lines 66–69 match pretraining exposure, which is appropriate for an end-to-end recipe comparison. However, the plan predicts that temporal and hybrid will enter stage 2 with better legality/state reconstruction. Their eventual value advantage can therefore reflect a stronger starting representation, rather than an advantage specific to decision computation. Stage 2 also changes the input mixture to human-only games.

Keep matched exposure as the primary analysis, but predeclare the matched-legality continuation rather than deciding to run it after seeing a favorable result. Choose the matching metric, tolerance and checkpoint rule on validation data, preferably including board-probe performance as a separate diagnostic. Report the additional pretraining cost and whether matching was actually possible for all arms. It is a diagnostic, not a replacement for the primary recipe comparison.

If the paper claims that target choice favors different axes, the human-pretrained engine continuation should become a small planned control rather than an unspecified option. A board-given control is especially informative if interpreting an engine advantage as reduced need for state reconstruction. This control may remain a later extension if the claim stays limited to the history-input training recipe.

A within-hybrid J=1 to J=4 improvement establishes that more deployed computation helps this model. It does not alone show that engine targets uniquely cause useful loops. Evaluate human and engine models on the same annotated decision panel and predeclare an objective-by-loop interaction where their policies are meaningful. More loops can also improve representations and cache histories accumulated earlier in the game; describing this as search or look-ahead requires additional evidence.

### 6. The proposed compute-matched temporal comparison is not yet defined

Plan line 144 says hybrid beats temporal-only at matched block applications. Under the current contract, temporal-only has J=1 and eight block applications; hybrid uses eight, twelve or twenty at J=1,2,4. There is no twenty-block temporal-only point in the specified four-arm experiment.

At eight blocks, hybrid J=1 versus temporal J=1 is a valid comparison of the trained recipes, but does not establish a depth-loop advantage at matched compute. Either limit the claim to that comparison or add a temporal baseline with the relevant deployed compute, such as a deeper untied core. Report the parameter difference if choosing the latter.

Keep data-matched and compute-matched results distinct. A quality–latency curve is useful, but does not create a missing compute-matched control. Include measured decode latency, prefill latency and cache bytes; block counts omit mixer and cache costs. Report training FLOPs including warm starts separately from token exposure.

### 7. Sparse legality BCE needs a meaningful baseline and loss reduction

About 35 of 1,968 targets are positive, or 1.8%. A constant predictor at that positive rate already has average BCE of about 0.089 nats; predicting every action illegal also looks excellent under plain per-label accuracy. Precision, recall and exact-set accuracy are therefore essential, as the plan already recognizes.

Add separate positive and negative losses and a constant-prevalence baseline. Benchmark unweighted BCE first; introduce balanced weighting only if the pilot exposes poor learning of positives, and freeze the choice before main runs. Do not treat a small average BCE as evidence that the rules are learned.

Also specify engine loss reduction: mean over legal moves per position, followed by mean over positions, differs from averaging all legal-action labels across the batch. The latter weights positions with many legal moves more heavily. Define rank-based weighting before using it and retain the unweighted pilot result. The same AdamW settings across differently scaled objectives are a starting point, not proof of comparable optimization; allow an equal small tuning budget.

### 8. Pilot selection should depend on information and feasibility

Requiring the arms to separate on a test set (line 153) selects for a desired architectural result. A useful task can have a genuine null effect, and a saturated legality task may still provide useful pretraining for the engine stage.

Use pilot criteria that do not require the hypothesis to win: valid labels, correct live execution, adequate throughput, measured learning dynamics, acceptable seed variance, and an evaluation panel with room to improve. Keep development panels separate from the untouched final tests. Predeclare a fallback scale or harder panel if all arms reach the metric ceiling.

With three training seeds, show individual seed effects and uncertainty. More positions do not replace more independent training runs. Cluster position resampling by game, and analyze the paired high-versus-low-disagreement interaction explicitly rather than comparing two separate significance statements. Set practical margins on pilot validation data; a confidence interval merely excluding zero can still represent a negligible effect.

## Data and budget corrections

- One pass over 1.5B human moves with random moves making up 20% of move exposure means approximately 1.875B moves in total, before game-start tokens and padding. A 1.5B total budget instead exposes 1.2B human moves. Resolve lines 76–87 and define counters for human moves, random moves, valid targets and physical row tokens.
- A maximum of 185 moves measured on 2,000 rows is not a corpus-wide bound. Validate all converted lengths and define how oversized games are handled. Express lengths in plies to remove the ambiguity between moves and move pairs.
- Hashing extracted move sequences groups identical sequences but does not group a truncated prefix with its full game. Preserve source game identity where possible; otherwise audit repeated fragments and prefix overlaps before asserting game-disjoint splits. Report familiar-board and novel-board engine results separately, since different games can reach identical positions.
- With 1.5B positions and 35 legal actions per position, uint16 action lists require roughly 105GB before offsets and metadata. Specify the storage format and count label-generation CPU cost in the pilot.
- Training on random games makes held-out random games an in-distribution test of that generator. They remain useful unfamiliar trajectories, but do not reproduce a human-only-to-random transfer result. The unseen-generator test and targeted rare-rule histories carry the distribution-shift claim. Separate seeds alone do not guarantee distinct games; deduplicate test sequences against training if making that guarantee.
- State carry across packed games is an intentional choice. Test whether the same game changes materially when placed first versus after another game, and distinguish packing position from within-game ply. It may introduce an extra learned reset demand. A reset/mask control is useful if this effect is material.
- Rebenchmark cost after including the larger output projection, sigmoid losses, padding, legal generation and warm-start settling. The old character-run price provides an order-of-magnitude starting point, not a measured cost for this pipeline.

## Literature-review corrections

The survey identifies the right neighbors: state tracking, history-input human models, board-input distillation, and mechanistic work on look-ahead. Four statements need tightening.

1. **Legality BCE is not the exact Othello objective.** At a fixed state, uniformly sampled legal next moves give expected categorical CE `−mean(log p(a))` over legal actions. Independent binary CE over legal/illegal membership has a different normalization and optimum. They share legal support; the latter is direct membership supervision. Correct the “exact, noise-free version” wording at survey line 22.
2. **The two-bin identity is mathematical, not a reproduction of Ruoss's method.** BCE with soft label Q is CE against `(1−Q,Q)` on support `{0,1}`. [Ruoss et al.](https://arxiv.org/html/2402.04494v2) instead use interval bins with smoothed categorical targets and condition their predictor on the candidate action; all-action scoring requires evaluating each legal action. The proposed simultaneous 1,968-output head is a distinct design. Their action-value advantage also partly reflects additional supervised action examples. Describe this plan as action-value distillation with a scalar soft-label loss, and avoid implying that their result validates this particular head/loss.
3. **History and mate tie-breaking do not guarantee consistent winning play.** The plan's `min(n,3)` makes mate-in-3 and mate-in-20 identical. Saturated centipawn values also retain ties. Replace “shorter mates are worth more” with the exact limited distinction, or use a strictly distance-sensitive utility within a fixed mate interval. Describe history and tie-breaking as hypotheses to evaluate.
4. **Qualify the theoretical and novelty claims.** [Merrill et al.](https://arxiv.org/html/2404.08819v1) analyze exact tracking over arbitrary-length source/target sequences, under computational assumptions including the usual complexity separation, and ignore draws in their chess construction. This motivates recurrence but does not predict failure over this finite 256-token task. Survey line 35 also still describes SAN character inputs and a move head; update it to the new UCI-token design. Keep novelty language scoped to the works surveyed and to the specific controlled comparison.

The [Chess-World-Model paper](https://arxiv.org/html/2605.30100v1) explicitly links construction and training code, so the first availability question in “Later extensions” has an answer. Check actual repository usability next. Its human-trained random-game evaluation differs from the mixed training distribution proposed here; its state target also includes auxiliary fields beyond piece placement. If pursuing a direct benchmark comparison, match that interface and protocol rather than only its move-token input.

My recommendation is to keep the four-arm pilot and the move-token design. Correct the execution and head semantics first; use a small history-aware engine panel to validate teacher quality and useful J-dependent gains; then freeze the primary comparison, compute control and analysis rules before committing to the 24 full pretraining runs.
