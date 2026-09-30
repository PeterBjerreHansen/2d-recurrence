# Conceptual review and paper strategy

Reviewed 2026-09-29. This is a review of the documented experiments and selected execution/evaluation code, not a rerun or a comprehensive code audit. Existing results below are reported results. Proposed experiments have not been run.

## Assessment

There is enough substance to start writing a serious empirical paper. There is not yet evidence for the original strongest claim: that joint temporal and depth recurrence develops a growing advantage over either axis alone. The most useful next step is to identify a condition under which their combination adds value, or establish carefully why it does not.

My preferred research question is:

> When does reusing computation across tokens complement performing more computation within a token?

The shared training trajectory is a useful method for studying this question. Chess gives unusually good access to the underlying state, action validity, and decision quality. The paper should turn that access into an explanation of the two axes, rather than rely on a small aggregate language-modeling gain.

## What the evidence currently supports

The main sources are the [20B report](../experiments/long_runs/20B_recurrence/REPORT.md), [board-state report](../experiments/interp/board_state/REPORT.md), [original protocol](20B_experiment_plan.md), and [execution contracts](INFERENCE_CONTRACT.md).

| Candidate claim | Assessment |
| --- | --- |
| Both axes can be trained in one shared pass trajectory | Implemented, with clear limiting cases and accounting. A defensible technical contribution if positioned precisely against prior work. |
| Adding depth is almost free during temporal multipass training | Supported by the estimated cost of this curriculum. This is a marginal cost claim conditional on already doing temporal multipass training; deployment depth still costs compute. |
| Hybrid outperforms both single-axis models | Not supported on the primary training-graph endpoint: hybrid and temporal are effectively tied. |
| Hybrid's advantage grows with training scale | The reported experiment gives the opposite trend: gaps shrink. Keep this negative finding visible. |
| Hybrid gives the best reported live NLL | Numerically yes, but the strongest repaired temporal baseline nearly matches it with much less inference work. |
| Temporal recurrence makes board state available earlier | Supported by the tested probes, with character-dependent effects. |
| Recurrence learns a more accurate world model | Not established. Peak board-probe accuracy is similar across the local arms, and the longer-trained reference is higher. |
| The mixer destroys board information | Not established. The tested probes recover less information, and nonlinear probes recover much of the apparent loss. |
| Temporal memory influences outputs | Supported by swaps and steering, with important limitations on isolating specific routes and locations. |
| More depth performs chess search | Not measured. Human-move likelihood, legality, and engine-rated decision quality are different targets. |

An important comparison belongs in the main paper, not in a footnote:

| Live execution | NLL | Transformer-block applications per token |
| --- | ---: | ---: |
| Transformer | 0.2264 | 8 |
| Temporal, post-hoc aligned | 0.2176 | 8 |
| Temporal, aligned decay | 0.2142 | 8 |
| Hybrid, J=2 | 0.2144 | 12 |
| Depth, J=4 | 0.2148 | 20 |
| Hybrid, J=4 | 0.2137 | 20 |

These rounded numbers come from the 20B report's live comparisons. They do not provide a new paired confidence interval for aligned-decay temporal versus hybrid. Recompute that comparison directly before treating the roughly 0.0005 difference as reliable. Mixers, cache storage, and training/alignment costs differ; block counts are not complete compute measures.

This table changes the scientific interpretation. The current evidence is consistent with temporal state reuse doing most of the useful work in this task, with only a small additional return from depth. That is a useful finding even if the hybrid does not become the preferred model.

## Novelty and positioning

The related-work search found close precedents. See the separate [primary-source review](paper_related_work_review.md).

- [Full-bandwidth Transformer](https://arxiv.org/html/2608.08888v1) already develops shifted multipass feedback training, discusses its relationship to looped training, and reports unstable extrapolation improved by mixing deeper passes. Neither the common training pattern nor generic training/deployment mismatch should be presented as newly discovered here.
- [T²MLR](https://arxiv.org/html/2607.15178v1) is a particularly close temporal comparison: middle-layer recurrence and parallel Jacobi training. Its treatment of state tracking, exact versus approximate execution, and training cost raises the standard for a new temporal recurrence claim.
- [Latent Recurrent Transformer](https://arxiv.org/html/2605.26797v1) already studies high-level temporal feedback and explicitly proposes combining it with depth recurrence as future work. The broad suggestion to combine the axes is therefore insufficient as the novelty claim.
- [Karvonen's chess study](https://arxiv.org/html/2403.15498v1) already establishes board probing and causal piece-removal interventions. The potentially distinctive contribution here is the timing, routing, and reuse of state under recurrence.

The remaining credible contribution is a specific implementation of joint state updates in one training trajectory, plus controlled evidence about complementarity, interference, and deployment behavior. This search does not establish priority for that exact implementation.

The two axes share scheduling machinery but do not solve the same computational problem. Temporal passes approximate a causal recurrence over positions; depth iterations are additional deployed computation at a position. Write counts, training passes, solver iterations, and inference depth must remain separate in the exposition.

## The decisive experiment

Separate the need to reconstruct state from the need to compute an answer from that state. A useful design crosses two task dimensions:

| Input | State-oriented target | Decision-oriented target |
| --- | --- | --- |
| Game history | Board query or legal-move target | Engine move distribution / regret |
| Explicit current position | Same board query or legal-move target | Same engine move distribution / regret |

Use the same underlying positions across conditions. Train or adapt each input condition appropriately; switching a PGN-only model to FEN at evaluation would introduce an uncontrolled distribution shift. Include castling rights and en-passant state, and handle history-dependent draw information if the target needs it. FEN reduces reconstruction demands but does not eliminate all temporal processing.

The hypothesized pattern is a larger temporal benefit with history input and a larger depth benefit for difficult decisions. A particularly convincing hybrid result would occur when both demands are present. Test these interactions directly rather than infer them from separate significant comparisons.

Start cheaply: run engine-based evaluation on existing checkpoints before collecting a large fine-tuning dataset. Include original and aligned-decay temporal, depth, transformer, and hybrid at J=1/2/4. Report paired changes in full-move quality as J increases. Extend J beyond training only as a separately labeled extrapolation test.

For a general architecture paper, add one compact non-chess task where state-update length and query-computation depth are independently controlled. For example, maintain registers through an update stream, then answer a variable-length computation over the final registers. Include an explicit-final-state control. This tests the interaction more cleanly than collecting several unrelated benchmarks. If the paper stays a narrowly scoped chess mechanism study, this additional domain is optional.

## Changes to the engine-policy proposal

The [engine-policy plan](engine_policy_plan.md) is a promising next direction, but several conclusions and metrics need tightening before freezing it.

1. **Use the strongest temporal baseline.** Aligned-decay temporal is already substantially better than the post-hoc aligned checkpoint. A win over the weaker model would leave the central question open. Equalize additional training treatment where practical and disclose all costs.
2. **Measure difficulty independently of the best–second-best gap.** A large gap can be a trivial recapture; a small gap can contain several moves requiring substantial calculation. Use shallow-versus-deeper engine disagreement or regret, with a stronger evaluation budget and a label-reliability check. Gap, captures, checks, and ply are useful strata, not proofs of search difficulty.
3. **Handle illegal probability explicitly.** The raw model probabilities of legal SAN strings need not sum to one. Summing regret only over those probabilities can reward putting mass on illegal outputs. Report legality separately and conditional legal regret, plus an unconditional utility or regret measure with a declared illegal-move penalty. State whether deployment constrains moves to the legal trie.
4. **Distinguish the loss from full action-distribution distillation.** The conditionals in the trie are exact, but training only at human-visited prefixes weights nodes according to the human policy, not the target policy. With finite data and shared parameters, important engine-preferred branches can be undersupervised. Add at least a sampled target-policy branch or the engine's best-move branch; describe the resulting sampling/weighting precisely.
5. **Name the objective correctly.** The plain exponential target corresponds to entropy-regularized improvement, or KL regularization to a uniform legal-move reference. KL improvement relative to a nonuniform reference requires its probability factor. The later reference-weighted version in the plan supplies that factor.
6. **Treat converted centipawn scores as a utility surrogate.** Define exactly what the engine target means and pin its implementation. Do not assume that its transformed value is a universally calibrated win probability.
7. **Weaken the decision rule.** Failure to find a gain does not show that depth is unnecessary in chess. It shows no detected benefit at this model size, objective, training regime, and budget. Predefine a practically meaningful improvement or equivalence margin. A within-hybrid J=1→4 gain establishes useful extra computation; beating a strong temporal baseline and a relevant compute control is a further claim.

## Close the deployment question before scaling

The temporal failure and repair are substantial observations, but the generic phenomenon has prior art. The new question is whether the hybrid prevents the problem for architectural reasons, because its schedule is broader, or through some other change in optimization.

Compare temporal and hybrid under the original support and a broad, gap-free support, with expected training work controlled. Start with the same maximum update count and change parity coverage; changing support to include an extra maximum pass confounds coverage with more computation. Keep a separate deeper-support arm if useful.

Then compare ordinary extra fine-tuning against settled-memory alignment under declared data and compute budgets. Compare sparse deeper forward passes against many detached settling passes where feasible. Log live NLL, class-specific NLL, even/odd update behavior, and distance to the exact live execution throughout training.

The proposed 1B experiment is a screen, not a conclusive stability test: the reported failure was much stronger after 4B and worsened during decay. Continuations from retained checkpoints are efficient causal diagnostics of a training change, but are not independent training seeds. Use independent seeds at a horizon that actually reproduces the phenomenon before making a general claim.

Be careful with fixed-point language. For a finite causal sequence, repeated updates of the appropriate live-equivalent operator become exact from left to right; the code tests equality after sequence-length many passes. A small residual after a short cap is an approximation, not a general guarantee of global contraction. Training-graph hybrid passes and the nested live-depth settling operator also need explicit distinction. Report maximum-position residuals, fraction hitting the cap, and direct logit agreement with sequential inference on representative full-length rows.

## Tighten the mechanistic story

The most interesting existing result is character-dependent timing: memory arms make the board accessible before the next decision character, yet the mixer reduces its readability and later layers reconstruct a stronger readout. This raises a precise question: is computation at a preceding character serving a later prediction?

Test that question by selectively intervening on the outgoing temporal write at the last move-number digit, at other digits, and at move endings. Measure the effect at the next decision and on complete move probabilities. Keep immediate-read interventions separate from interventions allowed to propagate through future memory and caches. Use same-role, same-ply donors and matched perturbation controls.

Distinguish causal dependence on temporal state from dependence on older cached states. The existing wide-window swap modifies a coupled computation over many positions; it demonstrates sensitivity but does not isolate a board-specific direct pathway. A crossed memory/cache intervention, carefully scoped to avoid conflating immediate and downstream effects, would be more informative than many additional unconstrained probe plots.

Several current interpretations should be narrowed:

- Lower linear or MLP probe accuracy means lower recoverability by those probes, not proven information destruction.
- Near-perfect side-to-move decoding does not establish causal use of that feature at that site.
- Blocking attention to one position does not rule out recovery from other positions.
- Failure after replacing gates with their means shows dependence on the trained gates; it does not prove input-dependent gating is indispensable in an independently trained alternative.
- A flat board or from-square probe across loops does not show the loop performs no useful computation. Measure changes in full-move probabilities and engine quality as well.
- Late-game gains are consistent with greater state-tracking demand, but game phase, predictability, and position difficulty also change with ply.

## Compute, seeds, and evaluation

Report distinct comparisons: fixed data, fixed training FLOPs, and fixed inference latency. Include parameters and KV-cache bytes. A dedicated cache per depth can be an important cost even when per-token timing looks similar. Measure prefill and decode separately on the same hardware; sequential prefill matters for short continuations.

The 0.2% marginal training estimate depends on the particular joint schedule, including when temporal-source work is performed. Report a paired add-depth comparison that preserves temporal write schedules to isolate architectural overhead. Do not present the curriculum average as a universal price of a depth connection.

At 2.81× estimated per-character training cost, the hybrid's 20B run corresponds roughly to 56.2B ordinary-transformer characters under the same counting assumptions. Karvonen's 61.4B-character reference is therefore a useful sanity check, but a different training recipe is not a controlled compute-matched baseline. “Three times less data” should stay a data-exposure claim.

Use at least three independent training seeds for the central comparison as a practical starting point, reporting individual effects and uncertainty rather than assuming three is enough for a tiny gain. Bootstrap rows/games separately from training variation. More evaluation rows cannot resolve seed uncertainty.

The preparation code rejects exact row overlap but explicitly does not assert game-level separation. Audit duplicate games and shared game fragments. Keep the historical upstream split for replication, and add a fresh game-disjoint final evaluation where possible. Reserve it before new mechanism or engine-policy decisions; report exploratory, protocol-fixed, and post-hoc results distinctly.

## A staged route to a paper

1. **Write around current evidence now.** Draft the method, original hypothesis, negative primary result, deployment behavior, and limitations. Put aligned-decay temporal in the central table. Build an explicit related-work comparison before using novelty language.
2. **Use existing checkpoints to choose the story.** Obtain paired live comparisons, complete-move engine metrics, and one carefully isolated causal timing test. Freeze selection rules for the next stage.
3. **Run the smallest experiment that distinguishes explanations.** Test schedule coverage and alignment controls, then pilot engine-policy adaptation with strong temporal and depth controls. Do not change mixer, objective, schedule, and architecture simultaneously.
4. **Replicate the decisive result.** Spend the next serious training budget on independent seeds and compute controls. Add one controlled task family if making a general architectural claim.
5. **Stop expanding once one claim is secure.** A precise paper about conditional complementarity or state reuse is more convincing than a catalogue of loosely connected analyses.

Suggested working title: **Temporal and Depth Recurrence: When Does Shared Training Yield Complementary Computation?**

A five-figure paper could contain: (1) train/live computation graphs; (2) live quality against inference and training cost, including the strongest temporal baseline; (3) the state-reconstruction × decision-computation interaction; (4) deployment mismatch and controlled repair; (5) a causal state-timing result. Architecture-selection history, exhaustive probes, gate sweeps, and full recurrence grids can go in the appendix.

If the controlled experiments still show no meaningful depth benefit, publish the bounded result: joint training is inexpensive, but its benefits are not automatically additive; temporal reuse explains most gains in this setting. If causal timing is the strongest new finding, narrow the paper to when recurrent models compute, transmit, and reuse state. Neither outcome requires forcing a hybrid-superiority narrative.
