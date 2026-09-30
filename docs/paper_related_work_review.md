# Paper positioning and closest prior work

Research date: 2026-09-29. This is a targeted conceptual review, not an exhaustive novelty search. Primary papers were inspected, including the full text for the closest temporal-recurrence work. Statements about the current project refer to its reports, not independently reproduced results.

## Main assessment

The best defensible contribution is a **controlled study of jointly training temporal and depth recurrence in one shared pass trajectory**, including the cost of adding the second connection, where the axes substitute for one another, and when parallel training fails to transfer to live recurrence. Neither recurrence axis, parallel Jacobi training, the general similarity between their training procedures, nor chess board-state probing is new by itself.

The strongest architectural distinction I found is the combination of two held states with independently sampled write counts and a common `max(U_T,U_D)+1` pass budget. The sources below do not establish that this exact sampler is unprecedented. A paper should describe the construction precisely and avoid a broad first-ever claim.

## Eight close comparisons

| Primary source | Established overlap | Implication for this paper |
| --- | --- | --- |
| [Huginn: Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171), Geiping et al., 2025 | Shared recurrent depth supports additional latent computation at inference; demonstrated at billion-parameter scale. Depth recurrence has older roots in [Universal Transformers](https://arxiv.org/abs/1807.03819). | Credit the depth axis as inherited. The experiment must establish what temporal feedback adds at a given deployed compute budget. |
| [Ouro: Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741), Zhu et al., 2025, revised 2026 | Looped pretraining, learned depth allocation, and large-scale results distinguish knowledge manipulation from knowledge capacity. | Four passes improving chess NLL is not sufficient evidence of a new reasoning mechanism. Compare iteration-dependent task performance and actual cost. |
| [Full-bandwidth transformer](https://arxiv.org/html/2608.08888v1), Wang et al., 2026, §3.3 and §5 | Gated previous-token latent feedback, non-detached parallel multipass training, and explicit comparison with looped training. Reports oscillatory failure outside trained feedback depths; adding a small fraction of deeper passes restores empirical stability. Uses losses on every pass. | The shared-training observation and generic training/live mismatch already have direct precedent. Distinctions include joint axes, independent state writes, final-pass supervision, and specific alignment diagnostics. Compare against its broad-depth training recipe before treating hybrid stability as intrinsic. |
| [T²MLR: Transformer with Temporal Middle-Layer Recurrence](https://arxiv.org/html/2607.15178v1), Cai et al., 2026, §2–3 and Appendices B–C | Later middle-layer state feeds an earlier layer of the next token. Only the recurrent middle block is rerun during Jacobi training. Separates forward approximation depth from backward depth; evaluates approximate versus exact execution, state tracking with retrieval, recurrence placement, and training-compute controls. | This is the closest temporal architecture, beyond input-level feedback. Localized temporal recurrence and cached outer layers are established. The main difference is the jointly trained within-token depth state and shared schedule. Its longer forward approximation is an important alignment baseline. |
| [Latent Recurrent Transformer](https://arxiv.org/html/2605.26797v1), Huang et al., 2026, §3–6 | Prior-token high-level state enters earlier computation; interleaved parallel training refreshes disjoint token subsets. Explores source layer, temporal source, and injection pathway. The conclusion explicitly suggests combining temporal feedback with depth recurrence. | The bare suggestion to combine the axes is anticipated. A concrete low-marginal-cost construction and evidence about its interaction can still be useful. Avoid claiming recurrence-site design as entirely unexplored. |
| [PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space](https://arxiv.org/html/2509.23184v1), Zeng et al., 2025, §3 | Uses Jacobi iteration to train latent autoregressive computation in parallel, randomizes iteration counts, and permits multiple latent thoughts per emitted token. It interleaves latent positions with real token embeddings. | Explain the difference between adding latent sequence positions and reusing a core at a fixed token position. Parallel approximation and variable iteration training have clear precedent. |
| [Addressing Some Limitations of Transformers with Feedback Memory](https://arxiv.org/abs/2002.09402), Fan et al., 2020 | High-level representations from the past are made available to lower-level computation at later time steps. | Cite as conceptual temporal-feedback lineage; do not equate all of its training or memory details with the present one-token gated connection. |
| [Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/html/2403.15498v1), Karvonen, 2024, §3–4 | Character-level chess models, linear square probes, player-skill representations, and causal piece-removal interventions already exist on this backbone. | Replication is validation. The new question is how recurrence changes **when and through which route** board information affects predictions, not whether a chess model has a board representation. |

## What the existing evidence supports

The [20B report](../experiments/long_runs/20B_recurrence/REPORT.md) supports a narrow empirical statement: recurrence improves data-matched chess likelihood; hybrid ties temporal in the training graph; temporal alignment matters greatly; the hybrid improves over depth at the same number of deployed block applications. It does not establish universal complementarity, compute-optimality, or improved search. The strongest temporal alignment result must be in the central comparison: aligned-decay temporal reaches live NLL 0.2142 with eight block applications, close to hybrid's 0.2137 with twenty. That is a different practical story from comparing hybrid only with the weaker post-hoc repair.

The [board-state report](../experiments/interp/board_state/REPORT.md) supplies a more distinctive mechanism direction:

- Board accessibility depends on the computational role of a character, including different roles of the same space token.
- Temporal feedback makes board information accessible earlier without giving the best final board accuracy.
- Narrow memory swaps can be corrected by the rest of the network; wider interventions alter later representations and decisions.
- Later loops add little to the tested board and human-move readouts.

These observations suggest **redistribution and redundancy of computation**, rather than a clean assignment of state tracking to time and search to depth. The latter remains a hypothesis.

## Experiments that would make the distinction convincing

1. **Establish a fair deployable frontier.** Include strongest aligned temporal, hybrid at each useful iteration count, depth, and ordinary transformer. Plot quality against measured inference cost and report training FLOPs separately. Replicate the central comparison across training seeds before interpreting a 0.0005–0.0012 gap.
2. **Separate the architecture from the training distribution.** Compare gap-free temporal-only schedules, hybrid schedules, and temporal warm starts with the same forward-pass budget. Add longer forward approximation with truncated backward depth as a T²MLR-inspired control. This tests whether hybrid robustness requires its second state or simply broader state exposure.
3. **Vary state-tracking burden and decision difficulty independently.** Use history versus explicitly supplied state and easy versus difficult decision targets, with training exposure matched to each input format. A synthetic benchmark can manipulate these factors more cleanly than PGN/FEN alone. FEN still requires parsing and leaves other temporally useful information; it is not a proof that temporal recurrence has nothing to do.
4. **Make the mechanism test route-specific.** Compare controlled memory interventions with controlled KV interventions and their interaction, with matched self-swap/random or natural donor controls. Test recovery across later tokens and depth iterations. Include both decision characters and at least one nondecision character. Distinguish a single intervention from a sustained intervention that also changes downstream cache contents.
5. **If pursuing engine targets, begin with frozen-model evaluation.** Check whether more loops help on positions whose engine preference changes as search budget increases. A large best-versus-second-best gap is not by itself a measure of required calculation. Teacher distillation can improve a policy without establishing that the student performs search.

## Claim hygiene

- Say **less linearly decodable** or **less recoverable by the tested probes**, not “information was lost.” An MLP probe leaving a gap does not prove information-theoretic destruction.
- A representation being causal under a sustained edit does not identify the unique route or time at which the unedited model uses it.
- Blocking attention to the previous token leaves older cached routes intact. It establishes limited dependence on that particular edge, not exclusive reliance on temporal memory.
- Probe saturation does not show that later passes perform no useful computation; they can affect policy calibration or representations absent from the probes.
- Use “depth did not help under this architecture, objective, and budget” rather than “depth is not needed in chess.”
- Keep the estimated 0.2% marginal training FLOPs explicitly conditional on this curriculum/layout. It is not the total price of recurrence, nor a hardware-independent wall-clock guarantee.

## Suggested paper center

A strong question is: **When does temporal feedback substitute for within-token recurrent computation, and when does a model benefit from both?**

An architecture-focused paper needs a reproducible quality–cost advantage over the strongest single-axis baselines and at least one task exposing complementarity. A mechanism-focused paper can be valuable even if hybrid gains remain small: use the unified architecture as an experimental instrument to explain task-dependent reuse, reconstruction, and deployment mismatch. The current evidence is closer to supporting the second story. The chess character cycle is a useful controlled setting, but its peculiar syntax should be tested against another tokenization or task before generalizing to language models broadly.
