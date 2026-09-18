# Project Proposal: Two-Axis Recurrent Transformer for Character-Level Chess

## 1. Research objective

This project combines temporal feedback with recurrent depth in a character-level chess language model. Ordinary causal attention already carries latent information across positions. The temporal mechanism adds a specific pathway: feed a later representation from one token back into the computation of the next token. The [Full-Bandwidth Transformer paper](https://arxiv.org/abs/2608.08888) explores this kind of feedback. Across model depth, the project reuses a shared transformer core to spend more computation without adding unique core parameters, following the motivation of [Huginn](https://arxiv.org/abs/2502.05171) and its [official implementation](https://github.com/seal-rg/recurrent-pretraining).

The main hypothesis is that the mechanisms are complementary: temporal feedback carries learned computational state across tokens, while depth recurrence transforms the current state before producing a prediction. Memory content is learned without prescribing a board representation or separating memory into world state and unfinished reasoning.

During training, $U_T$ and $U_D$ count temporal and depth state updates that feed a later pass. The trajectory executes $B=\max(U_T,U_D)+1$ shared-core passes. “2D recurrence” refers to this two-dimensional update-count surface, not a Cartesian computational lattice. At inference, the model uses live temporal feedback across tokens and iterates the recurrent depth core within each token. The training update counts are not two independent inference-time loop counts.

## 2. Canonical task

The supervised task is character-level next-token prediction over PGN chess games, following Adam Karvonen's ChessGPT setup. A sequence looks approximately like:

```text
;1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 ...
```

The model receives chess text without explicit board representations, legal-move masks, FEN states, engine scores, or handcrafted chess features. Character-level PGN keeps the vocabulary small, supports comparison with ChessGPT, and permits reuse of its preprocessing and training pipeline. Move-level tokenization is outside the initial project. Start from a cleaned fork of the reference repository and inherit its chess data preparation, vocabulary, row-aligned batching, and train/validation split, verifying these during setup rather than redesigning the pipeline.

Karvonen's work studies learned chess structure and internal board representations under language-model supervision. References are the [ChessGPT world-model writeup](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html), [Emergent World Models and Latent Variable Estimation](https://arxiv.org/abs/2403.15498), and [train_ChessGPT repository](https://github.com/adamkarvonen/train_ChessGPT).

## 3. Model architecture

Use Karvonen's small eight-layer backbone with width 512 and eight attention heads. Variation A is the default: one prelude block, one temporal-integration buffer block, four shared core blocks, one temporal-source block, and one coda block, followed by final LayerNorm and tied unembedding. A and B performed nearly on par in the [10k-update comparison](../experiments/ablations/architecture_sites/REPORT.md). A is a practical default with cheaper depth iteration, not a demonstrated universal winner.

The prelude produces fixed $p=P(x)$. Each training pass mixes temporal memory into $p$, applies the buffer $Q$, mixes depth state into that result, and runs the shared core $R$. The depth source is the core output; the temporal source is the raw output of $S$. The source also feeds the coda $C$ for final prediction. Only stored states connect successive passes. Masks control writes, never reads.

```text
tokens + positions -> L1 (prelude) -> fixed p
                                          |
latest m_T from L7 -> ShiftRight --------> T -> L2 (buffer) -> q_b
                                                               |
latest h_D from L6 -------------------------------------------> D
                                                               |
                                                        L3-L6 (core)
                                                               |
                                                   h_b -> depth write
                                                               |
                                                       L7 (source)
                                                               |
                                                      temporal write
                                                               |
                                         final pass: L8 -> norm -> head
```

Read every available state on every pass. Run L7 on nonfinal passes only when a temporal write consumes it; run it once after the final pass for prediction. At live inference, L1, temporal mixing, and L2 produce a fixed anchor before looping L3–L6; L7 and L8 then run once.

Five configurable block counts determine the sites: `n_prelude`, `n_buffer`, `n_core`, `n_source`, and `n_coda`. They sum to the total backbone depth. A uses `(1,1,4,1,1)` in that order; B uses `(1,0,6,0,1)`; the historical original uses `(2,0,4,1,1)`. An empty buffer gives adjacent destinations, and an empty source segment gives coincident source candidates. This supports moving the ordered boundaries without adding a general routing graph. See the [contract](RECURRENCE_CONTRACT.md) for exact boundary definitions and checkpoint compatibility.

## 4. Canonical training-time recurrence contract

This section defines the architecture. The [recurrence contract](RECURRENCE_CONTRACT.md) records the maintained implementation reference, including concrete MVP initialization choices; the code, tests, and implementation plan follow it.

### 4.1 Counts, masks, and initialization

For a sampled pair of nonnegative integers $(U_T,U_D)$, execute $B=\max(U_T,U_D)+1$ passes. Sample Boolean write masks $M_T,M_D\in\{0,1\}^{B-1}$ with $\sum_b M_T[b]=U_T$ and $\sum_b M_D[b]=U_D$. For each mask, choose its update positions uniformly without replacement among pass outputs $1,\ldots,B-1$. The two subsets are sampled independently conditional on the counts.

Initialize $m_T=h_D=\varnothing$ at the start of each training trajectory. Masks control writes only. Every pass reads both states whenever they exist, including passes whose write masks are false. A held state retains its value and its gradient connection to the computation that produced it. Neither update counts, pass indices, masks, nor state ages are supplied as explicit model inputs; masks are used only by the execution controller.

At least one state is updated after every nonfinal pass because one update count equals $B-1$. The final pass has no counted state writes. For $(0,0)$, both masks are empty and the model executes one core pass.

### 4.2 Temporal mixing

If temporal memory exists, read $r_b=\operatorname{ShiftRight}(m_T)$ and compute $a_b=T_\theta(p,r_b)$. Otherwise use $a_b=p$. The temporal write mask does not appear in this read condition.

The MVP mixer uses feature-wise gates $(\alpha_b,\beta_b)=\sigma(G_\theta([N_r(r_b);N_p(p)]))$ and computes $T_\theta(p,r_b)=\alpha_b\odot W_mN_r(r_b)+\beta_b\odot W_pN_p(p)$. The coefficients need not sum to one. The gate reads normalized source states before the value projections. Separate learned projections $W_m$ and $W_p$ map those normalized sources into the mixed space. Reuse the same normalized sources for gate and value computation.

The shift satisfies $\operatorname{ShiftRight}(m_T)[0]=0$ and $\operatorname{ShiftRight}(m_T)[t]=m_T[t-1]$ for $t>0$. At a position with no valid predecessor, bypass the temporal mixer and use the prelude representation directly. This distinguishes absent feedback from a valid memory vector whose numerical value happens to be zero.

### 4.3 Depth mixing and core computation

First compute $q_b=Q(a_b)$ through the buffer; $Q$ is L2 in A and identity if empty. If depth state exists, compute $z_b=D_\theta(h_D,q_b)$ with $D_\theta(h_D,q_b)=W_hN_h(h_D)+W_aN_a(q_b)$. Otherwise use $z_b=q_b$. The depth mixer has no explicit sigmoid gate in the MVP. Its read condition depends only on state availability, not on the depth write mask. Execute $h_b=R(z_b)$ on every pass.

### 4.4 Writes, output, and gradients

Default A uses one dedicated temporal-source block $S$. Configurable variants may use a longer source segment or identity for coincident sources. It consumes the core output and produces the raw residual-stream state used as temporal memory, before the coda or final normalization. It is an ordinary causal transformer block, including its standard internal normalization, attention, MLP, and residual connections. Its parameters are distinct from the core and coda and shared across its training-time invocations.

After a nonfinal pass, set $h_D\leftarrow h_b$ if $M_D[b]=1$ and set $m_T\leftarrow S(h_b)$ if $M_T[b]=1$. Otherwise hold the corresponding state unchanged, including leaving it absent if it has not yet been written. Both writes derive from the current core output and occur after all reads, but they store different representations. Evaluate $S$ on a nonfinal pass only when a temporal write is scheduled. Store its output directly, without an additional writer projection or write-time normalization. Depth feedback remains the pre-source core output; in A, source output never replaces depth state.

After pass $B$, compute logits from $C(S(h_B))$ through final normalization and the language-model head. The final source and coda execution creates no counted training update. Initially train with final-pass next-character cross entropy only and backpropagate through the complete trajectory, including every reuse of held states. Do not detach memory or depth state, truncate gradients, or run early passes without gradients.

## 5. Limiting cases and held states

| Update counts | Execution |
| --- | --- |
| $(0,0)$ | One core pass; both recurrent states remain absent. Equivalent to the ordinary prelude/buffer/core/source/coda transformer. |
| $(U_T>0,0)$ | Temporal-only recurrence; depth state remains absent. |
| $(0,U_D>0)$ | Depth-only recurrence; temporal memory remains absent. |
| $(U_T>0,U_D>0)$ | Both states are written and consumed by later passes. |

These are numerical equivalence requirements. For example, $(4,0)$ executes five core passes with evolving temporal inputs and no depth feedback. Shared weights do not imply identical computations across those passes.

For $(4,1)$, if the sole depth write follows pass one, passes two through five combine evolving temporal feedback with that frozen depth state. A later core output replaces only the states selected for writing. Different update frequencies can therefore expose each mixer to states produced at different points in the trajectory. More updates do not, by themselves, imply that one connection dominates the other.

## 6. Parallel Jacobi-style training

Each training pass processes the complete teacher-forced sequence in parallel. At position $t$, temporal feedback comes from position $t-1$ in the latest previously written memory sequence. If the temporal state is held, later passes reread that same sequence with the same one-position shift; holding does not repeatedly shift the stored memory.

This is a Jacobi-style forward computation: a position consumes a state completed in an earlier pass rather than waiting for a same-pass sequential update from its predecessor. The forward pass is exact for this defined graph. Full backpropagation through that graph remains part of training; Jacobi updates and backpropagation describe different aspects of the computation.

Live feedback generation uses a different forward execution, described in Section 14. Training-graph recomputation and live feedback must be evaluated separately rather than assumed numerically equivalent. The implementation precedents are [multipass-transformer-training](https://github.com/PeterBjerreHansen/multipass-transformer-training) and [multipass-transformer-memory](https://github.com/PeterBjerreHansen/multipass-transformer-memory).

## 7. Random update timing

For each training microbatch, sample $(U_T,U_D)\sim P(U_T,U_D)$ and randomize the write positions conditional on those counts. Random placement is part of the MVP. Its purpose is to avoid a fixed timing relationship between the mechanisms and encourage robustness to held or refreshed states. That robustness is a hypothesis, not a guarantee of randomization.

For $(U_T,U_D)=(2,4)$, two possible schedules are:

```text
Pass:                  1  2  3  4  5
Temporal write, A:     1  0  1  0  -
Temporal write, B:     0  1  0  1  -
Depth write, either:   1  1  1  1  -
```

Entries describe writes after the indicated pass, not permission to read during it. Under schedule A, passes two and three read the memory written after pass one; passes four and five read the memory written after pass three. Under schedule B, temporal feedback is absent in passes one and two, then remains available from pass three onward. There are no random placement choices for a mask with zero writes or with writes at every eligible position. Checkpoint the sampler's RNG state for reproducibility.

## 8. Development grid

Support arbitrary nonnegative integer update counts from the beginning. The initial nine-cell grid uses $U_T,U_D\in\{0,1,3\}$, covering ordinary execution, both single-axis families, balanced hybrids, and asymmetric hybrids. This preserves the original one-, two-, and four-pass budgets while adopting the corrected write-only semantics.

| $U_T\backslash U_D$ | 0 | 1 | 3 |
| --- | ---: | ---: | ---: |
| 0 | 0.10 | 0.12 | 0.04 |
| 1 | 0.12 | 0.26 | 0.08 |
| 3 | 0.04 | 0.08 | 0.16 |

This configurable starting distribution gives $P(B=1)=0.10$, $P(B=2)=0.50$, $P(B=4)=0.40$, and $E[B]=2.7$. Once the surface behaves sensibly, expand to $U_T,U_D\in\{0,\ldots,7\}$, giving up to eight core passes. Neither support nor probabilities belong in model code.

## 9. Experimental methodology

### 9.1 Within-checkpoint surface

For a jointly trained checkpoint, measure $L(U_T,U_D)$ under the training-graph execution. This describes which update settings the checkpoint can use effectively, not which architecture is intrinsically better. Report training exposure for each cell and evaluate multiple fixed write-mask seeds where distinct schedules exist. Report the mean and standard deviation without confusing mask variance with training-seed variance. Label live-feedback generation results separately from this surface.

### 9.2 Between-architecture comparison

The architectural comparison requires separately trained ordinary, temporal-only, depth-only, and hybrid models. It asks what each mechanism contributes and whether the combination improves the performance–compute frontier over either alone. Substantial component-baseline training should begin only after initial hybrid experiments show useful recurrent behavior. For inexpensive architecture comparisons, prefer at least three independently initialized training runs; evaluate mask variation within each run separately.

## 10. Parameter controls

Report both same-backbone and parameter-matched comparisons. The same-backbone regime keeps embedding width $D$ and prelude, buffer, core, source, and coda block counts $L_P,L_Q,L_R,L_S,L_C$ identical, while allowing the recurrent models their gates, read-side normalizations, and value and depth projections. Keep the source block in the prediction path for every same-backbone baseline, including ordinary and depth-only models, even when no temporal feedback is used. The default 1/1/4/1/1 partition totals eight blocks; buffer and source are allocated within those eight blocks. The parameter-matched regime constructs an ordinary reference transformer with approximately the same total trainable parameter count. These answer different questions and neither replaces the other.

## 11. Compute controls

Report the structural proxy $L_P+B(L_Q+L_R)+(U_T+1)L_S+L_C$ alongside complete analytic or profiler-based FLOP estimates and measured throughput. Here $L_Q=L_S=1$ for default A. The buffer runs each training pass and once per token before live depth iteration. The source runs $U_T+1$ times: once per nonfinal temporal write and once on the final prediction path. The coda runs once. At inference with $J$ core calls, the corresponding block count is $L_P+L_Q+J L_R+L_S+L_C$. Complete estimates must include attention, MLPs, temporal gates, read-side normalization and value projections, depth projections and normalization, and output computation.

Update counts alone do not determine mixer cost: every available state is read on every later pass, including passes where it is held. The position of the first write therefore affects the number of mixer applications. Account for actual schedule execution, not just $U_T$ and $U_D$. Label equal-step, equal-character, equal-parameter, and equal-FLOP comparisons explicitly.

## 12. Curriculum and supervision development

Begin with the fixed update distribution. Introduce a simple schedule toward larger $B$ only if experiments suggest that additional passes become useful later in training. Intermediate-pass supervision is a separate later experiment: it may be more compute-efficient to train on fewer samples with more passes and additional prediction losses. It is not required by the initial architecture.

Before adaptive scheduling, estimate how training on one region affects other regions. From a common checkpoint, run short controlled training interventions on the $\{0,1,3\}^2$ grid and measure changes in evaluation loss. Define $G_{a\rightarrow b}$ as improvement at evaluation setting $b$ after training on setting or region $a$, relative to the specified control. Normalize by intervention compute where useful. Study transfer across low and high compute, temporal-heavy and balanced settings, the need for off-diagonal exposure, and whether $(3,3)$ provides broad benefit.

Evaluating every cell reveals the current loss surface, but does not reveal the training gains of every unchosen intervention. It therefore does not, by itself, establish a full-information allocation problem. Consider Hedge or another adaptive method only after specifying how its reward is observed or estimated, with the objective of improving the eventual performance–compute frontier. Raw validation loss is not an adequate allocation reward by itself.

## 13. Evaluation priorities

During generation, stop at the first completed illegal or malformed move and log the generated text, failure reason, offending move, preceding board position, and number of legal moves completed. Do not retry or repair failures. Character prefixes are validated when a move boundary is reached. Log valid game endings and generation/context-limit truncations separately; an unfinished move at the limit is not a completed illegal move.

The MVP prioritizes validation NLL, character accuracy, PGN validity, legal-move rate, legal continuation length, valid termination, and recurrence-surface behavior. Playing strength follows later. Gate activity, state norms, and gradients are useful diagnostics but do not establish that memory content improves predictions. Board-state probes and mechanistic analysis are deferred until there is a reproducible recurrence effect worth explaining.

## 14. Live feedback inference and later optimization

At token position $t$, compute the prelude representation for $x_t$ and mix it with incoming memory $m_{t-1}$. Run the buffer on the temporal mixture, then hold that memory and the resulting depth anchor fixed throughout the token's depth computation. Initialize depth state as absent, execute the core once, then use successive core outputs as depth state for any additional iterations. The number of core calls is an inference depth budget, separate from the training update counts.

After the final depth iteration, run the temporal-source block once and store its raw output as $m_t$. Feed that same output through the coda, final normalization, and head to produce logits. For A, the source is outside the depth loop and is not rerun as depth state is refined; with an empty source segment, the final core output is emitted directly. There is no additional temporal writer transformation. Sample $x_{t+1}$ from those logits; its processing consumes $m_t$. No temporal refresh loop occurs within a token. The first token without incoming memory bypasses temporal mixing. This combines live feedback generation with ordinary recurrent-depth execution; the [feedback-inference reference](https://github.com/PeterBjerreHansen/multipass-transformer-memory/blob/main/docs/FEEDBACK_INFERENCE.md) provides the temporal precedent.

Implement fixed-depth semantics before optimizing KV caches for the prelude, core, source, and coda, temporal memory storage, or loop execution. Maintain separate reference paths for exact training-graph recomputation and live-feedback execution. Optimized logits must match the reference for the same semantics; divergence between the two execution modes is a measurement, not automatically a cache bug. Specify prompt prefill, memory handoff, and depth-indexed cache behavior before implementing optimized decoding.

The model receives no explicit update count or iteration index. Shared refinement may learn fixed-point behavior, but no convergence objective or monotonic improvement requirement is imposed initially. A learned exit gate is a later extension after fixed-depth inference works. State residuals and logit changes can be studied as diagnostics without defining the initial stopping rule. Related looped-model references include [Huginn](https://arxiv.org/abs/2502.05171) and [Ouro](https://arxiv.org/abs/2510.25741).

## 15. Self-play reinforcement learning

Self-play is a later research stage after stable supervised checkpoints. Begin with searchless self-play so MCTS does not introduce another compute mechanism simultaneously with recurrence. Compare improvement in playing strength from each architecture's supervised checkpoint, alongside absolute strength. This asks whether recurrence supports continued improvement through interaction as well as imitation of human games.

## 16. Central research question

The project asks whether a jointly trained model can use persistent latent state across tokens and repeated latent computation within tokens productively under controlled parameter and compute comparisons. Chess supplies a structured sequential task while retaining a pure language-modeling objective. The initial work should establish a useful signal before committing to extensive architecture comparisons, curriculum research, self-play, or interpretation.
