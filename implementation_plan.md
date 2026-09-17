# Implementation Plan: Two-Axis Recurrent Character-Level Chess Transformer

## Governing contract

The [project proposal](proposal.md#4-canonical-training-time-recurrence-contract) defines the architecture. Before implementing recurrence, extract that contract into `docs/RECURRENCE_CONTRACT.md` and use it as the maintained reference for code, tests, diagrams, and configuration terminology. The execution order is temporal mixing, depth mixing, shared core, then state writes. Depth writes store core output; temporal writes store the raw output of a dedicated temporal-source transformer block placed between the core and coda. Write masks never control reads.

Keep the implementation small enough to inspect the recurrent path in one file. The external repositories are references, not training frameworks to import wholesale. This plan preserves the staged approach: reproduce ordinary ChessGPT, establish a hybrid signal, and only then expand experiments and inference optimization.

## References

| Area | Sources |
| --- | --- |
| Chess and baseline | [train_ChessGPT](https://github.com/adamkarvonen/train_ChessGPT), [nanoGPT](https://github.com/karpathy/nanoGPT), [ChessGPT writeup](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html), [chess world-model paper](https://arxiv.org/abs/2403.15498) |
| Temporal recurrence | [multipass-transformer-training](https://github.com/PeterBjerreHansen/multipass-transformer-training), [multipass-transformer-memory](https://github.com/PeterBjerreHansen/multipass-transformer-memory), [memory contract](https://github.com/PeterBjerreHansen/multipass-transformer-memory/blob/main/docs/RECURRENT_MEMORY.md), [architecture contracts](https://github.com/PeterBjerreHansen/multipass-transformer-memory/blob/main/docs/ARCHITECTURES.md), [feedback inference](https://github.com/PeterBjerreHansen/multipass-transformer-memory/blob/main/docs/FEEDBACK_INFERENCE.md) |
| Related architectures | [Full-Bandwidth Transformer](https://arxiv.org/abs/2608.08888), [Recirculation](https://arxiv.org/abs/2608.17981), [Huginn](https://arxiv.org/abs/2502.05171), [Huginn implementation](https://github.com/seal-rg/recurrent-pretraining), [Ouro](https://arxiv.org/abs/2510.25741) |

## Stage 0: Bootstrap and freeze research contracts

Use a fork of [Karvonen's train_ChessGPT](https://github.com/adamkarvonen/train_ChessGPT) as the starting codebase. Record the upstream commit and retain its license. Keep the chess model, training loop, character vocabulary, preparation script, block-aligned batch loader, sampling support, and configuration mechanism. Remove unrelated datasets, GPT-2 import paths, and unused notebooks or examples after checking that the retained path does not depend on them. Make small, traceable changes rather than rewriting the pipeline.

Create `baseline-chessgpt` and `mvp-2d-recurrence` from the common cleaned baseline commit. The ordinary baseline uses eight layers, matching the later 2/4/1/1 partition in Stage 3. Freeze its training code and configuration after smoke tests, and continue recurrent development on the MVP branch.

### Data defaults

Inherit the reference [preparation script](https://github.com/adamkarvonen/train_ChessGPT/blob/master/data/lichess_hf_dataset/prepare.py): `adamkarvonen/chess_games`, `lichess_6gb_blocks.zip`, the supplied character vocabulary, `uint8` storage, and the shuffled 99% training / 1% validation split with seed 2357. The source rows contain 1,024 characters. Preserve the reference 1,023-character input context and one-character-shifted targets within each row. The batch loader selects row starts using a stride of 1,024; changing context length must not silently change that storage stride.

Copy the retained preparation path into `data/chess_v1/` with `train.bin`, `val.bin`, `meta.pkl`, and `manifest.json`. Record upstream and dataset revisions, file identity, split settings, vocabulary, row and character counts, preprocessing commit, and output hashes. Check that rows have the expected length and start marker, characters belong to the vocabulary, targets align correctly, and exact rows are not shared between training and validation. Inspect how game boundaries appear inside source rows and record that behavior; do not invent new packing or split rules without an observed reason. The reference split is over stored rows, so do not claim it guarantees game-level separation without checking the source data. For later recurrence, bypass temporal feedback at each row's first position; otherwise retain the reference's within-row context behavior initially.

Once verified, keep this data version fixed across comparisons. Use a small reproducible subset for setup and smoke tests before processing or training on the full archive. Adapt old dependency calls only where needed to make the retained reference pipeline run, and record the working package versions.

### Evaluation and checkpoints

Stop generation at the first completed illegal move or malformed move. Preserve the generated text and log the stopping reason, offending move text, board position before failure, and number of legal moves completed. Count the failed completed attempt in the move-attempt denominator. Do not retry, repair, or filter the model's output. Because generation is character-level, wait for a move boundary before validating a candidate; an unfinished prefix is not yet an illegal move.

Also stop on a valid game ending or a configured generation/context limit. Log an unfinished move at the limit as truncation, separately from an illegal completed move. Fix the prompts, random seed, decoding settings, and limits in evaluation configuration. Report validation NLL, character accuracy, legal continuation length, completed-move legality, PGN parse success, and termination reason. Use `python-chess` to check moves against the evolving board; Stockfish and board probes are not prerequisites.

Record the Git commit, configuration, dataset manifest hash, package versions, world size, optimizer state, and random-generator states needed to resume with every checkpoint. Save the recurrence sampler state when that sampler is introduced. Verify resume in the smoke test.

## Stage 1: Ordinary ChessGPT reproduction

Use the reference [chess training configuration](https://github.com/adamkarvonen/train_ChessGPT/blob/master/config/train_shakespeare_char.py), despite its inherited Shakespeare filename: eight layers, eight heads, width 512, context length 1,023, and zero dropout. Preserve learned position embeddings, the original transformer block, final LayerNorm, and the bias-free language-model head. Keep the reference tying of the token embedding and unembedding weights. This stage remains an ordinary eight-block transformer; the named architecture partition is introduced later.

Inherit the optimizer and schedule defaults from that configuration and `train.py`, including learning rate 0.0003, minimum learning rate 0.00003, and 2,000 warmup iterations. The upstream 600,000-iteration schedule is a reference setting, not a prerequisite for seeing an initial signal or an instruction to launch a full run immediately. Keep a short smoke-test configuration and an explicitly recorded baseline run configuration. Adjust batch size, accumulation, device, precision, and compilation for the actual hardware while recording the effective batch size and every change from upstream. Rename chess configuration and output paths for clarity.

Confirm that loss decreases, validation and checkpoint resume work, samples become recognizable, and stop-on-failure evaluation records legal continuations correctly. Then launch the initial baseline run and tag its code and configuration. Avoid unrelated architecture changes such as RoPE, RMSNorm, or SwiGLU. Continue recurrence development while the baseline trains.

## Stage 2: Define the canonical recurrence contract

### State and scheduling

For sequence batch size $N$, length $T$, and embedding width $D$, the prelude output and each available recurrent state have shape $N\times T\times D$. Let $p=P(x)$ remain fixed through the trajectory. Initialize temporal memory and depth state as absent.

Sample nonnegative update counts $(U_T,U_D)$ and execute $B=\max(U_T,U_D)+1$ core passes. The Boolean write masks have length $B-1$, with exactly $U_T$ and $U_D$ true entries. Sample each subset uniformly without replacement among the first $B-1$ pass outputs, independently conditional on the counts. The final output has no counted recurrent writes. For $(0,0)$, masks are empty and execution consists of one core pass.

Read every available state on every pass. A false write mask holds that state unchanged and does not bypass its mixer. Held tensors remain connected to their original producer in the gradient graph. Update counts, masks, iteration indices, and state ages are not explicit inputs to the learned model. Only the execution controller uses the schedule.

### Mixer and write equations

When temporal memory exists, set $r_b=\operatorname{ShiftRight}(m_T)$ and $a_b=T_\theta(p,r_b)$; otherwise set $a_b=p$. Use gates $(\alpha_b,\beta_b)=\sigma(G_\theta([N_r(r_b);N_p(p)]))$ and $T_\theta(p,r_b)=\alpha_b\odot W_mN_r(r_b)+\beta_b\odot W_pN_p(p)$. At positions without valid predecessor memory, bypass the mixer exactly. The gate reads normalized sources before their value projections; the value path uses separate learned $W_m$ and $W_p$ projections of those same normalized sources.

When depth state exists, set $z_b=W_hN_h(h_D)+W_aN_a(a_b)$; otherwise set $z_b=a_b$. Always compute $h_b=R(z_b)$. Write depth state as $h_D\leftarrow h_b$ when scheduled. Use a dedicated normal causal transformer block $S$ after the core and before the coda $C$. A scheduled temporal write computes $m_T\leftarrow S(h_b)$ and stores the raw output, without an additional writer projection or write-time normalization. Its parameters are distinct from the core and coda and shared across its training invocations. After the final pass, compute $C(S(h_B))$, final normalization, and the LM head; do not count this final source activation as another training update.

### Canonical pseudocode

The pseudocode samples a pair and two write masks per microbatch, using the checkpointed sampler RNG. `uniform_subset_mask` chooses exactly the requested number of positions uniformly without replacement; zero-length masks are valid. Pass indices are zero-based and masks have length `rounds - 1`. Reads depend on state availability, while only writes consult the masks. The temporal mixer receives predecessor validity separately from memory values so that a legitimate zero-valued memory is still read. Under DDP, broadcast the sampled pair and masks before this trajectory executes.

```python
u_t, u_d = sample_update_pair(pair_distribution, sampler_rng)
rounds = max(u_t, u_d) + 1
temporal_write_mask = uniform_subset_mask(rounds - 1, u_t, sampler_rng)
depth_write_mask = uniform_subset_mask(rounds - 1, u_d, sampler_rng)

p = prelude(token_embedding + position_embedding)
temporal_state = None
depth_state = None

for b in range(rounds):
    # Read both states whenever available, regardless of this pass's write masks.
    if temporal_state is None:
        anchor = p
    else:
        shifted_memory, valid = shift_right_with_validity(temporal_state)
        anchor = temporal_mixer(p, shifted_memory, valid)

    if depth_state is None:
        core_input = anchor
    else:
        core_input = depth_mixer(depth_state, anchor)

    h = recurrent_core(core_input)

    # False masks leave the stored tensors (and their gradients) unchanged.
    if b < rounds - 1:
        if depth_write_mask[b]:
            depth_state = h
        if temporal_write_mask[b]:
            temporal_state = temporal_source(h)  # raw source-block output

source_output = temporal_source(h)
logits = lm_head(final_norm(coda(source_output)))
loss = cross_entropy(logits, targets)
```

No additional carry of `h` bypasses the stored states. Holding a state must neither detach it nor mutate the historical tensor in place. Backpropagate through the complete execution, including repeated reads of held tensors.

### Exact reductions

| Counts | Required reduction |
| --- | --- |
| $(0,0)$ | Ordinary prelude/core/source/coda transformer; neither state is written or read. |
| $(U_T>0,0)$ | Temporal-only execution; depth state remains absent. |
| $(0,U_D>0)$ | Depth-only execution; temporal memory remains absent. |
| $(U_T>0,U_D>0)$ | Hybrid execution; both states are consumed after their first writes. |

Verify reductions numerically with matched weights and controlled dropout. For $(4,0)$, five passes can perform different computations because temporal inputs evolve. For $(4,1)$ with the depth write after pass one, subsequent passes read that fixed depth state alongside refreshed temporal memory. Do not equate update counts with read counts.

## Stage 3: Refactor nanoGPT into prelude, core, source, and coda

Partition the eight-layer backbone into two prelude blocks, four shared recurrent-core blocks, one dedicated temporal-source block, and one coda block. Retain width 512 and eight heads. At $(0,0)$ this is the ordinary eight-block transformer, followed by final LayerNorm and the tied unembedding matrix. Preserve configurable prelude, core, and coda counts; the MVP source is one normal causal transformer block, $L_S=1$, with the same width and block design as the backbone. It has its own parameters and is outside the depth loop. Remove the earlier coda-source-index configuration: the source is now an explicit module.

Add token and learned position embeddings once before the prelude; do not re-add them inside recurrence. Before adding mixers, verify that one core call followed by source and coda matches the equivalent ordinary stack within floating-point tolerance. Keep the source in the final prediction path even for ordinary and depth-only models. Map baseline blocks 1–2 to the prelude, 3–6 to the core, 7 to the source, and 8 to the coda, preserving block order and weights for the equivalence check.

## Stage 4: Implement the stochastic schedule sampler

Use a `RecurrenceSchedule` carrying `u_t`, `u_d`, `rounds`, `temporal_write_mask`, and `depth_write_mask`, plus a checkpointable `RecurrenceScheduleSampler`. Accept arbitrary nonnegative integer supports and a pair-probability matrix. Do not hard-code the pilot support or use independent Bernoulli dropout: the two update counts must be exact.

A false mask entry means no write after that pass. The current state remains available for every subsequent read. At least one write follows every nonfinal pass; this requirement does not apply to the final pass, which has no mask entry. Empty masks for the one-pass case are valid.

Checkpoint the support, pair distribution, sampler RNG state, draw count, pair histogram, and pass-count histogram. Resume must reproduce the schedule sequence. Under DDP, all ranks execute the same schedule per microbatch, through a rank-zero broadcast or equivalent synchronized mechanism. Random timing aims to discourage dependence on a fixed relationship between updates; it does not guarantee convergence or robustness.

## Stage 5: Implement temporal recurrence

Use $\operatorname{ShiftRight}(M)[0]=0$ and $\operatorname{ShiftRight}(M)[t]=M[t-1]$ for $t>0$. Shift the latest stored memory for each read without modifying it. Repeated reads of a held memory must not shift it farther through the sequence. Track predecessor validity and bypass the mixer where feedback is absent, including the first position.

Every core pass processes the full teacher-forced sequence in parallel. A temporal write refreshes the complete memory sequence; subsequent passes read it whether or not they also write. Describe this as exact execution of the defined Jacobi-style training graph, with full backpropagation through all passes. It is distinct from sequential live feedback generation.

Store the raw temporal-source output as memory. Run this block on each nonfinal core output selected for a temporal write, then run it once on the final core output and pass the result through the coda. Keep gradients through the source and every later read of its stored activation. Its standard transformer internals remain intact, but there is no additional `TemporalWriter` module, write projection, or write-time normalization. During live inference it runs once per token after all depth iterations, not inside the depth loop.

Implement the gate and projected-value equations from Stage 2, with feature-wise coefficients initialized approximately to $\alpha=0.1$ and $\beta=0.9$. Do not constrain their sum to one. Use separate bias-free $D\rightarrow D$ value projections $W_m$ and $W_p$; identity initialization is a starting choice to record in configuration. The gate sees normalized sources before these projections. Reuse each normalized source for gate and value computation. An absent memory bypasses the entire mixer and returns raw $p$, including bypassing $W_pN_p(p)$.

A held memory's normalized and projected forms can be reused within a forward trajectory if gradients and any stochastic operations are preserved. Treat this as an optimization of the direct equations. At inference, compute the fixed temporal anchor once before depth iteration. Do not cache training activations across optimizer steps.

## Stage 6: Implement depth recurrence

Use $D_\theta(h_D,a)=W_hN_h(h_D)+W_aN_a(a)$ whenever depth state exists. The anchor $a$ is the result of temporal mixing when temporal memory is available, or the prelude representation otherwise. A false temporal write mask does not revert the anchor to raw $p$.

Do not add a sigmoid interpolation, per-loop gate, exit gate, or unconditional extra carry of the last core output in the MVP. The residual transformer core and learned projections provide the initial transformation. Use Huginn as a reference for shared core reuse and repeated access to an input anchor, without importing its large-scale training stack.

## Stage 7: Integrate the hybrid trajectory

Use the following configurable pilot distribution over $U_T,U_D\in\{0,1,3\}$. It retains the one-, two-, and four-pass budgets under the new update convention; it does not retain the old read-masking semantics.

| $U_T\backslash U_D$ | 0 | 1 | 3 |
| --- | ---: | ---: | ---: |
| 0 | 0.10 | 0.12 | 0.04 |
| 1 | 0.12 | 0.26 | 0.08 |
| 3 | 0.04 | 0.08 | 0.16 |

The distribution gives $P(B=1)=0.10$, $P(B=2)=0.50$, $P(B=4)=0.40$, and $E[B]=2.7$. Keep it in configuration. Train initially with final-pass next-character cross entropy only. Use full gradients through the prelude, every core pass, both mixers, all writes and held-state reads, and the output stack. Exclude truncated backpropagation, no-gradient early passes, intermediate losses, convergence losses, exit losses, and curriculum logic from the initial run.

Memory content remains learned. Do not add board-state supervision or prescribe what each connection represents. Fixed-point behavior is a possible outcome of shared refinement, not an enforced property. Earlier-pass prediction losses remain a later compute-efficiency experiment comparing fewer samples with more passes and deep supervision against the initial training regime.

## Stage 8: Verify the recurrence contract

Use small deterministic models and controlled dropout for semantic tests. The following checks protect the computation, rather than merely checking configuration fields.

| Area | Required checks |
| --- | --- |
| Schedule | Nonnegative counts; $B=\max(U_T,U_D)+1$; masks of length $B-1$; exact true counts; at least one write after each nonfinal pass; empty masks at $(0,0)$; correct empirical pair frequencies and uniform conditional subsets. |
| Holds and reads | An unwritten state remains absent; a held state retains its value and continues to reach its mixer. Perturbing a held state changes a later computation with a deliberately sensitive test mixer. Cover both temporal-held and depth-held schedules. |
| First writes | A state is absent before its first write and read on every subsequent pass. Test both early and latest-eligible first writes; a write after pass $B-1$ must be consumed by pass $B$. |
| Final pass | No counted writes occur after the last pass. Prediction uses the source and coda on the final core output. |
| Feedback source | Count exactly $U_T+1$ source executions and one coda execution per training trajectory. Verify raw source-output capture, reuse of source weights across invocations, and distinct source/core/coda parameters. At inference, assert one source call per token regardless of the positive depth budget. |
| Temporal mixer | Verify that gates consume normalized unprojected sources, values use separate projections, and missing memory bypasses the full mixer. |
| Ordering | Use noncommuting test mixers to distinguish temporal-then-depth from depth-then-temporal execution. Depth stores the current core output; temporal memory stores the source-block output derived from that same core output, after all reads. |
| Causality | Check exact right-shift indices, boundary bypass, and that repeated reads do not accumulate shifts. Future characters cannot affect earlier logits. Perturbing stored memory at position $t$ cannot affect the temporal read at position $t$ in that pass. |
| Reductions | Verify ordinary, temporal-only, and depth-only numerical equivalence. Cover an asymmetric hybrid with one state held across several reads. |
| Gradients | Compare a held-state trajectory against an explicit unrolled reference and confirm gradients reach the original state-producing core or temporal-source execution through later reads. Check finite gradients through all used modules; do not demand gradients through unused branches in reductions. |
| Support and resume | Execute every pilot cell and several valid schedules with finite activations, losses, gradients, and correct shapes. Check model, optimizer, and sampler resume, including DDP schedule agreement. |

Add tests for inference semantics when that implementation is introduced. These should compare optimized and reference implementations of the same mode, not require equality between live feedback and the training graph.

## Stage 9: Hybrid signs-of-life experiments

Implement `evaluation/recurrence_grid.py` to evaluate all nine cells of $\{0,1,3\}^2$ using training-graph execution. For each cell, report NLL, character accuracy, actual pass count, estimated FLOPs, training probability, and mean and standard deviation over fixed mask seeds. Use distinct placements where they exist; diagonal and single-axis cells have no placement variation. Label generation results with their execution mode, prefill procedure, and depth budget.

Look for stable four-pass execution, useful prediction changes under recurrence, some benefit from additional computation, healthy state norms and gradients, and valid chess generation. Do not require monotonic improvement at every setting. Nonzero gates or gradients show that a path participates in computation, not that its memory content is useful; controlled feedback interventions can distinguish these when interpreting the signal. Establish this initial signal before committing to substantial component-baseline training.

## Stage 10: Expand the update grid

Once the pilot behaves sensibly, expand to $U_T,U_D\in\{0,\ldots,7\}$, giving up to eight core passes. Reuse the same sampler and contract with configuration changes only. Reevaluate the surface and examine behavior across adjacent update counts. The generic implementation must also accept other nonnegative supports, including the $(2,4)$ and $(4,1)$ explanatory examples.

## Stage 11: Component baselines

Train ordinary, temporal-only, depth-only, and hybrid models as restrictions of the same implementation where possible. In the same-backbone regime, keep embedding width and physical prelude, core, source, and coda block counts identical while allowing each model the recurrent modules it needs. Separately construct an ordinary reference with approximately matching total parameters.

Use multiple independent training seeds for affordable architecture comparisons, preferably at least three. Report between-run variance separately from within-run mask variance. These comparisons follow the hybrid pilot rather than becoming prerequisites for seeing an initial signal.

## Stage 12: Complete compute accounting

Log $U_T$, $U_D$, $B$, shared-core block applications, characters processed, and optimizer steps. Report $L_P+B L_R+(U_T+1)L_S+L_C$ as a structural proxy, with $L_S=1$ for the MVP. The source executes once per nonfinal temporal write and once on the final prediction path; the coda executes once. Include complete analytic or profiler-based estimates covering the prelude, core attention and MLPs, temporal gates, read normalizations and value projections, depth projections and normalization, all source executions, the coda, and the head. Measure characters per second, optimizer steps per second, and wall-clock training time.

Account for actual reads as well as writes. A state written early is read more often than one first written late, even with the same update count. Mask placement can therefore change mixer cost while leaving core cost unchanged. Label comparisons by whether they match steps, characters, parameters, or FLOPs. For live-feedback inference with $J$ core calls, report $L_P+J L_R+L_S+L_C$ and the actual mixer operations separately from training update counts.

## Stage 13: Curriculum and supervision research

First characterize the fixed distribution. If higher-pass execution becomes useful only later in training, test a simple interpretable schedule toward larger $B$. Study intermediate-pass supervision separately as a possible compute-efficiency tradeoff; compare sample count, pass count, prediction losses, and total compute explicitly.

Before adaptive allocation, clone a common checkpoint, train short controlled interventions concentrated on selected cells or regions, and evaluate all pilot cells. Estimate $G_{a\rightarrow b}=L_b(\theta_{\mathrm{before}})-L_b(\theta_{\mathrm{after\ training\ on\ }a})$, or differences against a specified matched control, and normalize by intervention compute where useful. Examine transfer between low and high compute, temporal-heavy and balanced settings, off-diagonal exposure, and broad transfer from $(3,3)$.

Only then specify an adaptive scheduler's objective, reward estimator, compute normalization, exploration floor, update frequency, and stability constraints. Evaluating the whole loss surface does not reveal the gains of unchosen training actions. Do not label the allocation problem full-information solely on that basis. Hedge or another method is a later option if its required feedback can be justified; raw validation loss alone is not the reward.

## Stage 14: Live feedback inference and optimization

### Fixed-depth live execution

For the current token $x_t$, compute its prelude representation using the causal prefix context and mix it with incoming memory $m_{t-1}$. Hold this memory and the resulting anchor fixed while the depth core iterates. Reset depth state to absent for each new token. The first core call uses the anchor directly; every additional call uses the preceding depth output mixed with that same anchor.

After the final core call, run the temporal-source block once and store its raw output as $m_t$. Feed that same output into the coda, final normalization, and head to obtain logits for $x_{t+1}$. Neither the source nor coda is inside the depth loop. No additional writer projection is used. Sample the next token and process it with that emitted memory. This final temporal emission is an inference operation, not an extra counted training update. There are no temporal refinement passes within a token. A token without incoming memory bypasses temporal mixing.

### References, prefill, and caching

Maintain two reference paths: exact full-sequence recomputation of the training graph for a specified write schedule, and a slow live-feedback implementation preserving historical temporal memories while looping depth within each token. Disable dropout for numerical comparisons. Test each optimized path against its corresponding reference. Measure divergence between the two modes without treating it as an implementation failure by itself.

Before KV optimization, explicitly define prompt prefill, which memory is handed to continuation, positional handling, and the historical attention context used by each depth iteration. The temporal reference repository's handoff behavior is a precedent, not a substitute for specifying the hybrid. Do not silently combine KV state from one execution with memory from another. Cache identity must account for the relevant layer and recurrent execution; variable depth across tokens requires its own policy before adaptive stopping is enabled.

Then implement and verify prelude, temporal-source, and coda KV caching, recurrent-core caching, temporal memory storage, and efficient loop execution. Keep the slow reference available as the correctness oracle for each mode.

### Later exit gate

Begin with a fixed positive number of core calls per token. A learned exit gate is a later extension after fixed-depth inference works. Do not impose a convergence loss or feed update counts, iteration indices, or state ages into the initial model. State residuals and logit changes may be studied as diagnostics. Stopping determines the final core state supplied to the temporal source. The source output becomes outgoing memory and feeds the coda for prediction.

## Stage 15: Self-play reinforcement learning

Start from stable, documented supervised checkpoints and use searchless self-play initially. Implement a character-to-action interface that accumulates characters until a complete SAN move is produced. Before RL training, specify illegal-string handling, move completion, draws, resignation, terminal rewards, policy objective, and opponent sampling. Keep MCTS outside the first comparison. Report post-RL minus pre-RL playing strength alongside absolute strength.

## Stage 16: Interpretability

Defer board-state probes, gate analysis, latent interventions, and mechanistic studies until recurrence produces a reproducible effect worth interpreting. Establish that both mechanisms affect performance, that the effect survives training-seed variation, and that its compute relationship is understood. Any specialization of temporal or depth states is an empirical finding, not a prescribed division of content.

## Suggested repository structure

```text
model.py                         # ordinary nanoGPT components
models/recurrent_2d.py            # hybrid model, mixers, and temporal-source block
recurrence/contract.py
recurrence/schedule.py
evaluation/chess.py
evaluation/recurrence_grid.py
data/chess_v1/prepare.py
configs/baseline_chessgpt.py
configs/recurrent_2d_pilot.py
configs/recurrent_2d_full.py
docs/RECURRENCE_CONTRACT.md
proposal.md
implementation_plan.md
tests/test_baseline_equivalence.py
tests/test_schedule.py
tests/test_recurrence_ordering.py
tests/test_temporal_causality.py
tests/test_limiting_cases.py
tests/test_state_hold.py
tests/test_gradient_flow.py
tests/test_checkpoint_resume.py
tests/test_recurrence_grid.py
```

## Coding order

1. Bootstrap ChessGPT, freeze the data and evaluator, and create both branches from a common baseline commit. Smoke-test and launch the ordinary baseline.
2. On the MVP branch, extract the recurrence contract, refactor prelude/core/source/coda, and prove $(0,0)$ equivalence.
3. Implement the checkpointable sampler, temporal-source block and shifted reads, temporal mixer, depth mixer, and complete trajectory. Verify ordering, held-state reads and gradients, causality, reductions, and resume.
4. Train and evaluate the $\{0,1,3\}^2$ pilot. Establish useful behavior before expanding to $\{0,\ldots,7\}^2$ or substantial component-baseline runs.
5. Extend compute-controlled comparisons, curriculum and supervision experiments, and live-feedback inference according to the signal. Optimize fixed-depth inference before adding an exit gate.
6. Pursue self-play and interpretability after supervised recurrence results justify them.

## Decisions to resolve during implementation

The recurrence semantics above are settled. Before coding the relevant components, specify the gate network's exact architecture, normalization types and epsilon values, and depth-projection initialization; the documents currently define their roles but not every parameterization. Stage 0 must verify and record the inherited source-row and game-boundary behavior, preserving the reference setup unless an actual issue requires a change. Stage 14 must settle prompt handoff and recurrent attention-cache semantics. These details should be recorded in the canonical contract when chosen, without silently importing defaults from an external repository.
