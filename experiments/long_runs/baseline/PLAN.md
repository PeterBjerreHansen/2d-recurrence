# Baseline configuration for a longer training run

Historical record of the completed experiment. Paths below follow the current repository layout; recorded configurations, measurements, and raw artifact provenance retain their original meaning. See the experiment README for current commands.

Status: proposed plan, not an executed experiment. The objective is one useful, reproducible recurrent chess model with a small configuration-selection overhead. This is not a factorial ablation study or a claim to have found an optimal configuration.

## Decisions informed by the review

The current recurrent model has 27,814,912 parameters, not approximately 50M. The completed pilot used a ten-update warmup; 2,000 is the upstream trainer default, not the setting actually tested. Our two-axis recurrence has at most four core passes, unlike the much deeper and larger models motivating several of the review's warnings. Existing two-seed results already show stable learning and late improvements from both temporal and depth recurrence. Those results should inform defaults alongside the literature.

Huginn reports representation collapse and recurrence being ignored, followed by a successful configuration with both normalization and learning-rate changes. This supports monitoring recurrent behavior and testing learning rate; it does not supply a transferable optimum for our model. Its optimizer also differs from stock PyTorch AdamW. Parcae's learning-rate sensitivity strengthens the warning without establishing universal thresholds. Retrofitted Recurrence supports Muon and recurrence curricula in its setting, but starts from pretrained models and studies much deeper looping. Keep AdamW and sweep only one alternative learning rate initially. Sources: [Huginn](https://arxiv.org/html/2502.05171#S4.SS3), [Parcae](https://arxiv.org/html/2604.12946), [Retrofitted Recurrence](https://arxiv.org/html/2511.07384).

Keep temporal memory equal to the raw dedicated source-block output, `m_T = S(h)`, and depth memory equal to core output. This is already resolved in the implementation and user-approved contract. Keep normalized gate inputs and normalized projected value streams. Identity value matrices do not make the full mixer identity-like. In a read-only audit on two fixed training rows at `(1,1)`, seed 1337's step-0 checkpoint had prelude RMS 0.09077 and temporal-anchor RMS 0.91786; at step 1,000 these were 0.79622 and 0.81216. Depth state/input RMS were 0.16089/0.89464 initially and 0.73942/1.02308 at step 1,000. These measurements expose an initialization scale transition, not evidence that it must be removed. Retain the tested implementation unless new diagnostics reveal a concrete failure. Do not add runtime norm matching or replace the backbone's normalization speculatively.

Keep gradient clipping at 1.0. Current training accumulates gradients from four independently scheduled microbatches, then clips the combined gradient. Log that actual norm and clip coefficient with the schedule mixture. It cannot be honestly labeled a per-cell clipping event. A larger standalone gradient for a high-pass cell also does not establish simple optimizer downweighting: accumulation, gradient directions, clipping, and Adam's moments all matter.

Keep the fixed update distribution, final-only loss, zero dropout, and full backpropagation. The current distribution already assigns 40% probability to four-pass trajectories. FBT's rare deeper-trajectory result is a reason to preserve composition training, not a reason to add another rare-deep mechanism immediately. Stress tests beyond four passes are diagnostic and do not certify live-feedback stability. Source: [FBT](https://arxiv.org/html/2608.08888#S3).

Deep supervision is a credible later option. FBT supervises the first pass plus an average of subsequent losses; LRT also reports intermediate supervision and matched-compute gains. Neither isolates the benefit of adding those losses to our architecture. A final-only randomly stopped model already directly supervises several horizons. If one objective experiment is eventually affordable, compare against `(L_final + 0.25 * mean(L_intermediate)) / 1.25` for multipass trajectories, and ordinary `L_final` for one pass. This assigns 80% weight to the final loss and 20% total to earlier losses. Averaging auxiliaries without the outer denominator leaves one-pass and multipass examples with different total coefficient mass. This normalization controls coefficients, not gradient magnitude. Use the same source/coda/readout, reuse an existing source activation when available, and keep auxiliary prediction computations separate from masked state writes. This costs extra source/coda computation. Sources: [FBT objective](https://arxiv.org/html/2608.08888), [LRT](https://arxiv.org/html/2605.26797).

Do not reduce independent training examples merely because recurrence reuses their representations. The existing distribution already averages 2.7 core passes and 16.1 block applications versus eight ordinary blocks, before mixer overhead. Our immediate objective is configuration selection at a fixed recurrence distribution. A full equal-token/equal-FLOP study, retuned weight decay, and component comparisons are later work. Full backpropagation is appropriate at four passes; activation checkpointing comes before truncation if memory becomes limiting.

## Fixed configuration

| Setting | Choice |
| --- | --- |
| Backbone | Current width-512, 8-head, 2 prelude / 4 shared core / 1 source / 1 coda |
| Context and vocabulary | 1,023 input characters; existing 32-character vocabulary |
| Mixers | Current LayerNorm and projections; temporal value identities; depth projections 0.5I; initial temporal gates 0.1/0.9 |
| Optimizer | PyTorch AdamW; betas 0.9/0.95; existing epsilon; weight decay 0.1 |
| Learning rate | Default 3e-4; sole initial challenger 1e-4 |
| Schedule | 100-update linear warmup; existing cosine schedule through update 10,000; minimum LR = peak/10 |
| Batch | Microbatch 2, accumulation 4, effective batch 8 rows / 8,184 target characters |
| Recurrence | Existing fixed 3-by-3 probability matrix over update counts 0/1/3; random exact-count write placement |
| Loss / gradients | Final-pass mean cross entropy; complete gradients through all passes |
| Dropout | 0 |
| Runtime | MPS, float32, eager execution |

The 100-update warmup is a modest practical default, not a number established by the cited papers. Keep it identical between learning-rate candidates. Keep batching fixed rather than introducing another gradient-noise variable. Keep the existing optimizer parameter grouping, including no weight decay on one-dimensional normalization and bias parameters.

## Data and selection protocol

Prepare the entire pinned `lichess_100mb_blocks.zip` archive, without the old 4,096-row cap, in a new immutable directory. Keep the vocabulary, row alignment, split seed, and 1% validation rule. Record the actual row count and hashes rather than assuming a count from the filename. This provides more varied training exposure at the same per-update compute cost. It does not turn the inherited stored-row split into a game-disjoint split.

Select a deterministic 128-row validation panel from the new validation split for configuration selection; keep its complement unexamined until the selected run's milestones. If fewer rows are available, record a smaller panel and preserve a disjoint complement where possible. The panel is for comparing settings and detecting failures, not demonstrating interesting generalization. Use identical row indices, data order, initialization seed, and schedule RNG seed for both learning-rate candidates. Start fresh on this data version; do not treat changing the dataset of an existing checkpoint as exact resume.

Before training, correct the feedback-diagnostic donor permutation and add a focused test. For two targets followed by their donors, `[2,3,0,1]` is the intended permutation; the current code constructs `[2,0,3,1]`. The existing diagnostic still substituted other-row content, but its donor labels do not match the actual substitution. Test new logging and freeze code/configuration provenance before launching.

## Minimal instrumentation

At each optimizer update, compute and record the aggregate pre-clipping gradient norm, clip coefficient, whether clipping occurred, learning rate, characters processed, elapsed training time, and the four microbatch schedules. Summarize clipping frequency over windows. Reject non-finite gradients rather than hoping clipping repairs them.

At evaluation milestones, use one fixed diagnostic batch to record prelude/core/source and mixer-input/output RMS, relative changes between passes, sampled cross-token cosine similarity, and temporal gate/value contribution summaries. Keep diagnostic gradients separate from accumulated training gradients. Do not build full position-by-position correlation matrices or log every activation. Interpret norms, similarities, and gates alongside NLL; a small state change can be useful convergence, and a small gate can still transmit information through a large value projection.

Evaluate the full nine-cell grid at 0, 250, 500, 750, and 1,000 during selection. Track absolute `(3,3)` NLL, distribution-weighted grid NLL, and differences against `(1,0)` and `(3,0)`. Optimizing the recurrence gap alone can select a model that merely damages shallow execution.

At update 1,000, do a cheap no-gradient eight-pass stress check on one fixed batch: temporal-only `(7,0)`, joint `(7,7)`, and fixed-memory depth refinement `(1,7)` with its sole temporal write in the first slot. A 16-pass check is optional if eight passes remain finite. These tests are outside training support; poorer loss there is not a rejection criterion. Instability within trained support is materially different from extrapolation failure. Fixed memory is important when interpreting depth dynamics as refinement of a fixed input.

## Run allocation and selection

Run candidates A and B for 1,000 updates each, using peak LRs 3e-4 and 1e-4 respectively. Set `lr_decay_iters=10000` from the start, with candidate-specific minima 3e-5 and 1e-5; set only the temporary stopping point `max_iters=1000`. Preserve optimizer and RNG state. This avoids comparing short runs that have already completed their learning-rate decay, and allows the winner to continue without restarting.

Select primarily by stable absolute `(3,3)` validation NLL across the later selection checkpoints, using the weighted grid metric and recurrent dynamics as checks. Do not discard a slower-learning candidate solely because its recurrence benefit has not emerged at the same early step; compare at similar attained loss when interpreting refinement. If results are effectively tied or noisy, retain the demonstrated 3e-4 setting. The 3e-5 candidate is a fallback only if both planned rates show recurrent instability, not a mandatory third run. Avoid selecting a lower LR merely because its gradients are smaller.

Continue the winner from update 1,000 to 10,000, retaining all its schedule, optimizer, and dataset settings. Change only the stopping limit. This costs 11,000 optimizer updates across the two candidates, of which 1,000 belong to the unselected candidate. If even that overhead is too high, skip the challenger and use 3e-4. Do not add a second training seed or objective sweep before this longer model has been trained.

Retain selected-run checkpoints at 1k, 2k, 5k, 8k, and 10k, with recoverable latest checkpoints between them. Evaluate the selection panel on the nine-cell grid at these milestones and the validation complement at 5k/10k. Use small fixed continuation samples at 2k/5k/10k to observe growing chess competence; generation remains training-graph execution until live feedback is implemented. Do not require stable live generation or a specific legality threshold to continue a numerically healthy early run.

Ten thousand updates process 81.84 million target characters, regardless of recurrent pass count. This is a useful next budget, not a promise of mature chess understanding. At 10k, use the learning curve, train/validation relationship, generation behavior, and recurrence surface to choose whether a further training tranche is worthwhile. Any later LR restart or dataset change should be recorded as an explicit new training phase.

## Optional work after the baseline

If the longer run learns well and compute remains, the first objective comparison is one final-only versus normalized-auxiliary-loss comparison at lambda 0.25, not a lambda sweep combined with learning-rate changes. Evaluate practical quality per wall-clock compute as well as at equal updates because the prediction tails add work. Curriculum, Muon, sandwich normalization, larger update support, and truncated gradients remain conditional responses to a measured problem or separate later research questions.
