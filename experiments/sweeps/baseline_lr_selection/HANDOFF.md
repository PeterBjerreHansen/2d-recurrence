# Execute baseline selection and continuation

Historical record of the completed experiment. Paths below follow the current repository layout; recorded configurations, measurements, and raw artifact provenance retain their original meaning. See the experiment README for current commands.

Work in `/Users/peterbjerrehansen/Desktop/projects/coding_projects/active/2d_recurrence` on `mvp-2d-recurrence`. Read `experiments/sweeps/baseline_lr_selection/PLAN.md` for rationale and `docs/RECURRENCE_CONTRACT.md` for model semantics. This handoff fixes the executable protocol and selection rule. The user authorizes implementation of the small prerequisite fixes below, two matched LR candidates, and continuation of one selected candidate. Do not expand the architecture, objective, optimizer, recurrence distribution, or number of training runs.

The configurations are defined in `experiments/sweeps/baseline_lr_selection/configs/lr3e4.py` and `experiments/sweeps/baseline_lr_selection/configs/lr1e4.py`. They deliberately fail before launch until `eval_panel_path` is supported; they are experiment definitions, not a claim that the prerequisite tooling is already implemented. Do not remove that check to work around missing functionality.

## Runs and budget

| Run | Config | Output directory | Peak / minimum LR | Updates |
| --- | --- | --- | --- | --- |
| A | `experiments/sweeps/baseline_lr_selection/configs/lr3e4.py` | `experiments/sweeps/baseline_lr_selection/results/lr3e-4` | 3e-4 / 3e-5 | Fresh 0 to 1,000 |
| B | `experiments/sweeps/baseline_lr_selection/configs/lr1e4.py` | `experiments/sweeps/baseline_lr_selection/results/lr1e-4` | 1e-4 / 1e-5 | Fresh 0 to 1,000 |
| C | Selected A or B config | Same selected directory | Unchanged | Resume 1,000 to 10,000 |

Total requested training is 11,000 optimizer updates, including the losing candidate's 1,000. These are matched optimization candidates, not independent training seeds: both use model/data seed 1337 and schedule seed 1729. The selected model sees 81.84 million target characters; the total across candidates is 90.024 million. Recurrent passes do not count as additional characters.

All runs use the current 2/4/1/1 width-512 model, context 1,023, final-only loss, full backpropagation, zero dropout, AdamW betas 0.9/0.95, weight decay 0.1, clip 1, batch size 2 and accumulation 4, fixed update distribution, MPS float32, and eager execution. The LR warmup is 100 updates and the cosine horizon is 10,000 from the first update. Only peak LR, minimum LR, and output directory differ between A and B. Run sequentially on this Mac. The former Stage 9 45-minute cap does not apply to this longer task. Measure throughput and report an ETA; use the prescribed update budget rather than silently shortening training.

## Preflight implementation and verification

Inspect and preserve existing modified and untracked work; do not reset the checkout or overwrite prior results. Verify MPS and run the existing tests. Make only these necessary implementation changes before training:

1. Correct the donor permutation in `evaluation/feedback_diagnostic.py`. For targets followed by their donors, use `[2,3,0,1]` for two targets and `[1,0]` for one. Add a test with distinguishable source outputs that verifies actual donor identity for every target, including the final one-row batch. Rerun the two old step-100 diagnostic checkpoints into new files under `experiments/sweeps/baseline_lr_selection/results/preflight/`, without overwriting the original reports.
2. Add a string trainer configuration field `eval_panel_path`, defaulting to the empty string for backward compatibility. When provided, validate the panel's dataset-manifest hash and row indices, and restrict routine trainer validation sampling to its selection indices. Training sampling must remain unchanged. Record the panel file hash in run metadata and checkpoints, and reject a changed panel on exact resume. The nine-cell evaluator also needs `--panel-file` and `--panel-split=selection|confirmation`, using every specified row exactly once and respecting the final partial batch. Reuse `evaluate_grid(..., fixed_batches=...)`; do not create a second recurrence implementation. Reports must retain explicit row indices, panel-file hash, dataset identity, batch fingerprint, and target counts.
3. Log aggregate pre-clipping gradient norm, applied clipping coefficient, and clipping status every optimizer update, together with LR, elapsed training time, characters processed, and the actual four-microbatch schedule list. Use the norm returned by the clipping operation; do not change gradient scaling or clip each microbatch separately. For clip threshold 1, the coefficient follows the actual operation, including its epsilon. Detect non-finite gradients before an optimizer step. Label these statistics as accumulated-update statistics, not per-cell clipping probabilities.
4. Extend the bounded fixed-batch diagnostics as necessary to record prelude/core/source and mixer-input/output RMS, relative changes between successive passes, sampled cross-token cosine similarity, and gate/value contribution summaries. Compare identical positions between passes. Do not allocate quadratic token-correlation matrices. Keep hooks observational and remove them after use; preserve model state, gradients, and training RNG. Existing gradient probes are eval-mode diagnostics, not accumulated training gradients.

Test panel isolation, exact target counts for a partial final batch, non-finite gradient rejection, clipping-log accuracy, and resume equivalence with the new logging/panel support. Retain existing baseline and distributed tests. Smoke-test changes on tiny CPU fixtures; use one disposable full-config MPS forward/backward check if needed, not another exploratory training run. Freeze the training code and configuration before A or B starts, recording a commit or a complete patch plus hash. A dirty flag alone is insufficient to reproduce new changes.

## Dataset and fixed panels

Prepare the full pinned archive into a new data directory. Do not pass `--max-rows` and do not reuse the old smoke dataset:

```sh
uv run python data/chess_v1/prepare.py \
  --file lichess_100mb_blocks.zip \
  --revision 1a932e1abca935aae585f417ede39ecde4f2a620 \
  --out-dir data/chess_143K_v1 --seed=2357 --val-fraction=0.01
```

If that directory already exists, verify its source revision, archive, uncapped preparation, split settings, hashes, and completion manifest. Reuse it only if it matches. Preserve the existing preparation checks, vocabulary, row alignment, and within-row boundary semantics. Do not silently deduplicate differently or change the split to make preparation pass. Record actual rows and characters; the archive filename is not a reliable row count.

Create `experiments/sweeps/baseline_lr_selection/results/panels.json` once. Let `n` be the number of validation rows. For `n >= 129`, select 128 distinct indices using `random.Random(2027).sample(range(n), 128)` and sort them. Otherwise select `max(1, n // 2)` when `n >= 2`; if there is only one validation row, stop and report the insufficient split. The sorted remaining indices form confirmation. Store `dataset_manifest_hash`, `validation_row_count`, `selection_seed=2027`, `selection_indices`, and `confirmation_indices`. Reuse this file unchanged throughout. Routine trainer validation must use selection only; do not evaluate confirmation during candidate selection.

## Candidate execution and evaluation

After all prerequisites pass, run:

```sh
uv run python train.py experiments/sweeps/baseline_lr_selection/configs/lr3e4.py
uv run python train.py experiments/sweeps/baseline_lr_selection/configs/lr1e4.py
```

Inspect existing checkpoints before launching. For an interrupted compatible candidate, use the same configuration with `--init_from=resume`; never start over in a directory containing a checkpoint. Compatibility includes data/panel identity and every training setting. Candidate runs stop at total step 1,000, not 1,000 additional updates.

Both configurations retain steps 0, 250, 500, 750, and 1,000. The trainer also refreshes its recoverable latest checkpoint every 100 updates. Once a candidate is complete, evaluate the retained steps with the selection panel. The following CLI flags must have been implemented and tested in preflight:

```sh
for run in experiments/sweeps/baseline_lr_selection/results/lr3e-4 experiments/sweeps/baseline_lr_selection/results/lr1e-4; do
  for step in 000000 000250 000500 000750 001000; do
    uv run python -m evaluation.recurrence_grid \
      --checkpoint "$run/ckpt-step$step.pt" --device=mps \
      --panel-file experiments/sweeps/baseline_lr_selection/results/panels.json --panel-split=selection \
      --batch-size=2 --mask-seeds 11 23 37 \
      --output "$run/grid-selection-step$step.json"
  done
 done
```

Do not replace panel evaluation with eight randomly sampled batches. Each grid uses all panel rows, all nine cells, and all 13 distinct pilot placements. Compare batch fingerprints and panel hashes across reports. Keep diagnostics enabled on the first batch. On resume, validate and reuse completed matching reports; generate only missing ones. Preserve raw JSON and CSVs.

At step 1,000 for each candidate, run no-gradient eight-pass checks on the same first selection batch: `(7,0)`, `(7,7)`, and `(1,7)` with the temporal mask `(True,False,False,False,False,False,False)` and all depth writes enabled. Use the existing model and explicit schedule API. Record NLL, finite status, and state RMS by pass. Keep failures beyond training support separate from numerical faults within trained support. Do not add these schedules to training or require convergence. Skip the optional 16-pass experiment for this initial run allocation.

## Predeclared selection rule

An eligible candidate completes training and the nine-cell selection evaluations without non-finite values in trained support. A non-finite failure must be investigated and reported, not hidden by restarting at a different LR or silently changing precision. If one candidate is ineligible and the other is eligible, choose the eligible one. If neither is eligible, stop before long continuation and report the failure; a third LR run is not authorized by this protocol.

For each eligible candidate, compute `S`, the mean `(3,3)` selection NLL across steps 750 and 1,000, and `W`, the mean across those same checkpoints of the nine-cell NLL weighted by the fixed training probabilities. Use placement-mean NLL for asymmetric cells.

Choose B over A only if all three conditions hold: `S_B <= S_A - 0.005`, B's `(3,3)` NLL is no worse than A's at both steps 750 and 1,000, and `W_B <= W_A + 0.005`. Otherwise choose A. The 0.005 margin is a predeclared practical tolerance to avoid choosing on tiny fluctuations, not a statistical confidence bound. Smaller gradients or a larger recurrence advantage alone do not determine the winner. Report state dynamics, clipping behavior, and the individual cell curves alongside the decision; do not infer a dead recurrence pathway just because one candidate learns more slowly.

Write `experiments/sweeps/baseline_lr_selection/results/selection.json` before continuing. Include candidate config/checkpoint hashes, panel hash, both S/W values, per-checkpoint `(3,3)` values, eligibility status, selected config and directory, and the rule outcome. Do not use confirmation-set results or generation samples to override the decision.

## Continue only the winner

For A, run:

```sh
uv run python train.py experiments/sweeps/baseline_lr_selection/configs/lr3e4.py --init_from=resume --max_iters=10000
```

For B, substitute `experiments/sweeps/baseline_lr_selection/configs/lr1e4.py`. Keep the selected output directory and every training setting unchanged. Do not reset optimizer moments, sampler state, warmup, or LR decay. Do not train both candidates to 10,000. The selected config already retains steps 2k, 5k, 8k, and 10k; latest checkpoints continue every 100 updates. Use normal resumable execution through interruptions, not a new scheduler service.

Evaluate the selection panel on the full nine-cell grid at 2k, 5k, 8k, and 10k with the same flags and file naming. At 5k and 10k, also evaluate all confirmation rows exactly once using `--panel-split=confirmation` and distinct `grid-confirmation-stepNNNNNN.json` files. Confirmation is a later monitoring panel within the inherited row split, not a claim of game-disjoint generalization and not a reason to tune this run further. Do not retune the model after seeing it.

At 2k, 5k, and 10k, generate 20 samples for each of `(0,0)`, `(1,0)`, `(3,0)`, and `(3,3)`. Use the fixed prompts `;1.`, `;1.e4 e5 2.`, `;1.d4 d5 2.`, and `;1.e4 c5 2.`, seed 2027, temperature 1, top-k 0, maximum 128 new characters, and mask seed 11. Save a proper JSON prompt list under `experiments/sweeps/baseline_lr_selection/results/`. Use `sample.py --execution=training_graph --device=mps` with the appropriate retained checkpoint and counts. Preserve stop-on-first-illegal-move behavior and execution labels. These are developmental observations; no generation-quality threshold gates continued training.

## Deliverables and boundaries

Write `experiments/sweeps/baseline_lr_selection/REPORT.md` with the selected configuration, data size and identity, exact selection decision, code/environment provenance, training time and characters, nine-cell learning curves, clipping summaries, state diagnostics, and continuation statistics. Keep raw artifacts and checkpoints under `experiments/sweeps/baseline_lr_selection/results/`, and link them from the report. Produce static NLL-versus-updates and NLL-difference-versus-`(1,0)` plots, showing LR candidates separately during selection and the selected trajectory afterward. Retain source CSVs. Report useful refinement as an observed effect, not as proof of reasoning; forward matmul estimates are not complete training FLOPs.

Respect the finite 11,000-update allocation. No additional seeds, third LR candidate, deep supervision, curriculum, new optimizer, normalization changes, expanded training support, live-feedback implementation, or cloud provisioning. Keep the baseline branch/tag and previous experiments intact. Poor finite early metrics are not a reason to stop a healthy run or to change the experiment after seeing results. If an implementation or resource problem prevents completion, retain resumable state and report exactly what was completed and what remains.
