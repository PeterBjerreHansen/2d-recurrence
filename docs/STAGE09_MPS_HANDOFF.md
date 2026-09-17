# Initial Stage 9 experiments on this Mac

Run the initial two-axis recurrence experiments in `/Users/peterbjerrehansen/Desktop/projects/coding_projects/active/2d_recurrence`, on `mvp-2d-recurrence`. Read `docs/RECURRENCE_CONTRACT.md`, `docs/STAGE_09_VALIDATION.md`, and Stage 9 of `implementation_plan.md`. The task is to measure early learning and whether additional recurrent computation helps. Do not redesign the architecture to improve these first results. Use MPS, float32, eager execution, and the existing prepared `data/smoke_real` dataset. Keep all checkpoints and raw reports local; preserve the frozen baseline branch and tag.

First inspect the worktree and existing output directories, run `uv run pytest -q`, and verify `torch.backends.mps.is_available()`. If the prepared subset is missing, use the README's 4,096-row preparation command, not the full archive. The data loader verifies the manifest and file hashes. Run jobs sequentially to avoid competing MPS allocations. The two training runs retain about 4 GB of checkpoints in total. Budget at most 45 minutes for this initial experiment; if that is insufficient, preserve completed work and report what remains. Do not silently reduce context or model size.

## Training and checkpoint curves

Use the committed 100-update configuration, which retains snapshots at steps 0, 25, 50, 75, and 100. Train two independent seeds, keeping the architecture, data, batch size, learning-rate schedule, and pair probabilities fixed:

```sh
uv run python train.py configs/stage09_mps.py --out_dir=out-stage09-seed1337 --seed=1337 --recurrence_seed=1729
uv run python train.py configs/stage09_mps.py --out_dir=out-stage09-seed1338 --seed=1338 --recurrence_seed=1730
```

If a run already has a compatible checkpoint, inspect its saved configuration and use the same command with `--init_from=resume`. Do not overwrite or relabel an unrelated run. Keep `lr_decay_iters=100` and the ten-step warmup unchanged. If training or evaluation produces non-finite values, stop that run, preserve the evidence, and diagnose the fault before spending more compute. Poor but finite losses are experimental results, not permission to tune the architecture after seeing validation data.

Evaluate each retained checkpoint using the same data seed, mask seeds, batch count, and batch size. The following shell loop is for fresh report paths; on a resumed task, validate existing report metadata and run only missing evaluations:

```sh
for run in out-stage09-seed1337 out-stage09-seed1338; do
  for step in 000000 000025 000050 000075 000100; do
    uv run python -m evaluation.recurrence_grid \
      --checkpoint "$run/ckpt-step$step.pt" --device=mps \
      --batches=8 --batch-size=2 --data-seed=2027 --mask-seeds 11 23 37 \
      --output "$run/grid-step$step.json"
  done
done
```

Keep diagnostics enabled. Verify that all grid reports have the same dataset manifest hash and batch fingerprint. Each grid has nine cells and 13 distinct placement evaluations. The two asymmetric cells have three placements each; all other cells have one. Retain each placement's result. Compare within-checkpoint NLL against `(0,0)`, temporal-only `(3,0)`, and depth-only `(0,3)`, and compare those differences over training. Report each training seed separately before summarizing across seeds. Mask-placement standard deviations are not training-seed confidence intervals.

## Chess continuations

At the final checkpoint, evaluate `(0,0)`, `(3,0)`, `(0,3)`, and `(3,3)` with identical prompts and sampling seeds. Save the JSON prompt list `[';1.', ';1.e4 e5 2.', ';1.d4 d5 2.', ';1.e4 c5 2.']` with proper JSON double quotes to `out-stage09-prompts.json`. For each run and pair, use the following command with the appropriate counts and a distinct output filename:

```sh
uv run python sample.py --checkpoint out-stage09-seed1337/ckpt-step000100.pt \
  --device=mps --execution=training_graph --u-t=3 --u-d=3 --mask-seed=11 \
  --prompts out-stage09-prompts.json --num-samples=20 --seed=2027 \
  --temperature=1 --top-k=0 --max-new-tokens=128 \
  --output out-stage09-seed1337/generation-3-3.json
```

These are training-graph prefix recomputations, not live-feedback generation. Preserve that distinction in every result. Use the existing stop-on-first-illegal-move policy without legal-move filtering or retries. Report attempted moves, legality, legal continuation length, and stopping reasons; zero attempts means undefined legality, not perfect or zero legality. If the existing ordinary baseline checkpoint is available, evaluate it with the same prompts/settings as a contextual reference and label it separately. The hybrid's `(0,0)` setting is not a separately trained baseline, and equal optimizer steps are not equal compute.

## Deliverable

Write `docs/STAGE_09_INITIAL_RESULTS.md` with configurations, code provenance, dataset identity, completed steps, per-seed nine-cell tables, placement variation, generation results, and links to local raw artifacts. Create standard static learning-curve plots under the ignored experiment output directory, using a plotting tool available in the environment; retain tabular CSVs as the reproducible source. Plot NLL versus optimizer updates and differences relative to `(0,0)` separately so that general language-model learning does not masquerade as a recurrence benefit.

Inspect state RMS across passes and the one-batch diagnostic gradient norms. These gradients are eval-mode probes, not accumulated training gradients. A null gradient is expected for an unused mixer. Neither a nonzero gradient nor changing predictions demonstrates useful memory content. FLOP fields estimate forward matrix multiplications and exclude several costs; do not claim a complete compute frontier. The small stored-row validation split and two short runs support a preliminary decision only.

Conclude whether to continue the current pilot, collect a longer curve, or investigate a specific observed failure. A negative or inconclusive result is acceptable. Do not launch grid expansion, extensive component training, hyperparameter searches, or live-inference implementation as part of this task. Report all incomplete checks honestly.
