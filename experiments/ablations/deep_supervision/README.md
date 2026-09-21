# Compute-matched supervision selection

Compare variation A with final-only supervision against normalized deep supervision: for multi-pass trajectories, `L = (L_final + 0.25 * mean(L_intermediate)) / 1.25`; one-pass trajectories use `L_final`. Keep the architecture, update distribution and all other training settings identical. This directory contains the controlled comparison and its raw artifacts.

The serious profile is in `experiments/serious.py`: full pinned Lichess 6GB blocks, effective batch 100 (microbatch 5 × accumulation 20), context 1,023, eight blocks/512 width/eight heads, A layout 1/1/4/1/1, AdamW 3e-4 to 3e-5, betas .9/.95, weight decay .1, clipping 1, dropout 0, seed 1337, schedule seed 1729. CUDA BF16 with float32 parameters, TF32 enabled, eager execution for both models. Compilation is disabled because the recurrent trainer does not support its variable schedules. These settings are frozen before benchmarking, not independently tuned per arm. BF16/memory feasibility still needs validation on the actual A6000; if it fails, revise the shared profile before freezing a fresh experiment.

The first ten benchmark updates are excluded from throughput averages. Budget each ablation arm for the measured final-only cost of at least 250M characters, rounded up to complete batch-100 updates. This is about three times the previous pilot's data exposure at the old throughput, and is a supervision-selection experiment rather than a claim about mature recurrence. Both arms stop on accumulated, synchronized training-update seconds, including data loading and optimizer work, excluding evaluation and checkpoint I/O. They may exceed the budget by one update. LR warmup occupies 2% of this time budget and cosine decay follows consumed training time. A one-million-update ceiling is a fail-safe, not the intended duration.

Checkpoints save the consumed training time, so spot recovery continues the same schedule. Lost work after the last durable checkpoint is replayed and does not count as retained-model training compute; billable wall time can therefore be larger. Source/runtime, dataset and panel hashes are frozen. Do not compare timing from another GPU/backend or silently change microbatching after freeze.

Evaluate both endpoints in CUDA float32 on the same exact 256-row selection panel and mask seeds 11/23/37. Inspect (3,3), improvement from (1,1) to (3,3), the full nine-cell surface, training curves, clipping and nonfinite diagnostics. Compare final-pass NLL, never the differing training objectives. Choose deep supervision if it offers a useful practical improvement at equal time without a material regression in deeper refinement; otherwise keep final-only. Record the judgment and limitations explicitly, including a near-tie if applicable. One seed is a default-selection exercise, not statistical proof. Evaluate the chosen endpoint once on a separate fixed 512-row confirmation sample. Confirmation is not a second tuning set.

`choose` records the reviewed default and unlocks the 1B/64B paired runners. It does not launch them. The optional `deep_more` benchmark changes max-update probabilities from 10/50/40% to 10/30/60% for 0/1/3 updates (one/two/four physical passes); it is a cost probe only and is not included in this supervision ablation or selection decision.

See [HANDOFF.md](HANDOFF.md) for commands. Every generated artifact belongs under this experiment's ignored `results/`, except long-run outputs, which live beside their own configs. No VM is provisioned by these scripts.

The raw receipts were produced before this directory was renamed from
`experiments/ablations/supervision_compute`. Their historical JSON metadata
retains those original path strings so the recorded artifacts and hashes are
not rewritten. The relocation is recorded in
[`experiments/relocations.json`](../../relocations.json); the current runner
and documentation use `deep_supervision` consistently.
