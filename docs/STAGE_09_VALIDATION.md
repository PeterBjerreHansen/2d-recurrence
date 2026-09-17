# Stage 9 tooling validation

Stage 9 evaluation and experiment tooling is implemented. The initial 100-update experiments remain to be run using [the MPS handoff](STAGE09_MPS_HANDOFF.md). This document records implementation checks, not evidence that recurrence improves chess modeling.

`evaluation/recurrence_grid.py` evaluates all nine pilot cells on identical seeded validation batches. It selects distinct mask placements without replacement, covering all three placements in each asymmetric cell by default. Other cells run once. Reports contain placement-level and aggregate NLL, accuracy, changes from ordinary execution, pass counts, training probabilities, and forward matrix-multiply estimates that account for held-state reads. JSON retains masks and diagnostic traces; CSV provides the cell-level table. Checkpoint hashes, step, training seed, dataset identity, evaluation seeds, fixed-batch fingerprint, and evaluation-code provenance make reports traceable.

Diagnostics use one fixed batch and one placement per cell. They record core and source RMS by invocation and gradient norms by parameter group. They run in eval mode, leave parameter gradients untouched, and reject non-finite activations or gradients. They are a bounded diagnostic probe, not a substitute for statistics across training batches or evidence of useful state content.

The trainer optionally retains step checkpoints through `keep_checkpoints=True`. The default remains false for existing configurations. `configs/stage09_mps.py` retains steps 0/25/50/75/100, enabling fixed-data curves without adding an evaluation framework to the training loop. `sample.py` accepts recurrent checkpoints only with explicit `--execution=training_graph`; each character recomputes the whole prefix with the same write masks. Reports label execution, prefill, write counts, masks, and core-pass budget. Live feedback remains Stage 14.

## Verification

`uv run pytest -q` passes 50 tests. New coverage includes exhaustive pilot placement counts, exact schedule-dependent FLOP arithmetic, direct NLL/accuracy comparison, fixed-data reproducibility, diagnostic non-mutation and unused gradients, checkpoint retention across resume, checkpoint/data mismatch rejection, and fixed-schedule generation labels. Existing recurrence, baseline, data, chess legality, distributed gradient, and exact-resume tests continue to pass.

The CLI was exercised on the existing tiny CPU checkpoint and on the full eight-block, width-512, context-1,023 MPS checkpoint from the four-update Stage 8 validation. Both produced nine-cell JSON/CSV reports and finite diagnostic gradients. Two short generation samples were also run on each device through the existing chess validator. Local ignored artifacts are in `out-stage09-validation/`.

The MPS grid check used one batch of two rows. Its `(0,0)` NLL was 3.23585 and `(3,3)` NLL was 3.26535; the asymmetric cells produced distinct placement results. This is a tooling check on a checkpoint trained for only four updates, not a finding about recurrent learning. Generation likewise checks execution and reporting, not established chess competence.

## Scope and remaining assumptions

The compute estimate counts transformer and mixer matrix multiplications, full dense attention products, and the final head. It excludes normalization, softmax, nonlinearities, elementwise operations, backward, and kernel overhead. This is explicit in the report field names and metadata. Complete compute accounting remains Stage 12.

The evaluator uses stored-row sampling with replacement, matching the existing loader. Its standard deviation is across placements on the same sampled data, not validation-sampling uncertainty or seed-to-seed uncertainty. The handoff fixes evaluation batches across checkpoints and training seeds. Using `(0,0)` from the same hybrid checkpoint measures the effect of execution budget at fixed weights; it does not replace separately trained component baselines.

No architecture or loss changes were needed. Checkpoint retention is opt-in, the grid is a standalone evaluator, and generation reuses the existing legality policy. Additional caches, experiment schedulers, dashboards, convergence claims, and optimization machinery are unnecessary for the initial measurement.
