# Stage 0 and 1 validation

The implementation retains the Karvonen/nanoGPT transformer computation and adds verified row-aligned data preparation, explicit generation stopping, and complete checkpoint state. The full eight-layer model has 25,714,688 parameters including learned positions. With copied weights, its logits and loss matched the pinned upstream model exactly on a CPU input.

The automated suite covers row alignment under shorter contexts, shifted targets, data integrity, refusal to overwrite a dataset version, rejection of train/validation overlap and invalid characters, legal/malformed/illegal move handling, checkmate, castling, promotion, prompt accounting, causal model behavior, tied weights, and exact checkpoint resume with dropout. The initial run passed 14 tests.

A 40-update CPU smoke run on real chess text reduced validation NLL from 3.5064 to 2.3392 and reached 43.0% character accuracy. This uses a two-layer width-64 model and a 128-character context; it verifies learning and infrastructure, not the proposed architecture. Five sampled continuations did not complete a legal move, so this smoke test is not evidence of chess strength.

A two-process CPU/Gloo run completed training, evaluation, distributed checkpoint saving, and resume from two to three completed updates. On this machine, the default standalone rendezvous could not resolve the hostname; using an explicit localhost rendezvous succeeded. A two-update MPS run of the full eight-layer model and 1,023-character context completed with finite outputs and saved a checkpoint. CUDA, NCCL, compilation, and a full-corpus training run have not been validated here.

## Dataset used for local checks

The subset uses the first 4,096 rows of `lichess_100mb_blocks.zip` from dataset revision `1a932e1abca935aae585f417ede39ecde4f2a620`, followed by the upstream shuffled 99/1 split with seed 2357. It contains 4,055 training rows and 41 validation rows with no exact row overlap. Internal game markers occur in 3,574 training rows and 36 validation rows; their within-row causal context is preserved. Preparation is row-based, so game-level disjointness is not asserted.

The local data, checkpoints, and detailed logs are ignored by Git. Their manifests and run records contain source and output hashes, settings, and provenance. The full 6 GB reference archive was not downloaded. The local subset is intended for validation and a short initial baseline run.

## Initial eight-layer run

The 100-update run completed on MPS with the full eight-layer model, width 512, eight heads, 1,023-character context, and effective batch size eight. The code was clean at commit `02433146649ba54b83a98d1ff6bb5a65ae79dab9` on `baseline-chessgpt`, tagged `baseline-stage01`. Steady training updates took approximately 0.47 seconds each.

| Metric | Initial | After 100 updates |
| --- | ---: | ---: |
| Validation NLL | 3.5744 | 1.7267 |
| Validation character accuracy | 1.5% | 40.0% |

Twenty temperature-1 continuations from `;1.` with seeds 1337–1356 produced 38 legal moves across 58 completed move attempts (65.5%). Mean legal continuation was 1.9 moves. Nine samples stopped on malformed moves and eleven on illegal moves; none completed a valid game. These are early learning results, not a trained chess-playing benchmark. The same untrained architecture and initialization produced no legal moves under the same evaluation settings.

Loading the final checkpoint in evaluation-only mode reproduced the final NLL and accuracy. [Machine-readable results](baseline_pilot_results.json) record evaluations, sampling settings, data manifest, and checkpoint checksum. The default full-corpus training run remains unexecuted. The working branch is `mvp-2d-recurrence`; its baseline implementation starts from the same frozen commit, with validation documentation recorded afterward.
