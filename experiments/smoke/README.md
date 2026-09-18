# Pipeline checks

`configs/` contains bounded ordinary, recurrent, and full-width baseline checks. `results/` preserves the completed CPU, MPS, DDP, baseline, and recurrent checks; no results were deleted in cleanup. See [stages 0–1 validation](STAGE_01_VALIDATION.md) and [stages 2–8 validation](STAGE_02_08_VALIDATION.md).

These configs preserve their historical model shapes for reproducibility. For the current default A on the larger dataset, use [local recurrent MPS config](../../configs/local/recurrent_mps.py). Choose a fresh experiment-local output directory when rerunning a scratch check; an existing checkpoint is never silently overwritten.
