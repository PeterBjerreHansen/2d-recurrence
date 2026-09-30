# Agent guide: `engine-policy` branch

This branch retrains the recurrence architectures (transformer, temporal-only, depth-only, hybrid) on chess with **move tokens** and **move targets**: the played move, the legal set, and Leela's search distribution. Most of the repository still describes the **character-level** setup on `main`. Read this file first.

## Start here

- [`docs/engine_policy_plan.md`](docs/engine_policy_plan.md) is the single source of truth for this branch: the question, design, primary analyses, **Pitfalls** and the **Decisions** log.
- [`docs/chess_related_work.md`](docs/chess_related_work.md) covers related work.

## Which docs apply

| Status on this branch | Docs |
| --- | --- |
| **Current** | the two above |
| **Applies:** architecture, recurrence, execution | `CONTEXT.md` (terminology), `docs/concepts.md`, `docs/RECURRENCE_CONTRACT.md` (move models use its optional untied readout and target objects, under "Readout and objectives"), `docs/INFERENCE_CONTRACT.md` (except its next-character NLL helpers) |
| **Character-level setup only:** context, not instructions | the README's data and training sections, `docs/usage.md`, `docs/20B_experiment_plan.md`, and the READMEs and reports under `experiments/` |

## Rules

1. **Say "ply", not "move",** for one side's move, in code and results. Token budgets count human plies, random plies, supervised positions and row tokens separately.
2. **The training graph is not live execution** for temporal and hybrid models. Their gap is a result, not a bug. Never change recurrence semantics to make them agree; the correctness checks are listed in the plan's Pitfalls.
3. **Each output head answers only its own question.** Never read legality from a value head, or move quality from a legality head.
4. **Engine values are from the perspective of the player making the move.** Scores of the position after a move must be negated.
5. **Make changes additive.** Put new code in new modules and config options, and keep the character-level path working. If a change alters shared behaviour, update the contract doc in the same commit.
6. **Log design changes** in the plan's Decisions section, with the reason. Fold reviews into the plan instead of committing them beside it.
7. **Git:**
   - stage explicit paths, never `git add -A`;
   - never commit data, symlinks, or anything under a `results` path. Tracked `results` symlinks once destroyed the 20B checkpoints;
   - data, `.venv` and checkpoints live only in the main checkout. Other worktrees need symlinks to them, which must stay uncommitted.

## Code

- **`moves/`:**
  - the vocabulary (`vocab.py`);
  - parsing games and random games (`games.py`);
  - legal and policy targets (`objectives.py`);
  - the dataset format, packer and loader (`rows.py`);
  - the stage-1 builder (`build_stage1.py`);
  - Leela archives: reading (`leela.py`, `leela_policy.py`) and the stage-2 builder (`build_leela.py`);
  - Stockfish labelling for the deferred evaluation panel (`teacher.py`, `values.py`).
- **Trainer:** `train.py` trains on a move dataset with `data_format='moves'` and `objective='played' | 'legal' | 'engine'`. `init_from='continue'` starts stage 2 from an earlier run's trunk.
- **Pilot runner:** `experiments/move_pilot/pilot.py` resolves and runs every pilot run by name.
- **Tests:** `tests/test_moves.py`, `tests/test_move_models.py`, `tests/test_rows.py`, `tests/test_leela.py` and `tests/test_move_pilot.py`. The Stockfish tests are skipped unless `stockfish` is on the PATH or `STOCKFISH_PATH` is set. The real-archive tests are skipped unless `LEELA_ARCHIVE` points to a downloaded Leela archive.

## State of data and checkpoints

- **`data/chess_8M_v1`:** character-level Lichess rows (Karvonen's release). This is the source of the move-token conversion.
- **5B checkpoints** (temporal, depth, hybrid) in `experiments/long_runs/5B_axis/runpod_live_backups/`. These are character models and can't be reused for move tokens.
- **20B checkpoints are lost.** Their results stand in `experiments/long_runs/20B_recurrence/REPORT.md`.
- **Interpretability work** on the character models is on the `interp` branch.

## Commands

- **Tests:** `uv run pytest -q`
