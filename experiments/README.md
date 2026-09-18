# Experiments

Each experiment keeps its definition, execution/analysis scripts, concise report, and ignored `results/` together. Run Python modules from the repository root. Shared evaluators stay in `evaluation/`; shared immutable dataset versions stay in `data/`.

| Experiment | Purpose and status | Entry points |
| --- | --- | --- |
| [Architecture sites](ablations/architecture_sites/README.md) | Completed A/B ablation; A adopted as the practical default | `configs/`, `run.py`, [report](ablations/architecture_sites/REPORT.md) |
| [Recurrence grid](sweeps/recurrence_grid/README.md) | Completed 100-update, two-seed evaluation sweep | `config.py`, `curves.py`, [report](sweeps/recurrence_grid/REPORT.md) |
| [Recurrence pilot](long_runs/recurrence_pilot/README.md) | Completed 1k-update continuation on the small dataset | `config.py`, `curves.py`, [report](long_runs/recurrence_pilot/REPORT.md) |
| [Long baseline](long_runs/baseline/README.md) | Completed initial LR selection and selected 10k continuation | `configs/`, `select.py`, `curves.py`, [report](long_runs/baseline/REPORT.md) |
| [Separated pilot](long_runs/separated/README.md) | Ready-to-run current A configuration; not launched by cleanup | `../../../configs/recurrent.py` |
| [Smoke checks](smoke/README.md) | Historical pipeline, device, and distributed checks | `configs/`, validation notes |

The recurrence-grid sweep varies evaluation settings of jointly trained checkpoints; it is not a set of independently trained per-cell models. The long baseline's LR selection and winning continuation stay together because they form one experiment with common panel, provenance, and selection receipts. Its LR candidates A/B are unrelated to the architecture variants A/B.

For a new experiment, add `experiments/<category>/<name>/` with a config or `configs/`, a short README describing the question and run commands, and `results/` as its output location. Keep derived artifacts there too. Do not put logs, checkpoints, or generated plots at the repository root, and do not overwrite an earlier experiment to launch a new one.

## Historical artifacts and relocation

The cleanup moved outputs without rewriting checkpoints, raw JSON/CSV, metrics, frozen protocols, or transfer archives. Their embedded paths and source hashes describe the original run. [relocations.json](relocations.json) maps old repository-relative paths to current locations; an old absolute path can be translated by its repository-relative suffix. Reports and maintained commands link to current locations.

The old missing `n_buffer` field means zero on checkpoint load. Every historical training config now pins its original layout explicitly. Resume permits a panel path change only when its saved content hash still matches. This preserves the numerical meaning of old runs while making A the default for new configs.

Completed `HANDOFF.md`, `PLAN.md`, and validation notes are historical records, not instructions to rerun completed experiments. Source-freezing checks deliberately reject resuming a frozen experiment under modified source. To reproduce a completed cloud run exactly, use its retained transfer archive and recorded environment in an isolated directory. To conduct a new run from current source, define a fresh experiment/output directory and freeze it anew.
