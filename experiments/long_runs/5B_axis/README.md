# 5B recurrence-axis study

This study defines four fresh, data-matched training arms at a five-billion
target-character horizon:

| Arm | Execution mode | Evaluation setting |
| --- | --- | --- |
| `transformer_5B` | ordinary eight-block transformer | `(0,0)` |
| `temporal_5B` | temporal recurrence only | `(3,0)` |
| `depth_5B` | depth recurrence only | `(0,3)` |
| `hybrid_5B` | temporal and depth recurrence | `(3,3)` |

Each arm processes 48,876 optimizer updates and 5,000,014,800 target
characters at effective batch 100 and context length 1,023.  All arms use the
same pinned serious profile, AdamW settings, final-only objective, BF16 CUDA
training, and fresh optimizer/RNG state.  The learning-rate schedule is
defined for the full 48,876-update horizon: 978 warmup updates followed by
cosine decay to `3e-5` at the endpoint.

The recurrent probability tables are the reviewed axis-ablation definitions in
`experiments/long_runs/5B_axis/configs/common.py`. The temporal-only
and depth-only arms use the same distribution over active pass counts (`0`, `1`,
and `3`). The hybrid arm uses the fixed symmetric near-diagonal table from that
module: it preserves the same distribution over `max(U_T, U_D)`, leaves 80% of
each nonzero bucket on the diagonal, and exposes asymmetric state-age cases.
Therefore this is matched by core-pass distribution and data exposure, not by
the sum of axis-specific writes. The hybrid arm remains somewhat more
expensive per update, and the exact compute difference is reported separately.

Intermediate checkpoints are retained at steps `0, 100, 250, 500, 1,000,
2,500, 5,000, 10,000, 25,000, 40,000`, and `48,876`.  The study definitions
do not launch training.  The explicit `freeze` command creates only the
experiment's protocol and panel receipts; training creates the arm result
directories.

## Freeze and run protocol

The canonical source is the branch and working tree recorded by `run.py
freeze`. The freeze receipt includes the branch, base commit, working-tree
patch identity, source hashes (including untracked implementation files),
resolved configurations, dataset manifest, panel hash, and local environment.
It is written to `results/protocol.json`; the panel is written to
`results/panel.json`. The working tree may not change after the freeze. A
source/data bundle for transfer to Verda can be built with `package.py`; it
includes the frozen source files, protocol, panel, and complete pinned dataset
but never credentials or training outputs.

From the repository root:

```sh
uv run python -m experiments.long_runs.5B_axis.run freeze
uv run python -m experiments.long_runs.5B_axis.run status
uv run python -m experiments.long_runs.5B_axis.package --output /path/to/5B_axis_source.tar.gz
```

On the prepared Verda VM, after extracting the verified bundle:

```sh
uv sync --frozen --python 3.11
uv run pytest -q
uv run python -m experiments.long_runs.5B_axis.run preflight
uv run python -m experiments.long_runs.5B_axis.run study --evaluate
```

`study` runs the arms sequentially in transformer, temporal-only, depth-only,
hybrid order. The transfer bundle also contains `TRANSFER_MANIFEST.json` so
the runner can verify the source without Git metadata. Repeating a command
resumes only a compatible `ckpt.pt`; a
changed source, dataset, panel, or resolved configuration is rejected. The
runner does not provision Verda or silently substitute another GPU.

## Configuration files

- `transformer_5B/config.py`
- `temporal_5B/config.py`
- `depth_5B/config.py`
- `hybrid_5B/config.py`
- `configs/common.py`, `configs/temporal.py`, `configs/depth.py`, and
  `configs/hybrid_matched.py` are the shared schedule and resolved-config
  entry points for this study.

The shared constructor is `study.py`; the executable freeze/resume logic is in
`run.py`. Do not resume a shorter run into this study: each arm has its
own full-horizon schedule and starts from scratch.  A resumed run is valid
only when its checkpoint is under the matching arm directory and its resolved
configuration is unchanged.
