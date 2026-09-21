"""Plot retained 5B axis-arm training and validation NLL curves.

The input logs are the experiment-local backups copied from the completed
CUDA runs.  This script does not modify them; it writes only the requested
figure to the study results directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
BACKUPS = ROOT / "runpod_live_backups"
OUT = ROOT / "results" / "training_nll_curves.png"

ARMS = {
    "depth": ("Depth-only (0,3)", "#0072B2"),
    "temporal": ("Temporal-only (3,0)", "#D55E00"),
    "hybrid": ("Hybrid (3,3)", "#009E73"),
}


def load_rows(arm: str) -> tuple[list[dict], list[dict]]:
    path = BACKUPS / arm / "results" / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    train = [row for row in rows if row.get("event") == "train"]
    evaluation = [row for row in rows if row.get("event") == "evaluation"]
    if not train or not evaluation:
        raise ValueError(f"{path} does not contain both train and evaluation rows")
    return train, evaluation


def rolling_median(values: list[float], window: int = 101) -> np.ndarray:
    """Return a centered rolling median without changing the source samples."""

    half = window // 2
    result = np.empty(len(values), dtype=float)
    array = np.asarray(values, dtype=float)
    for index in range(len(values)):
        lo = max(0, index - half)
        hi = min(len(values), index + half + 1)
        result[index] = np.median(array[lo:hi])
    return result


def main() -> None:
    data = {arm: load_rows(arm) for arm in ARMS}
    reference_steps = [row["step"] for row in data["depth"][0]]
    if any([row["step"] for row in train] != reference_steps for train, _ in data.values()):
        raise ValueError("training logs do not share the same update axis")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, (train_ax, late_ax, val_ax) = plt.subplots(
        3,
        1,
        figsize=(10.5, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 0.9, 0.9]},
        constrained_layout=True,
    )

    for arm, (label, color) in ARMS.items():
        train, evaluation = data[arm]
        train_steps = np.asarray([row["step"] for row in train])
        train_nll = [row["nll"] for row in train]
        train_ax.plot(
            train_steps,
            train_nll,
            color=color,
            alpha=0.16,
            linewidth=0.6,
            label=f"{label} — raw",
        )
        train_ax.plot(
            train_steps,
            rolling_median(train_nll),
            color=color,
            linewidth=2.0,
            label=f"{label} — rolling median",
        )
        late_ax.plot(
            train_steps,
            rolling_median(train_nll),
            color=color,
            linewidth=2.0,
            label=label,
        )

        eval_steps = [row["step"] for row in evaluation]
        eval_nll = [row["val_nll"] for row in evaluation]
        val_ax.plot(
            eval_steps,
            eval_nll,
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=2.5,
            label=label,
        )

    train_ax.set_title("5B axis-arm learning curves")
    train_ax.set_ylabel("training NLL")
    train_ax.set_ylim(bottom=0)
    train_ax.grid(True, alpha=0.22)
    train_ax.legend(ncol=2, fontsize=8, frameon=False)

    late_ax.set_title("Late training detail")
    late_ax.set_ylabel("training NLL")
    late_ax.set_xlim(left=20_000)
    late_ax.set_ylim(0.22, 0.36)
    late_ax.grid(True, alpha=0.22)
    late_ax.legend(fontsize=8, frameon=False)

    val_ax.set_xlabel("optimizer updates")
    val_ax.set_ylabel("validation NLL")
    val_ax.set_xlim(left=0)
    val_ax.set_ylim(bottom=0)
    val_ax.grid(True, alpha=0.22)
    val_ax.legend(fontsize=8, frameon=False)

    fig.text(
        0.01,
        0.005,
        "Raw training values are shown faintly; bold curves are 101-point rolling medians. "
        "Validation points are the fixed training-time evaluation panel.",
        fontsize=8,
        color="#555555",
    )
    fig.savefig(OUT, dpi=180)
    print(OUT)


if __name__ == "__main__":
    main()
