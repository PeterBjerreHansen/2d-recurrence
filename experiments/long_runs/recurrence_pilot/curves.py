"""Summarize and plot the exact-row long Stage 9 pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "experiments/long_runs/recurrence_pilot/results/all_rows.json"
OUT = ROOT / "experiments/long_runs/recurrence_pilot/results/curves"
CELLS = [(u_t, u_d) for u_t in (0, 1, 3) for u_d in (0, 1, 3)]
SELECTED = [(0, 0), (1, 0), (0, 3), (3, 0), (3, 3)]


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    report = json.loads(REPORT.read_text())
    curve_rows = []
    for checkpoint in report["checkpoints"]:
        seed = checkpoint["training_seed"]
        step = checkpoint["checkpoint_step"]
        if checkpoint["architecture"] != "recurrent":
            continue
        for cell in checkpoint["report"]["cells"]:
            curve_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "u_t": cell["u_t"],
                    "u_d": cell["u_d"],
                    "nll_mean": cell["nll_mean"],
                    "nll_std_placement": cell["nll_std"],
                    "accuracy_mean": cell["accuracy_mean"],
                    "nll_delta_vs_00": cell["nll_delta_vs_00"],
                    "flops_per_sequence": cell[
                        "estimated_forward_matmul_flops_per_sequence_mean"
                    ],
                    "placement_count": cell["placement_count"],
                }
            )
    fields = list(curve_rows[0].keys())
    write_csv(OUT / "grid-curves-by-seed.csv", curve_rows, fields)

    def select(seed, step, cell):
        return next(
            row
            for row in curve_rows
            if row["seed"] == seed
            and row["step"] == step
            and (row["u_t"], row["u_d"]) == cell
        )

    steps = sorted({row["step"] for row in curve_rows})
    summary_rows = []
    comparison_rows = []
    for step in steps:
        for cell in CELLS:
            rows = [select(seed, step, cell) for seed in (1337, 1338)]
            nlls = [row["nll_mean"] for row in rows]
            deltas = [row["nll_delta_vs_00"] for row in rows]
            mean = sum(nlls) / len(nlls)
            sd = (sum((value - mean) ** 2 for value in nlls) / len(nlls)) ** 0.5
            delta_mean = sum(deltas) / len(deltas)
            delta_sd = (sum((value - delta_mean) ** 2 for value in deltas) / len(deltas)) ** 0.5
            summary_rows.append(
                {
                    "step": step,
                    "u_t": cell[0],
                    "u_d": cell[1],
                    "nll_seed_mean": mean,
                    "nll_seed_std": sd,
                    "delta_vs_00_seed_mean": delta_mean,
                    "delta_vs_00_seed_std": delta_sd,
                }
            )
        for seed in (1337, 1338):
            reference = select(seed, step, (1, 0))
            row = {"seed": seed, "step": step}
            for cell in SELECTED:
                current = select(seed, step, cell)
                tag = f"{cell[0]}-{cell[1]}"
                row[f"nll_{tag}"] = current["nll_mean"]
                row[f"delta00_{tag}"] = current["nll_delta_vs_00"]
                row[f"delta10_{tag}"] = current["nll_mean"] - reference["nll_mean"]
                row[f"flops_{tag}"] = current["flops_per_sequence"]
            row["flops_ratio_33_vs_10"] = row["flops_3-3"] / row["flops_1-0"]
            comparison_rows.append(row)
    write_csv(
        OUT / "grid-curves-seed-summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )
    write_csv(
        OUT / "selected-comparisons.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )

    training_rows = []
    for seed in (1337, 1338):
        metrics_path = ROOT / f"experiments/long_runs/recurrence_pilot/results/seed{seed}" / "metrics.jsonl"
        for line in metrics_path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "evaluation":
                training_rows.append(
                    {
                        "seed": seed,
                        "step": event["step"],
                        "train_nll": event["train_nll"],
                        "val_nll": event["val_nll"],
                        "train_accuracy": event["train_accuracy"],
                        "val_accuracy": event["val_accuracy"],
                    }
                )
    write_csv(OUT / "training-evaluations.csv", training_rows, list(training_rows[0].keys()))

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, seed in zip(axes, (1337, 1338)):
        for cell in CELLS:
            rows = [
                select(seed, step, cell)
                for step in steps
            ]
            width = 2.0 if cell in SELECTED else 1.0
            ax.plot(
                steps,
                [row["nll_mean"] for row in rows],
                marker="o",
                linewidth=width,
                label=f"({cell[0]},{cell[1]})",
            )
        ax.set_title(f"Training seed {seed}")
        ax.set_xlabel("optimizer updates")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("validation NLL; all 41 rows")
    axes[1].legend(ncol=3, fontsize=8, loc="upper right")
    fig.suptitle("Stage 9 long pilot: exact-row NLL")
    fig.tight_layout()
    fig.savefig(OUT / "nll-vs-updates.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, seed in zip(axes, (1337, 1338)):
        reference = {(row["step"]): row["nll_mean"] for row in [select(seed, step, (1, 0)) for step in steps]}
        for cell in [cell for cell in CELLS if cell != (1, 0)]:
            rows = [select(seed, step, cell) for step in steps]
            width = 2.0 if cell in ((0, 0), (0, 3), (3, 0), (3, 3)) else 1.0
            ax.plot(
                steps,
                [row["nll_mean"] - reference[row["step"]] for row in rows],
                marker="o",
                linewidth=width,
                label=f"({cell[0]},{cell[1]})",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Training seed {seed}")
        ax.set_xlabel("optimizer updates")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("NLL difference versus (1,0)")
    axes[1].legend(ncol=3, fontsize=8, loc="lower right")
    fig.suptitle("Stage 9 long pilot: repeated-refinement effect")
    fig.tight_layout()
    fig.savefig(OUT / "delta-nll-vs-10.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
