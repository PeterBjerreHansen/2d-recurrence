"""Build long-baseline curve tables, optimizer summaries, and static plots."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/long_runs/baseline/results"
CELLS = [(u_t, u_d) for u_t in (0, 1, 3) for u_d in (0, 1, 3)]
SELECTION_STEPS = (0, 250, 500, 750, 1000)
SELECTED_STEPS = (0, 250, 500, 750, 1000, 2000, 5000, 8000, 10000)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_cell(report: dict, cell: tuple[int, int]) -> dict:
    return next(c for c in report["cells"] if (c["u_t"], c["u_d"]) == cell)


def load_grid(run: str, step: int, split: str = "selection") -> dict:
    path = OUT / run / f"grid-{split}-step{step:06d}.json"
    return json.loads(path.read_text())


def grid_rows(run: str, steps: tuple[int, ...], split: str = "selection") -> list[dict]:
    rows = []
    for step in steps:
        report = load_grid(run, step, split)
        for cell in CELLS:
            value = report_cell(report, cell)
            rows.append(
                {
                    "run": run,
                    "split": split,
                    "step": step,
                    "u_t": cell[0],
                    "u_d": cell[1],
                    "nll_mean": value["nll_mean"],
                    "nll_std_placement": value["nll_std"],
                    "accuracy_mean": value["accuracy_mean"],
                    "nll_delta_vs_00": value["nll_delta_vs_00"],
                    "flops_per_sequence": value["estimated_forward_matmul_flops_per_sequence_mean"],
                    "placement_count": value["placement_count"],
                    "target_count": value["target_count"],
                    "batch_fingerprint": report["batch_fingerprint"],
                    "panel_sha256": report["panel_file_sha256"],
                }
            )
    return rows


def training_rows(run: str) -> list[dict]:
    rows = []
    for line in (OUT / run / "metrics.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("event") in {"evaluation", "train"}:
            rows.append(
                {
                    "run": run,
                    "event": event["event"],
                    "step": event["step"],
                    "nll": event.get("nll"),
                    "train_nll": event.get("train_nll"),
                    "val_nll": event.get("val_nll"),
                    "train_accuracy": event.get("train_accuracy"),
                    "val_accuracy": event.get("val_accuracy"),
                    "lr": event.get("lr"),
                    "seconds": event.get("seconds"),
                    "characters_processed": event.get("characters_processed"),
                    "characters_this_update": event.get("characters_this_update"),
                }
            )
    return rows


def clipping_summary(run: str) -> dict:
    events = [
        json.loads(line)
        for line in (OUT / run / "metrics.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "train"
    ]
    norms = [e["grad_norm_pre_clip"] for e in events]
    coefficients = [e["applied_clip_coefficient"] for e in events]
    clipped = [bool(e["gradients_clipped"]) for e in events]
    elapsed = sum(e["seconds"] for e in events)
    chars = events[-1]["characters_processed"] if events else 0
    sorted_norms = sorted(norms)
    p95 = sorted_norms[min(len(sorted_norms) - 1, math.floor(0.95 * len(sorted_norms)))] if norms else None
    return {
        "run": run,
        "optimizer_updates": len(events),
        "clipped_updates": sum(clipped),
        "clipped_fraction": sum(clipped) / len(events) if events else None,
        "grad_norm_pre_clip_mean": sum(norms) / len(norms) if norms else None,
        "grad_norm_pre_clip_max": max(norms) if norms else None,
        "grad_norm_pre_clip_p95": p95,
        "applied_clip_coefficient_mean": sum(coefficients) / len(coefficients) if coefficients else None,
        "applied_clip_coefficient_min": min(coefficients) if coefficients else None,
        "update_compute_seconds_sum": elapsed,
        "characters_processed": chars,
        "characters_per_update": chars / len(events) if events else None,
    }


def plot_grid(rows: list[dict], runs: list[str], filename: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 4.5), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        for cell in CELLS:
            values = [r for r in rows if r["run"] == run and (r["u_t"], r["u_d"]) == cell]
            values.sort(key=lambda r: r["step"])
            width = 2.0 if cell in {(1, 0), (3, 3)} else 1.0
            ax.plot([r["step"] for r in values], [r["nll_mean"] for r in values], marker="o", linewidth=width,
                    label=f"({cell[0]},{cell[1]})")
        ax.set_title(run)
        ax.set_xlabel("optimizer updates")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("validation NLL")
    axes[-1].legend(ncol=3, fontsize=8, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    selection = grid_rows("lr3e-4", SELECTION_STEPS) + grid_rows("lr1e-4", SELECTION_STEPS)
    selected = grid_rows("lr3e-4", SELECTED_STEPS)
    write_csv(OUT / "selection-grid-curves.csv", selection)
    write_csv(OUT / "selected-grid-curves.csv", selected)
    write_csv(OUT / "training-events.csv", training_rows("lr3e-4") + training_rows("lr1e-4"))
    summaries = [clipping_summary(run) for run in ("lr3e-4", "lr1e-4")]
    (OUT / "clipping-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    write_csv(OUT / "clipping-summary.csv", summaries)

    plot_grid(selection, ["lr3e-4", "lr1e-4"], "selection-nll-vs-updates.png", "Long baseline selection: nine-cell NLL")
    plot_grid(selected, ["lr3e-4"], "selected-nll-vs-updates.png", "Selected long baseline: nine-cell NLL")

    reference = {(r["step"]): r["nll_mean"] for r in selected if (r["u_t"], r["u_d"]) == (1, 0)}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for cell in CELLS:
        if cell == (1, 0):
            continue
        values = [r for r in selected if (r["u_t"], r["u_d"]) == cell]
        values.sort(key=lambda r: r["step"])
        width = 2.0 if cell in {(0, 0), (3, 0), (3, 3)} else 1.0
        ax.plot([r["step"] for r in values], [r["nll_mean"] - reference[r["step"]] for r in values], marker="o",
                linewidth=width, label=f"({cell[0]},{cell[1]})")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("optimizer updates")
    ax.set_ylabel("NLL difference versus (1,0)")
    ax.set_title("Selected long baseline: refinement relative to (1,0)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "selected-delta-nll-vs-10.png", dpi=180)
    plt.close(fig)

    evaluations = [r for r in training_rows("lr3e-4") + training_rows("lr1e-4") if r["event"] == "evaluation"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for run in ("lr3e-4", "lr1e-4"):
        values = [r for r in evaluations if r["run"] == run]
        values.sort(key=lambda r: r["step"])
        ax.plot([r["step"] for r in values], [r["train_nll"] for r in values], label=f"{run} train")
        ax.plot([r["step"] for r in values], [r["val_nll"] for r in values], linestyle="--", label=f"{run} validation")
    ax.set_xlabel("optimizer updates")
    ax.set_ylabel("routine NLL")
    ax.set_title("Training and routine selection-panel validation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "train-validation-nll-vs-updates.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
