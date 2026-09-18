"""Build Stage 9 curve tables and static plots from retained grid reports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/sweeps/recurrence_grid/results/curves"
STEPS = (0, 25, 50, 75, 100)
SEEDS = (1337, 1338)


def read_grid(seed: int, step: int) -> dict:
    path = ROOT / f"experiments/sweeps/recurrence_grid/results/seed{seed}" / f"grid-step{step:06d}.json"
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    reports = {(seed, step): read_grid(seed, step) for seed in SEEDS for step in STEPS}
    cells = [(c["u_t"], c["u_d"]) for c in reports[(SEEDS[0], STEPS[0])]["cells"]]

    curve_rows = []
    placement_rows = []
    diagnostic_rows = []
    for seed in SEEDS:
        for step in STEPS:
            for cell in reports[(seed, step)]["cells"]:
                key = (cell["u_t"], cell["u_d"])
                placements = cell["placements"]
                curve_rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "u_t": key[0],
                        "u_d": key[1],
                        "core_passes": cell["core_passes"],
                        "placement_count": cell["placement_count"],
                        "nll_mean": cell["nll_mean"],
                        "nll_std_placement": cell["nll_std"],
                        "accuracy_mean": cell["accuracy_mean"],
                        "accuracy_std_placement": cell["accuracy_std"],
                        "nll_delta_vs_00": cell["nll_delta_vs_00"],
                        "prediction_change_rate_mean": cell[
                            "prediction_change_rate_mean"
                        ],
                    }
                )
                for placement in placements:
                    placement_rows.append(
                        {
                            "seed": seed,
                            "step": step,
                            "u_t": key[0],
                            "u_d": key[1],
                            **{
                                name: placement[name]
                                for name in (
                                    "mask_seed",
                                    "nll",
                                    "accuracy",
                                    "prediction_change_rate",
                                    "temporal_write_mask",
                                    "depth_write_mask",
                                )
                            },
                        }
                    )
                diag = cell["diagnostics"]
                diagnostic_rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "u_t": key[0],
                        "u_d": key[1],
                        "mask_seed": diag["mask_seed"],
                        "loss": diag["loss"],
                        "core_output_rms": "|".join(map(str, diag["core_output_rms"])),
                        "source_output_rms": "|".join(
                            map(str, diag["source_output_rms"])
                        ),
                        **diag["gradient_l2"],
                    }
                )

    curve_fields = [
        "seed",
        "step",
        "u_t",
        "u_d",
        "core_passes",
        "placement_count",
        "nll_mean",
        "nll_std_placement",
        "accuracy_mean",
        "accuracy_std_placement",
        "nll_delta_vs_00",
        "prediction_change_rate_mean",
    ]
    write_csv(OUT / "grid-curves-by-seed.csv", curve_rows, curve_fields)

    placement_fields = [
        "seed",
        "step",
        "u_t",
        "u_d",
        "mask_seed",
        "nll",
        "accuracy",
        "prediction_change_rate",
        "temporal_write_mask",
        "depth_write_mask",
    ]
    write_csv(OUT / "placement-results.csv", placement_rows, placement_fields)

    diagnostic_fields = [
        "seed",
        "step",
        "u_t",
        "u_d",
        "mask_seed",
        "loss",
        "core_output_rms",
        "source_output_rms",
        "embedding_and_head",
        "prelude",
        "core",
        "source",
        "coda",
        "temporal_mixer",
        "depth_mixer",
    ]
    write_csv(OUT / "diagnostics.csv", diagnostic_rows, diagnostic_fields)

    aggregate_rows = []
    for step in STEPS:
        for u_t, u_d in cells:
            selected = [
                row
                for row in curve_rows
                if row["step"] == step and row["u_t"] == u_t and row["u_d"] == u_d
            ]
            nlls = [row["nll_mean"] for row in selected]
            deltas = [row["nll_delta_vs_00"] for row in selected]
            accuracies = [row["accuracy_mean"] for row in selected]
            mean = lambda values: sum(values) / len(values)
            pop_sd = lambda values: (
                sum((value - mean(values)) ** 2 for value in values) / len(values)
            ) ** 0.5
            aggregate_rows.append(
                {
                    "step": step,
                    "u_t": u_t,
                    "u_d": u_d,
                    "nll_seed_mean": mean(nlls),
                    "nll_seed_std": pop_sd(nlls),
                    "delta_vs_00_seed_mean": mean(deltas),
                    "delta_vs_00_seed_std": pop_sd(deltas),
                    "accuracy_seed_mean": mean(accuracies),
                    "accuracy_seed_std": pop_sd(accuracies),
                }
            )
    write_csv(
        OUT / "grid-curves-seed-summary.csv",
        aggregate_rows,
        [
            "step",
            "u_t",
            "u_d",
            "nll_seed_mean",
            "nll_seed_std",
            "delta_vs_00_seed_mean",
            "delta_vs_00_seed_std",
            "accuracy_seed_mean",
            "accuracy_seed_std",
        ],
    )

    generation_rows = []
    generation_paths = sorted(
        list((ROOT / "experiments/sweeps/recurrence_grid/results/seed1337").glob("generation-*.json"))
        + list((ROOT / "experiments/sweeps/recurrence_grid/results/seed1338").glob("generation-*.json"))
        + [ROOT / "experiments/smoke/results/baseline-pilot/generation-stage09.json"]
    )
    for path in generation_paths:
        report = json.loads(path.read_text())
        samples = report["samples"]
        reasons = Counter(sample["termination_reason"] for sample in samples)
        attempted = [sample["attempted_moves"] for sample in samples]
        legal = [sample["legal_moves"] for sample in samples]
        settings = report.get("settings", {})
        run = path.parent.name
        generation_rows.append(
            {
                "run": run,
                "file": str(path.relative_to(ROOT)),
                "execution": settings.get("execution", "baseline"),
                "u_t": settings.get("u_t")
                if settings.get("execution") == "training_graph"
                else "",
                "u_d": settings.get("u_d")
                if settings.get("execution") == "training_graph"
                else "",
                "samples": report["summary"]["samples"],
                "completed_move_legality": report["summary"].get(
                    "completed_move_legality"
                ),
                "mean_legal_continuation": report["summary"].get(
                    "mean_legal_continuation"
                ),
                "mean_attempted_moves": sum(attempted) / len(attempted),
                "mean_sample_legal_moves": sum(legal) / len(legal),
                "pgn_parse_success_rate": report["summary"].get(
                    "pgn_parse_success_rate"
                ),
                "valid_termination_rate": report["summary"].get(
                    "valid_termination_rate"
                ),
                "malformed_move": reasons.get("malformed_move", 0),
                "illegal_move": reasons.get("illegal_move", 0),
                "generation_limit": reasons.get("generation_limit", 0),
            }
        )
    write_csv(
        OUT / "generation-summary.csv",
        generation_rows,
        list(generation_rows[0].keys()),
    )

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for u_t, u_d in cells:
            rows = [
                row
                for row in curve_rows
                if row["seed"] == seed and row["u_t"] == u_t and row["u_d"] == u_d
            ]
            label = f"({u_t},{u_d})"
            width = 2.0 if (u_t, u_d) in ((0, 0), (3, 0), (0, 3), (3, 3)) else 1.0
            ax.plot(
                [row["step"] for row in rows],
                [row["nll_mean"] for row in rows],
                marker="o",
                linewidth=width,
                label=label,
            )
        ax.set_title(f"Training seed {seed}")
        ax.set_xlabel("optimizer updates")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("validation NLL")
    axes[1].legend(ncol=3, fontsize=8, loc="upper right")
    fig.suptitle("Stage 9 fixed-batch NLL by execution setting")
    fig.tight_layout()
    fig.savefig(OUT / "nll-vs-updates.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    delta_cells = [cell for cell in cells if cell != (0, 0)]
    for ax, seed in zip(axes, SEEDS):
        for u_t, u_d in delta_cells:
            rows = [
                row
                for row in curve_rows
                if row["seed"] == seed and row["u_t"] == u_t and row["u_d"] == u_d
            ]
            width = 2.0 if (u_t, u_d) in ((3, 0), (0, 3), (3, 3)) else 1.0
            ax.plot(
                [row["step"] for row in rows],
                [row["nll_delta_vs_00"] for row in rows],
                marker="o",
                linewidth=width,
                label=f"({u_t},{u_d})",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Training seed {seed}")
        ax.set_xlabel("optimizer updates")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("NLL difference vs (0,0)")
    axes[1].legend(ncol=3, fontsize=8, loc="lower right")
    fig.suptitle("Stage 9 recurrence effect at fixed trained weights")
    fig.tight_layout()
    fig.savefig(OUT / "delta-nll-vs-00.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
