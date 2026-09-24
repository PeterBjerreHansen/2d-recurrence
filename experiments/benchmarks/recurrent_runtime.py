"""Short, repeatable throughput sweeps for the recurrent 20B training setup.

Run from the repository root.  Each condition holds effective batch size at
100, uses the late hybrid schedule (P(max(U_T,U_D)=1)=.2, =3=.8), and writes
its metrics and checkpoint beneath the selected output directory.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

if os.environ.get("RECURRENCE_TORCH_DETERMINISTIC") == "1":
    torch.use_deterministic_algorithms(True)

from experiments.serious import base
from recurrence.schedule import build_update_probability_matrix
from train import get_lr, train


DEFAULT_RESULTS = ROOT / "experiments/benchmarks/recurrent_runtime/results"
LATE_HYBRID = build_update_probability_matrix(
    [0, 1, 3], "hybrid", [0.0, 0.2, 0.8], hybrid_diagonal_mass=0.8)
CURRICULUM_PHASES = (
    (0, (0.10, 0.80, 0.10)),
    (0.05, (0.05, 0.60, 0.35)),
    (0.20, (0.00, 0.40, 0.60)),
    (0.50, (0.00, 0.20, 0.80)),
)


def make_config(output, updates, batch_size, accumulation, *, profile="late_hybrid",
                init_from="scratch"):
    config = base("recurrent")
    if profile == "late_hybrid":
        phases = [{"start_step": 0, "update_probabilities": LATE_HYBRID}]
        lr_schedule, warmup, decay_start = "constant", 0, max(1, updates // 2)
    elif profile == "wsd_curriculum":
        phases = [
            {"start_step": round(fraction * updates),
             "update_probabilities": build_update_probability_matrix(
                 [0, 1, 3], "hybrid", probabilities, hybrid_diagonal_mass=0.8)}
            for fraction, probabilities in CURRICULUM_PHASES
        ]
        lr_schedule = "wsd"
        warmup = max(1, round(0.01 * updates))
        decay_start = round(0.90 * updates)
    else:
        raise ValueError(f"Unknown benchmark profile: {profile}")
    config.update(
        out_dir=str(output), eval_panel_path="", init_from=init_from,
        max_iters=updates, batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=3e-4, min_lr=3e-5,
        lr_schedule=lr_schedule, decay_lr=(lr_schedule != "constant"),
        warmup_iters=warmup, lr_decay_start=decay_start,
        lr_decay_iters=updates, training_budget_seconds=0.0,
        update_probabilities=[],
        update_probability_schedule={
            "type": "piecewise_constant",
            "phases": phases,
        },
        recurrence_mode="hybrid", eval_u_t=3, eval_u_d=3,
        eval_interval=(updates // 2 if profile == "wsd_curriculum" else updates + 1),
        eval_iters=2, log_interval=1,
        compile=False, keep_checkpoints=(profile == "wsd_curriculum"),
        checkpoint_steps=([updates // 2, updates] if profile == "wsd_curriculum" else None),
    )
    return config


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def worker(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    config = make_config(output, args.updates, args.batch_size, args.accumulation,
                         profile=args.profile,
                         init_from="resume" if args.resume else "scratch")
    torch.cuda.reset_peak_memory_stats()
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    checkpoint = train(config)
    wall_seconds = time.perf_counter() - started
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    events = read_events(output / "metrics.jsonl")
    updates = [event for event in events if event.get("event") == "train"]
    evaluations = [event for event in events if event.get("event") == "evaluation"]
    steady = [event for event in updates if event.get("step", 0) > min(10, args.updates // 10)]
    selected = steady or updates
    schedules = [item for event in updates for item in event.get("schedules", [])]
    count_histogram = {}
    pair_histogram = {}
    for item in schedules:
        maximum = max(item["u_t"], item["u_d"])
        count_histogram[str(maximum)] = count_histogram.get(str(maximum), 0) + 1
        pair = f'{item["u_t"]},{item["u_d"]}'
        pair_histogram[pair] = pair_histogram.get(pair, 0) + 1
    result = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "updates": payload["iter_num"],
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation,
        "effective_batch_size": args.batch_size * args.accumulation,
        "initial_evaluation": next((event for event in evaluations if event.get("step") == 0), None),
        "final_evaluation": next((event for event in reversed(evaluations)
                                  if event.get("step") == payload["iter_num"]), None),
        "late_hybrid_probability_matrix": LATE_HYBRID,
        "wall_seconds": wall_seconds,
        "training_seconds": payload.get("training_seconds"),
        "steady_updates_per_second": len(selected) / sum(e["seconds"] for e in selected),
        "steady_characters_per_second": statistics.mean(
            e["characters_per_second"] for e in selected),
        "all_updates_characters_per_second": statistics.mean(
            e["characters_per_second"] for e in updates),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "mean_grad_norm_pre_clip": statistics.mean(
            e["grad_norm_pre_clip"] for e in updates),
        "median_grad_norm_pre_clip": statistics.median(
            e["grad_norm_pre_clip"] for e in updates),
        "clipped_fraction": statistics.mean(
            float(e["gradients_clipped"]) for e in updates),
        "mean_applied_clip_coefficient": statistics.mean(
            e["applied_clip_coefficient"] for e in updates),
        "sampled_schedule_count_histogram": count_histogram,
        "sampled_schedule_pair_histogram": pair_histogram,
    }
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def run_one(output, updates, batch_size, accumulation, *, profile="late_hybrid",
            resume=False, env=None):
    command = [sys.executable, "-m", "experiments.benchmarks.recurrent_runtime", "--worker",
               "--output", str(output), "--updates", str(updates),
               "--batch-size", str(batch_size), "--accumulation", str(accumulation),
               "--profile", profile]
    if resume:
        command.append("--resume")
    log = output.parent / f"{output.name}.stdout.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(f"worker failed ({completed.returncode}); {log}\n{tail}")
    return json.loads((output / "benchmark.json").read_text())


def microbatch(args):
    root = Path(args.results) / args.run_id / "microbatch"
    outcomes = []
    for batch_size, accumulation in ((5, 20), (10, 10), (20, 5), (25, 4)):
        label = f"batch{batch_size}_accum{accumulation}"
        output = root / label
        outcomes.append(run_one(output, args.updates, batch_size, accumulation))
    reference = next(item for item in outcomes if item["batch_size"] == 5)
    eligible = [item for item in outcomes if item["gradient_accumulation_steps"] >= 4]
    fastest = max(eligible, key=lambda item: item["steady_characters_per_second"])
    result = {
        "stage": "microbatch",
        "run_id": args.run_id,
        "updates_per_condition": args.updates,
        "effective_batch_size": 100,
        "reference": reference,
        "conditions": outcomes,
        "fastest_with_at_least_four_schedule_draws_per_update": fastest,
        "selection_note": "Short-run throughput ranking only; compare loss and clipping diagnostics before adopting.",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def concurrency(args):
    root = Path(args.results) / args.run_id / "concurrency"
    pipe_dir = Path("/tmp/recurrent-mps-pipe")
    log_dir = Path("/tmp/recurrent-mps-log")
    pipe_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    outcomes = []
    for process_count in (1, 2, 4):
        trial = root / f"processes{process_count}"
        trial.mkdir(parents=True, exist_ok=False)
        commands = []
        for index in range(process_count):
            output = trial / f"job{index + 1}"
            log = trial / f"job{index + 1}.stdout.log"
            stream = log.open("w")
            command = [sys.executable, "-m", "experiments.benchmarks.recurrent_runtime", "--worker",
                       "--output", str(output), "--updates", str(args.updates),
                       "--batch-size", str(args.batch_size),
                       "--accumulation", str(args.accumulation),
                       "--profile", "late_hybrid"]
            commands.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                              stderr=subprocess.STDOUT), stream, output, log))
        group_started = time.perf_counter()
        failures = []
        for process, stream, output, log in commands:
            status = process.wait()
            stream.close()
            if status:
                failures.append(f"exit {status}: {log}")
        group_seconds = time.perf_counter() - group_started
        if failures:
            raise RuntimeError("concurrent worker failure: " + "; ".join(failures))
        jobs = [json.loads((output / "benchmark.json").read_text())
                for _, _, output, _ in commands]
        total_characters = sum(item["updates"] * item["effective_batch_size"] * 1023
                               for item in jobs)
        training_window_seconds = max(item["training_seconds"] for item in jobs)
        outcomes.append({
            "processes": process_count,
            "group_wall_seconds": group_seconds,
            "end_to_end_aggregate_characters_per_second": total_characters / group_seconds,
            "aggregate_training_characters_per_second": total_characters / training_window_seconds,
            "end_to_end_per_process_characters_per_second": [
                item["updates"] * item["effective_batch_size"] * 1023 / item["wall_seconds"]
                for item in jobs],
            "per_process_training_characters_per_second": [
                item["updates"] * item["effective_batch_size"] * 1023 / item["training_seconds"]
                for item in jobs],
            "jobs": jobs,
        })
    training_baseline = outcomes[0]["aggregate_training_characters_per_second"]
    end_to_end_baseline = outcomes[0]["end_to_end_aggregate_characters_per_second"]
    for item in outcomes:
        item["multiplex_gain"] = (
            item["aggregate_training_characters_per_second"] / training_baseline)
        item["end_to_end_multiplex_gain"] = (
            item["end_to_end_aggregate_characters_per_second"] / end_to_end_baseline)
    result = {
        "stage": "concurrency",
        "run_id": args.run_id,
        "updates_per_process": args.updates,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation,
        "effective_batch_size": 100,
        "cuda_mps": True,
        "conditions": outcomes,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def _tree_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and
                left.dtype == right.dtype and left.shape == right.shape and
                torch.equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and
                left.keys() == right.keys() and
                all(_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(_tree_equal(a, b) for a, b in zip(left, right)))
    try:
        comparison = left == right
        return bool(comparison.all()) if hasattr(comparison, "all") else bool(comparison)
    except (TypeError, ValueError):
        return False


def _tree_max_abs_diff(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape:
        if left.numel() == 0:
            return 0.0
        if not (left.is_floating_point() or left.is_complex()):
            return 0.0 if torch.equal(left, right) else float("inf")
        return float((left.float() - right.float()).abs().max().item())
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return max((_tree_max_abs_diff(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and type(left) is type(right) and len(left) == len(right):
        return max((_tree_max_abs_diff(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if _tree_equal(left, right) else float("inf")


def sanity(args):
    root = Path(args.results) / args.run_id / "sanity"
    baseline_dir = root / "baseline"
    resumed_dir = root / "resumed_from_midpoint"
    midpoint = args.updates // 2
    env = os.environ.copy()
    if args.deterministic:
        # Bitwise resume comparison needs deterministic CUDA kernels; without them,
        # two uninterrupted runs also diverge and the check cannot isolate resume bugs.
        env["RECURRENCE_TORCH_DETERMINISTIC"] = "1"
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    baseline = run_one(baseline_dir, args.updates, args.batch_size, args.accumulation,
                       profile="wsd_curriculum", env=env)
    midpoint_checkpoint = baseline_dir / f"ckpt-step{midpoint:06d}.pt"
    if not midpoint_checkpoint.is_file():
        raise FileNotFoundError(f"Missing midpoint checkpoint: {midpoint_checkpoint}")
    resumed_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(midpoint_checkpoint, resumed_dir / "ckpt.pt")
    resumed = run_one(resumed_dir, args.updates, args.batch_size, args.accumulation,
                      profile="wsd_curriculum", resume=True, env=env)
    if baseline["deterministic_algorithms"] != resumed["deterministic_algorithms"]:
        raise RuntimeError("Baseline and resumed workers used different determinism settings")

    baseline_checkpoint = torch.load(baseline_dir / "ckpt.pt", map_location="cpu",
                                     weights_only=False)
    resumed_checkpoint = torch.load(resumed_dir / "ckpt.pt", map_location="cpu",
                                    weights_only=False)
    exact = {
        key: _tree_equal(baseline_checkpoint[key], resumed_checkpoint[key])
        for key in ("model", "optimizer", "scaler", "recurrence_sampler", "rng_by_rank")
    }
    baseline_events = [event for event in read_events(baseline_dir / "metrics.jsonl")
                       if event.get("event") == "train"]
    resumed_events = [event for event in read_events(resumed_dir / "metrics.jsonl")
                      if event.get("event") == "train"]
    baseline_by_step = {event["step"]: event for event in baseline_events}
    resumed_by_step = {event["step"]: event for event in resumed_events}
    compare_keys = (
        "lr", "nll", "final_nll", "grad_norm_pre_clip",
        "applied_clip_coefficient", "gradients_clipped", "schedules",
    )
    trajectory_differences = 0
    maximum_metric_delta = 0.0
    for step in range(midpoint + 1, args.updates + 1):
        left, right = baseline_by_step.get(step), resumed_by_step.get(step)
        if left is None or right is None:
            trajectory_differences += 1
            continue
        for key in compare_keys:
            if not _tree_equal(left.get(key), right.get(key)):
                trajectory_differences += 1
                maximum_metric_delta = max(
                    maximum_metric_delta,
                    _tree_max_abs_diff(left.get(key), right.get(key)))

    config = make_config(baseline_dir, args.updates, args.batch_size, args.accumulation,
                         profile="wsd_curriculum")
    lr_errors = [abs(event["lr"] - get_lr(event["step"] - 1, config))
                 for event in baseline_events]
    phase_starts = [round(fraction * args.updates) for fraction, _ in CURRICULUM_PHASES]
    phase_ends = phase_starts[1:] + [args.updates]
    phase_histograms = []
    for phase_index, (start, end) in enumerate(zip(phase_starts, phase_ends)):
        histogram = {"0": 0, "1": 0, "3": 0}
        clipped = []
        for event in baseline_events:
            update_index = event["step"] - 1
            if start <= update_index < end:
                clipped.append(float(event["gradients_clipped"]))
                for sample in event.get("schedules", []):
                    key = str(max(sample["u_t"], sample["u_d"]))
                    histogram[key] += 1
        phase_histograms.append({
            "start_step": start,
            "end_step_exclusive": end,
            "target_max_update_probabilities": list(CURRICULUM_PHASES[phase_index][1]),
            "sampled_max_update_histogram": histogram,
            "clipped_fraction": statistics.mean(clipped) if clipped else None,
        })
    lr_points = {}
    for update_index in (0, config["warmup_iters"] - 1, config["warmup_iters"],
                         config["lr_decay_start"], args.updates - 1):
        lr_points[str(update_index)] = get_lr(update_index, config)
    result = {
        "stage": "sanity",
        "run_id": args.run_id,
        "updates": args.updates,
        "resume_midpoint": midpoint,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation,
        "effective_batch_size": args.batch_size * args.accumulation,
        "deterministic_algorithms": baseline["deterministic_algorithms"],
        "learning_rate_schedule": "wsd",
        "warmup_updates": config["warmup_iters"],
        "decay_start_update": config["lr_decay_start"],
        "learning_rates_at_boundary_indices": lr_points,
        "maximum_observed_lr_formula_error": max(lr_errors, default=0.0),
        "curriculum_phase_histograms": phase_histograms,
        "resume_exact_matches": exact,
        "resume_max_model_abs_difference": _tree_max_abs_diff(
            baseline_checkpoint["model"], resumed_checkpoint["model"]),
        "resume_max_optimizer_abs_difference": _tree_max_abs_diff(
            baseline_checkpoint["optimizer"], resumed_checkpoint["optimizer"]),
        "resume_training_metric_differences": trajectory_differences,
        "resume_max_metric_delta": maximum_metric_delta,
        "exact_resume": all(exact.values()) and trajectory_differences == 0,
        "baseline_benchmark": baseline,
        "resumed_benchmark": resumed,
    }
    (root / "sanity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("microbatch", "concurrency", "sanity"))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--accumulation", type=int, default=20)
    parser.add_argument("--output")
    parser.add_argument("--profile", choices=("late_hybrid", "wsd_curriculum"),
                        default="late_hybrid")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deterministic", action="store_true",
                        help="Sanity stage: run both workers with deterministic CUDA kernels")
    args = parser.parse_args()
    if args.worker:
        if not args.output:
            parser.error("--output is required in worker mode")
        worker(args)
    elif args.stage == "microbatch":
        if args.updates < 11:
            parser.error("Use at least 11 updates to report steady-state timing")
        microbatch(args)
    elif args.stage == "concurrency":
        if args.updates < 11:
            parser.error("Use at least 11 updates to report steady-state timing")
        concurrency(args)
    elif args.stage == "sanity":
        if args.updates < 100:
            parser.error("Use at least 100 updates to exercise scaled curriculum and WSD")
        sanity(args)
    else:
        parser.error("Choose --stage microbatch, concurrency, or sanity")


if __name__ == "__main__":
    main()
