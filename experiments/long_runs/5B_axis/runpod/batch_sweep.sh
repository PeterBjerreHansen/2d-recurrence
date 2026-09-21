#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

config="experiments/long_runs/5B_axis/hybrid_5B/config.py"
sweep_root="experiments/long_runs/5B_axis/hybrid_5B/results/batch_sweep"
steps="${BATCH_SWEEP_STEPS:-20}"
read -r -a batch_sizes <<< "${BATCH_SIZES:-5 10 20 25 50}"

if [[ ! -f data/chess_8M_v1/train.bin || ! -f data/chess_8M_v1/val.bin ]]; then
    echo "the complete pinned dataset is required before the batch sweep" >&2
    exit 1
fi
if [[ ! -f experiments/long_runs/5B_axis/results/panel.json ]]; then
    echo "the frozen evaluation panel is required before the batch sweep" >&2
    exit 1
fi

mkdir -p "$sweep_root"

for batch_size in "${batch_sizes[@]}"; do
    if (( 100 % batch_size != 0 )); then
        echo "batch size must divide effective batch 100: $batch_size" >&2
        exit 2
    fi
    accumulation=$((100 / batch_size))
    output="$sweep_root/batch_${batch_size}"
    if [[ -e "$output" ]]; then
        echo "refusing to reuse or overwrite existing sweep output: $output" >&2
        exit 1
    fi
    mkdir -p "$output"

    # The hybrid (3,3) graph is the conservative memory/throughput case.  The
    # short runs are a throughput probe only; they do not change the study's
    # training horizon or serve as model-quality evidence.
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total \
        --format=csv,noheader,nounits --loop-ms=500 > "$output/gpu_memory.csv" 2>&1 &
    monitor_pid=$!
    set +e
    uv run python train.py "$config" \
        --out_dir="$output" \
        --max_iters="$steps" \
        --batch_size="$batch_size" \
        --gradient_accumulation_steps="$accumulation" \
        --eval_interval=100000 \
        --eval_iters=1 \
        --log_interval=1 \
        --keep_checkpoints=False \
        --init_from=scratch
    status=$?
    set -e
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    if (( status != 0 )); then
        echo "batch-size probe failed for batch_size=$batch_size" >&2
        exit "$status"
    fi
done

uv run python - "$sweep_root" "${batch_sizes[@]}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
batch_sizes = [int(value) for value in sys.argv[2:]]
rows = []
for batch_size in batch_sizes:
    output = root / f'batch_{batch_size}'
    events = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    updates = [event for event in events if event.get('event') == 'train']
    seconds = [float(event['seconds']) for event in updates]
    rows.append(dict(
        batch_size=batch_size,
        gradient_accumulation_steps=100 // batch_size,
        updates=len(seconds),
        mean_update_seconds=statistics.mean(seconds),
        median_update_seconds=statistics.median(seconds),
        min_update_seconds=min(seconds),
        max_update_seconds=max(seconds),
        effective_batch_size=batch_size * (100 // batch_size),
        characters_per_second=statistics.mean(float(event['characters_per_second']) for event in updates),
        gpu_memory_log=str((output / 'gpu_memory.csv').resolve()),
    ))
(root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
with (root / 'summary.tsv').open('w') as stream:
    stream.write('batch_size\taccumulation\tmean_update_s\tmedian_update_s\tchars_per_s\n')
    for row in rows:
        stream.write(f"{row['batch_size']}\t{row['gradient_accumulation_steps']}\t"
                     f"{row['mean_update_seconds']:.6f}\t{row['median_update_seconds']:.6f}\t"
                     f"{row['characters_per_second']:.2f}\n")
print(json.dumps(rows, indent=2))
PY
