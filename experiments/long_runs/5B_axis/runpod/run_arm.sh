#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 temporal|depth|hybrid" >&2
    exit 2
fi

arm="$1"
case "$arm" in
    temporal) config="experiments/long_runs/5B_axis/temporal_5B/config.py" ;;
    depth) config="experiments/long_runs/5B_axis/depth_5B/config.py" ;;
    hybrid) config="experiments/long_runs/5B_axis/hybrid_5B/config.py" ;;
    *)
        echo "unknown arm: $arm" >&2
        exit 2
        ;;
esac

root="$(git rev-parse --show-toplevel)"
cd "$root"

dataset="data/chess_8M_v1"
for required in "$dataset/train.bin" "$dataset/val.bin" "$dataset/meta.pkl" "$dataset/manifest.json"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required dataset file: $required" >&2
        exit 1
    fi
done

output="experiments/long_runs/5B_axis/${arm}_5B/results"
batch_size="${BATCH_SIZE:-5}"
if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || (( 100 % batch_size != 0 )); then
    echo "BATCH_SIZE must be a positive divisor of effective batch 100: $batch_size" >&2
    exit 2
fi
accumulation=$((100 / batch_size))
if [[ -f "$output/ckpt.pt" ]]; then
    init_from="resume"
    saved_batch="$(uv run python - "$output/ckpt.pt" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
config = checkpoint['config']
print(config['batch_size'], config['gradient_accumulation_steps'])
PY
)"
    read -r saved_batch_size saved_accumulation <<< "$saved_batch"
    if [[ "$saved_batch_size" != "$batch_size" || "$saved_accumulation" != "$accumulation" ]]; then
        echo "existing checkpoint uses batch_size=$saved_batch_size, accumulation=$saved_accumulation; " \
             "requested batch_size=$batch_size, accumulation=$accumulation" >&2
        echo "choose a matching BATCH_SIZE or use a fresh isolated output directory" >&2
        exit 1
    fi
else
    init_from="scratch"
fi

# This receipt is intentionally separate from the frozen Verda protocol.  The
# old protocol records A6000 hardware and its runner rejects a 3090 before
# training; the resolved arm config, dataset, source commit, and CUDA runtime
# are still recorded here before the direct train.py invocation.
uv run python - "$arm" "$config" "$output" "$init_from" "$batch_size" "$accumulation" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from importlib import import_module

study_module = import_module('experiments.long_runs.5B_axis.study')
ACTUAL_CHARACTERS = study_module.ACTUAL_CHARACTERS
UPDATES = study_module.UPDATES
run_config = study_module.run_config

arm, config_path, output, init_from, batch_size, accumulation = sys.argv[1:]
root = Path.cwd()
manifest = root / 'data/chess_8M_v1/manifest.json'
protocol = root / 'experiments/long_runs/5B_axis/results/protocol.json'

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return f'<unavailable: {error}>'

config = run_config(arm)
config['batch_size'] = int(batch_size)
config['gradient_accumulation_steps'] = int(accumulation)
helper = root / 'experiments/long_runs/5B_axis/runpod/run_arm.sh'
payload = dict(
    schema_version=1,
    study='5B_axis',
    arm=arm,
    config_path=config_path,
    config=config,
    expected_optimizer_updates=UPDATES,
    expected_characters=ACTUAL_CHARACTERS,
    init_from=init_from,
    effective_batch_size=config['batch_size'] * config['gradient_accumulation_steps'],
    helper_sha256=sha256(helper),
    source=dict(
        branch=command('git', 'branch', '--show-current'),
        commit=command('git', 'rev-parse', 'HEAD'),
        status=command('git', 'status', '--porcelain=v1', '--untracked-files=all'),
    ),
    dataset=dict(path=str(manifest), manifest_sha256=sha256(manifest)),
    frozen_protocol_sha256=sha256(protocol) if protocol.is_file() else None,
    runtime=dict(
        python=sys.version,
        platform=platform.platform(),
        torch=command('uv', 'run', 'python', '-c', 'import torch; print(torch.__version__)'),
        cuda=command('nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'),
    ),
    execution_note='Direct train.py wrapper for RTX 3090; frozen 5B_axis run.py A6000 preflight is not used.',
)
path = root / output / 'runpod_execution.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
print(path)
PY

echo "Starting $arm with $init_from, batch_size=$batch_size, accumulation=$accumulation: $config"
exec uv run python train.py "$config" \
    --init_from="$init_from" \
    --batch_size="$batch_size" \
    --gradient_accumulation_steps="$accumulation" \
    --device=cuda \
    --dtype=bfloat16 \
    --compile=False
