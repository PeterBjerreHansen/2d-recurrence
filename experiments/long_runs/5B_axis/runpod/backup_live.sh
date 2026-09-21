#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
backup_root="${RUNPOD_BACKUP_ROOT:-$root/experiments/long_runs/5B_axis/runpod_live_backups}"
key="${RUNPOD_SSH_KEY:?Set RUNPOD_SSH_KEY to the SSH private-key path}"
spec_list="${RUNPOD_BACKUP_SPECS:?Set RUNPOD_BACKUP_SPECS to comma-separated arm|host|port entries}"
interval="${RUNPOD_BACKUP_INTERVAL_SECONDS:-600}"

IFS=',' read -r -a specs <<< "$spec_list"
if ((${#specs[@]} == 0)); then
    echo 'RUNPOD_BACKUP_SPECS must contain at least one arm|host|port entry' >&2
    exit 1
fi

mkdir -p "$backup_root"

while true; do
    for spec in "${specs[@]}"; do
        IFS='|' read -r arm ip port <<< "$spec"
        if [[ -z "$arm" || -z "$ip" || -z "$port" ]]; then
            echo "Invalid backup specification: $spec" >&2
            exit 1
        fi
        destination="$backup_root/$arm"
        mkdir -p "$destination/results" "$destination/runlogs"
        ssh_opts="-p $port -i $key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        rsync -az --partial --inplace --exclude='*.tmp' \
            -e "ssh $ssh_opts" \
            "root@$ip:/workspace/2d_recurrence/experiments/long_runs/5B_axis/${arm}_5B/results/" \
            "$destination/results/" 2>>"$backup_root/backup.log" || true
        rsync -az --partial --inplace \
            -e "ssh $ssh_opts" \
            "root@$ip:/workspace/2d_recurrence/runlogs/${arm}.log" \
            "$destination/runlogs/" 2>>"$backup_root/backup.log" || true
    done
    date -u +%FT%H:%M:%SZ > "$backup_root/last_attempt"
    sleep "$interval"
done
