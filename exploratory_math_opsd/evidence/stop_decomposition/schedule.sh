#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
metrics=/mnt/data2/ForkAudit/math-opsd/run/stop-decomposition-v1
until test "$(find "$metrics" -maxdepth 1 -name '[0-9]*.npz' | wc -l)" -ge 100; do
    systemctl is-active --quiet forkaudit-stop-pilot || exit 1
    sleep 1
done
start=$(cat "$metrics/pilot.start")
last=$(find "$metrics" -maxdepth 1 -name '[0-9]*.npz' -printf '%T@\n' | sort -n | tail -1 | cut -d. -f1)
elapsed=$((last-start))
n=1500
if (( elapsed * 15 > 14400 )); then n=500; fi
printf '{"pilot_questions":100,"seconds":%d,"estimated_full_seconds":%d,"selected_questions":%d}\n' "$elapsed" "$((elapsed*15))" "$n" > "$metrics/timing.json"
bash run.sh "$n"
