#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
pids=()
for lane in {0..3}; do CUDA_VISIBLE_DEVICES="$lane" "$root/conda-env/bin/python" probe.py "$lane" "$1" >"$root/logs/stop-probe-$1-$lane.log" 2>&1 & pids+=("$!"); done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit "$rc"
