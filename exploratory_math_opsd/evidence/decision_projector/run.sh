#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
py="$root/conda-env/bin/python"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
for stage in prepare fit calibrate measure; do
    pids=()
    for i in {0..3}; do CUDA_VISIBLE_DEVICES="$i" "$py" diagnostic.py "$stage" "$i" >"$root/logs/decision-projector-$stage-$i.log" 2>&1 & pids+=("$!"); done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; test "$rc" = 0
done
CUDA_VISIBLE_DEVICES='' "$py" report.py
