#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
py="$root/conda-env/bin/python"
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH" CUDA_HOME="$root/cuda-12.8" HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$root/triton-cache" TORCH_EXTENSIONS_DIR="$root/torch-extensions"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
for stage in calibrate generate score; do
    if [[ "$stage" == calibrate && -s "$root/run/projector-calibration-rollins-v1/calibration.json" ]]; then continue; fi
    pids=()
    for i in {0..3}; do CUDA_VISIBLE_DEVICES="$i" "$py" measure.py "$stage" "$i" >"$root/logs/cal-roll-$stage-$i.log" 2>&1 & pids+=("$!"); done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; test "$rc" = 0
done
CUDA_VISIBLE_DEVICES='' "$py" report.py
