#!/bin/bash
set -eu
cd "$(dirname "$0")"
export PATH="/mnt/data/ForkAudit/math-opsd/conda-env/bin:/mnt/data/ForkAudit/math-opsd/cuda-12.8/bin:$PATH"
export CUDA_HOME=/mnt/data/ForkAudit/math-opsd/cuda-12.8 OMP_NUM_THREADS=2 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python setup_data.py prepare
pids=()
for lane in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$lane python setup_data.py sample "$lane" >"/mnt/data2/ForkAudit/contrast-v2/sample-$lane.log" 2>&1 & pids+=($!); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
test "$failed" = 0
python setup_data.py combine
