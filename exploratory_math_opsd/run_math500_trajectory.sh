#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=/mnt/data/ForkAudit/math-opsd
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH"
export CUDA_HOME="$root/cuda-12.8" TORCH_EXTENSIONS_DIR="$root/torch-extensions" TRITON_CACHE_DIR="$root/triton-cache"
export HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR=/mnt/data2/ForkAudit/math-opsd/tmp CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1
while [[ ! -s "$here/evidence/rres/outside_probability.json" ]]; do
    systemctl is-active --quiet forkaudit-rres-extra || { test -s "$here/evidence/rres/outside_probability.json" || exit 1; }
    sleep 5
done
for bench in aime24 aime25 hmmt25; do test -s "$root/run/evaluation/Rres@100/$bench/1001.receipt.json"; done
exec python "$here/math500_trajectory.py"
