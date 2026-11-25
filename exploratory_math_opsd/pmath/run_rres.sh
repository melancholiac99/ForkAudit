#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
root=/mnt/data/ForkAudit/math-opsd
work="$root/run/P-projector/rres-cpu"
exec 9>"$work/pipeline.lock"
flock -n 9
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH"
export PYTHONPATH="$here"
export CUDA_HOME="$root/cuda-12.8" TORCH_CUDA_ARCH_LIST=8.0 TORCH_EXTENSIONS_DIR="$root/torch-extensions"
export TRITON_CACHE_DIR="$root/triton-cache" TMPDIR=/mnt/data2/ForkAudit/math-opsd/tmp
export WANDB_MODE=offline WANDB_DIR="$root/logs" HF_HOME="$root/hf-cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
while systemctl is-active --quiet forkaudit-rres-cpu; do sleep 5; done
test "$(find "$work" -maxdepth 1 -name 'w*-*.pt' | wc -l)" -eq 300
export CUDA_VISIBLE_DEVICES=''
python -m unittest test_rres -q
python -c 'from rres_summary import diagnostics; diagnostics()'
python install_rres.py
export CUDA_VISIBLE_DEVICES=0,1,2,3
python rres_run.py
