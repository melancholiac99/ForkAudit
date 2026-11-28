#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH" CUDA_HOME="$root/cuda-12.8" HF_HOME="$root/hf-cache" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DIR="$root/logs" VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$root/triton-cache" TORCH_EXTENSIONS_DIR="$root/torch-extensions"
exec 9>"$root/run/cond-kl/forcing/all-gpus.lock"; flock -x 9
python start.py
