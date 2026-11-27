#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
r=/mnt/data/ForkAudit/math-opsd
export PATH="$r/conda-env/bin:$r/cuda-12.8/bin:$PATH" CUDA_HOME="$r/cuda-12.8" HF_HOME="$r/hf-cache" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DIR="$r/logs" VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$r/triton-cache" TORCH_EXTENSIONS_DIR="$r/torch-extensions"
exec 9>"$r/run/cond-kl/forcing/all-gpus.lock"; flock -x 9
python endpoint200.py aime
workers=()
for gpu in {0..3}; do CUDA_VISIBLE_DEVICES=$gpu python endpoint200.py probe "$gpu" & workers+=($!); done
rc=0
for pid in "${workers[@]}"; do wait "$pid" || rc=1; done
test "$rc" = 0
python endpoint200.py collect
python evaluate.py
