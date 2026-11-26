#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
r=/mnt/data/ForkAudit/math-opsd
export PATH="$r/conda-env/bin:$r/cuda-12.8/bin:$PATH" CUDA_HOME="$r/cuda-12.8" HF_HOME="$r/hf-cache" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DIR="$r/logs" VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$r/triton-cache" TORCH_EXTENSIONS_DIR="$r/torch-extensions"
for arm in R-lambda0 R-lambda05; do
    until [[ -s "$r/run/math500-long-seed1001-n2/nonthinking-${arm}@100/receipt.json" ]]; do sleep 2; done
done
# Yield the four cards at the @100 evaluation boundary before the optional follow-up.
if [[ ! -s ../lambda_family/REPORT.md && ! -s ../lambda_family/receipts-4096-R-lambda0@100.json && ! -s ../lambda_family/receipts-4096-R-lambda05@100.json ]]; then
    systemctl stop forkaudit-interior-priority
fi
exec 9>"$r/run/cond-kl/forcing/all-gpus.lock"; flock -x 9
python lambda_aime24.py
if [[ ! -s ../lambda_family/REPORT.md ]]; then
    cd ../lambda_family
    exec python evaluate.py
fi
