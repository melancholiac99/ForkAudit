#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
work=/mnt/data2/ForkAudit/contrast-v2
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH" CUDA_HOME="$root/cuda-12.8" HF_HOME="$root/hf-cache" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DIR="$root/logs" VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$root/triton-cache" TORCH_EXTENSIONS_DIR="$root/torch-extensions"
while [[ ! -s coverage.json ]]; do systemctl is-active --quiet forkaudit-repair-sample.service || { test -s coverage.json || exit 1; }; sleep 10; done
test -s padding_test.json
export OPSD_ARM=R-contrast-v2 OPSD_MODEL="$root/models/Qwen3-1.7B" OPSD_OUTPUT="$work/checkpoints" REPAIR_PAIRS="$work/pairs.json"
exec 9>"$root/run/cond-kl/forcing/all-gpus.lock"; flock -x 9
(cd "$work/source" && bash scripts/run_opsd_1b.sh --stop_after_updates 100 --seed 42 --data_seed 42 --save_steps 25 --save_only_model True --local_train_glob "$root/datasets/train/data/*.parquet") > "$work/training.log" 2>&1
python finish.py
