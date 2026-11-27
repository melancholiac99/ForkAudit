#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
r=/mnt/data/ForkAudit/math-opsd
export PATH="$r/conda-env/bin:$r/cuda-12.8/bin:$PATH" CUDA_HOME="$r/cuda-12.8" HF_HOME="$r/hf-cache" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DIR="$r/logs" VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$r/triton-cache" TORCH_EXTENSIONS_DIR="$r/torch-extensions"
exec 9>"$r/run/cond-kl/forcing/all-gpus.lock"; flock -x 9
test -s ../cond_kl/lambda_family/r200_probe.json
lanes() {
    local -a jobs=()
    local rc=0 gpu pid
    for gpu in {0..3}; do CUDA_VISIBLE_DEVICES=$gpu python "$@" "$gpu" & jobs+=($!); done
    for pid in "${jobs[@]}"; do wait "$pid" || rc=1; done
    return "$rc"
}
python -m unittest discover -p 'test_contrast.py'
lanes contexts.py sample
python contexts.py combine
python train.py
for k in 100 200; do
    python evaluate_contrast.py aime24 "$k"
    python evaluate_contrast.py math500 "$k"
done
lanes evaluate_contrast.py probe
python evaluate_contrast.py collect
python evaluate_contrast.py gate
