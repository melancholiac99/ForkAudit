#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
root=/mnt/data/ForkAudit/math-opsd
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH" CUDA_HOME="$root/cuda-12.8" HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$root/triton-cache" TORCH_EXTENSIONS_DIR="$root/torch-extensions"
raw="$root/run/cond-kl/forcing/budget32"
mkdir -p "$raw"
exec 9>"$raw/../all-gpus.lock"
flock -x 9
python test_forcing.py
for mode in pilot full; do
 for arm in B T0@100; do
    pids=(); for gpu in {0..3}; do CUDA_VISIBLE_DEVICES=$gpu python budget.py "$gpu" "$mode" "$arm" budget32 >>"$raw/$mode-$arm-$gpu.log" 2>&1 & pids+=("$!"); done
    rc=0;for p in "${pids[@]}";do wait "$p" || rc=1; done
    test "$rc" = 0
 done
done
python budget32/report.py
