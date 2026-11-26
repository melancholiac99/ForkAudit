#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
r=/mnt/data/ForkAudit/math-opsd
export PATH="$r/conda-env/bin:$r/cuda-12.8/bin:$PATH" CUDA_HOME="$r/cuda-12.8" HF_HOME="$r/hf-cache" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DIR="$r/logs" VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$r/triton-cache" TORCH_EXTENSIONS_DIR="$r/torch-extensions"
while [[ ! -s ../forcing/budget32/REPORT.md ]]; do sleep 5; done
exec 9>"$r/run/cond-kl/forcing/all-gpus.lock"; flock -x 9
lanes(){
 local pids=() rc=0
 for g in {0..3}; do CUDA_VISIBLE_DEVICES=$g python "$1" "$g" & pids+=($!); done
 for p in "${pids[@]}";do wait "$p" || rc=1; done
 test "$rc" = 0
}
python test_positions.py
lanes measure_clip.py
python finish_v2.py & summary=$!
python -c 'from finish import verify_secondary; verify_secondary()' & secondary=$!
for arm in R-clip-boundary R-clip-interior; do python train.py "$arm"; done
lanes fixed_eval.py
python evaluate_cells.py
python effects.py
wait "$summary"
wait "$secondary"
