#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
r=/mnt/data/ForkAudit/math-opsd
export PATH="$r/conda-env/bin:$r/cuda-12.8/bin:$PATH" CUDA_HOME="$r/cuda-12.8" HF_HOME="$r/hf-cache" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_DIR="$r/logs" VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$r/triton-cache" TORCH_EXTENSIONS_DIR="$r/torch-extensions"
while [[ ! -s ../clip_positions/FACTORIAL.md ]]; do sleep 10; done
exec 9>"$r/run/cond-kl/forcing/all-gpus.lock"; flock -x 9
CUDA_VISIBLE_DEVICES='' python cpu_pairs.py & cpu=$!
python test_gradient.py
python ../train.py R-condA
pids=()
for g in {0..3}; do (
 export CUDA_VISIBLE_DEVICES=$g
 PYTHONPATH=.. python -c 'import sys,measure; measure.STEPS=(100,); measure.score("fixed","R-condA",int(sys.argv[1]))' "$g"
 python secondary_conda.py "$g"
) & pids+=($!);done
rc=0;for p in "${pids[@]}"; do wait "$p" || rc=1; done;test "$rc" = 0
python eval_cells.py
CUDA_VISIBLE_DEVICES='' python summarize.py
wait "$cpu"
