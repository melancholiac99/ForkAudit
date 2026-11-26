#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH" CUDA_HOME="$root/cuda-12.8" HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$root/triton-cache" TORCH_EXTENSIONS_DIR="$root/torch-extensions"
raw="$root/run/cond-kl/forcing"
exec 9>"$raw/all-gpus.lock"
flock -x 9
# Allow both already-running AIME24 cells to finish.
while [[ ! -s "$root/run/evaluation/R-cond@100/aime24/1001.receipt.json" ]]; do sleep 10; done
lanes(){
  local script=$1
  shift
  local pids=() rc=0
  for gpu in {0..3}; do CUDA_VISIBLE_DEVICES=$gpu python "$script.py" "$gpu" "$@" >> "$raw/$script-$gpu.log" 2>&1 & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  test "$rc" = 0
}
if [[ ! -s pilot.json ]]; then
  start=$(date +%s)
  for arm in B R-cond@100; do lanes budget pilot "$arm"; done
  python pilot.py "$start"
  cat pilot.json
  bash publish.sh pilot
fi
for arm in B R-cond@100; do lanes budget full "$arm"; done
python budget_report.py
bash publish.sh budget
lanes secondary
python secondary_report.py
bash publish.sh secondary
