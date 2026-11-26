#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
raw="$root/run/cond-kl/forcing/reflection"
export PATH="$root/conda-env/bin:$PATH" HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export REFLECTION_HANDOFF="$root/run/evaluation/R-cond@100/aime24/1001.receipt.json"
test -s pilot.json
cards=(0 2 3)
pids=()
for lane in 0 1 2; do
  (
    gpu=${cards[lane]}
    test -s "$root/run/evaluation/R-cond@100/aime24/1001-lanes/lane$gpu.json"
    exec 6>"$raw/gpu-$gpu.lock"; flock -x 6
    [[ ! -s "$REFLECTION_HANDOFF" ]] || exit 0
    CUDA_VISIBLE_DEVICES=$gpu REFLECTION_LANES=3 python fork.py "$lane" full >"$raw/tail-$gpu.log" 2>&1
  ) &
  pids+=("$!")
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit "$rc"
