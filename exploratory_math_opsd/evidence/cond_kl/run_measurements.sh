#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH" CUDA_HOME="$root/cuda-12.8" HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$root/triton-cache" TORCH_EXTENSIONS_DIR="$root/torch-extensions"
publish(){
 local stage=$1
 git add -- REPORT.md results.json
 if ! git diff --cached --quiet; then git -c user.name=ForkAudit -c user.email=forkaudit@localhost commit -m "Report conditional KL $stage"; fi
 local ref
 ref=$(git rev-parse HEAD)
 if [[ -n "${COND_PUBLISH_SCRIPT:-}" ]]; then
     (cd /mnt/data/ForkAudit/s26-code && source env/verl_agent.sh && python "$COND_PUBLISH_SCRIPT" "$ref") >"$root/run/cond-kl/published-$stage.log"
 fi
}
for arm in R-noclip R-cond; do
 while [[ ! -s "$root/run/$arm/complete.json" ]]; do sleep 20; done
done
python report.py training
publish training
for stage in fixed generate own; do
 for arm in R-noclip R-cond; do
    pids=()
    for gpu in {0..3}; do CUDA_VISIBLE_DEVICES=$gpu python measure.py "$stage" "$arm" "$gpu" > "$root/logs/cond-$stage-$arm-$gpu.log" 2>&1 & pids+=("$!"); done
    rc=0; for pid in "${pids[@]}"; do wait "$pid" || rc=1; done; test "$rc" = 0
 done
 if [[ $stage != generate ]]; then
    python report.py "$stage"
    if [[ $stage == fixed ]]; then python compare_fixed.py; fi
    publish "$stage"
 fi
done
for bench in math500 aime24 aime25 hmmt25; do
 python evaluate.py "$bench"
 python report.py "$bench"
 if [[ $bench == hmmt25 ]]; then python report.py three_benchmarks; fi
 publish "$bench"
done
