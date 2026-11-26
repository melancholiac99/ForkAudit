#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH" CUDA_HOME="$root/cuda-12.8" HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn TRITON_CACHE_DIR="$root/triton-cache" TORCH_EXTENSIONS_DIR="$root/torch-extensions"
raw="$root/run/cond-kl/forcing/reflection"
exec 9>"$raw/../all-gpus.lock"
flock -x 9
test -s ../secondary.json
lanes(){
    local file=$1
    shift
    local jobs=() status=0
    for gpu in {0..3}; do CUDA_VISIBLE_DEVICES=$gpu python "$file.py" "$gpu" "$@" >>"$raw/$file-$gpu.log" 2>&1 & jobs+=("$!"); done
    for job in "${jobs[@]}"; do wait "$job" || status=1; done
    test "$status" = 0
}
publish(){
    git add -- REPORT.md "$1.json"
    git -c user.name=ForkAudit -c user.email=forkaudit@localhost commit -m "Report reflection probe $1"
    local ref
    ref=$(git rev-parse HEAD)
    (cd /mnt/data/ForkAudit/s26-code && source env/verl_agent.sh && python .runtime/runs/shop_gap26_20261123/first_command_replay/publish_tree.py "$ref") > "$raw/published-$1.log"
}
if [[ ! -s targeting.json ]]; then lanes score; python report.py targeting; publish targeting; fi
if [[ ! -s pilot.json ]]; then
    start=$(date +%s)
    lanes fork pilot
    python - "$start" <<'PY'
import time
from settings import *
assert len(list(RAW.glob('fork-*.json')))==32
save(HERE/'pilot.json',dict(sites=32,continuations=256,seconds=time.time()-int(sys.argv[1])))
PY
    python report.py render
    publish pilot
fi
lanes fork full
python report.py forks
publish forks
