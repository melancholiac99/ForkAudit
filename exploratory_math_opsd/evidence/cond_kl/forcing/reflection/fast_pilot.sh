#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
raw="$root/run/cond-kl/forcing/reflection"
export PATH="$root/conda-env/bin:$PATH" HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
finished="$root/run/evaluation/R-cond@100/aime24/1001.receipt.json"
while (( $(find "$raw" -name 'score-*.npz' |wc -l) < 1500 )); do
   [[ ! -s "$finished" ]] || exit 0
   sleep 2
done
publish(){
 git add -- "$1.json" REPORT.md
 git -c user.name=ForkAudit -c user.email=forkaudit@localhost commit -m "Report reflection probe $1"
 local ref
 ref=$(git rev-parse HEAD)
 (cd /mnt/data/ForkAudit/s26-code && source env/verl_agent.sh && python .runtime/runs/shop_gap26_20261123/first_command_replay/publish_tree.py "$ref") >"$raw/published-$1.log"
}
python report.py targeting
publish targeting
[[ ! -s "$finished" ]] || exit 0
exec 6>"$raw/gpu-0.lock"; flock -x 6
exec 7>"$raw/gpu-3.lock"; flock -x 7
[[ ! -s "$finished" ]] || exit 0
start=$(date +%s)
CUDA_VISIBLE_DEVICES=0 REFLECTION_LANES=2 python fork.py 0 pilot >"$raw/pilot-fast-0.log" 2>&1 & a=$!
CUDA_VISIBLE_DEVICES=3 REFLECTION_LANES=2 python fork.py 1 pilot >"$raw/pilot-fast-1.log" 2>&1 & b=$!
rc=0; wait "$a" || rc=1; wait "$b" || rc=1; test "$rc" = 0
flock -u 7; flock -u 6
python - "$start" <<'PY'
from settings import *
import time
assert len(list(RAW.glob('fork-*.json')))==32
save(HERE/'pilot.json',dict(sites=32,continuations=256,gpus=2,seconds=time.time()-int(sys.argv[1])))
PY
python report.py render
publish pilot
