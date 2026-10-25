#!/bin/bash
# After the shared Phase 2 chain: a'_u sensitivity (D-037), then the fp32 hfprefix scoring pass (D-036).
set -euo pipefail
cd /root/autodl-tmp/idea
while pgrep -f "^bash scripts/run_phase2_shared.sh" > /dev/null; do sleep 30; done
grep -q "SHARED PHASE 2 COMPLETE" logs/p2_shared_chain.log || { echo "chain did not complete; stopping"; exit 1; }
step() { echo "[$(date -Is)] START $*"; "$@"; echo "[$(date -Is)] END $*"; }
step scripts/run.sh 11_replay.py --engine async --bank 10 --points D --seeds 0-99 \
    --points_file /root/autodl-tmp/idea/eval/au_points.json --n_rep 6 --no_factual \
    --out_root /root/autodl-tmp/idea/data/replay_au > logs/au_replay.log 2>&1
[ -f data/scores_fp32.parquet ] || M0_HF_DTYPE=fp32 step scripts/run.sh 20_score_candidates.py --backend hfprefix \
    --seeds 0-99 --chunk_points 100 --out_dir /root/autodl-tmp/idea/data --tag _fp32 > logs/p2_score_fp32.log 2>&1
echo "[$(date -Is)] AFTER-CHAIN COMPLETE"
