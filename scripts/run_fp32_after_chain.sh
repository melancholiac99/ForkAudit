#!/bin/bash
# D-036: after the shared Phase 2 chain finishes, score everything again with HF fp32 + shared-prefix KV cache.
set -euo pipefail
cd /root/autodl-tmp/idea
while pgrep -f "^bash scripts/run_phase2_shared.sh" > /dev/null; do sleep 30; done
grep -q "SHARED PHASE 2 COMPLETE" logs/p2_shared_chain.log || { echo "chain did not complete; not starting fp32"; exit 1; }
echo "[$(date -Is)] START fp32 hfprefix"
M0_HF_DTYPE=fp32 scripts/run.sh 20_score_candidates.py --backend hfprefix --seeds 0-99 --chunk_points 100 \
    --out_dir /root/autodl-tmp/idea/data --tag _fp32 > logs/p2_score_fp32.log 2>&1
echo "[$(date -Is)] END fp32 hfprefix"
