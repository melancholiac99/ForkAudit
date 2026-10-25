#!/bin/bash
# Phase 2 work that is identical under CLAUDE.md (2026-10-23) and RULING 2026-10-25 (D-034):
# raw candidate/chosen scoring (arms A, B inputs, H1) and two-position hidden states (arm C, V).
# No feature computation, no label access.
set -euo pipefail
cd /root/autodl-tmp/idea
R=scripts/run.sh
step() { echo "[$(date -Is)] START $*"; "$@"; echo "[$(date -Is)] END $*"; }
[ -f data/scores.parquet ] || step $R 20_score_candidates.py --backend vllm --seeds 0-99 --chunk_points 100 \
    --out_dir /root/autodl-tmp/idea/data > logs/p2_score_vllm.log 2>&1
[ -f hidden/index.parquet ] || step $R 23_hidden_export.py --seeds 0-99 --out_dir /root/autodl-tmp/idea/hidden \
    > logs/p2_hidden.log 2>&1
echo "[$(date -Is)] SHARED PHASE 2 COMPLETE"
