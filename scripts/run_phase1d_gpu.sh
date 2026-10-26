#!/bin/bash
# FINAL RULING 2026-10-26 §6.3 GPU queue (D-043): gold -> siblings G -> peek-state hidden -> peek-state scoring
# -> W1-J (only if the prompt is approved, see 25_w1_judge.py). Every step is resumable or cheap to redo.
# Run in GPU mode:  nohup scripts/run_phase1d_gpu.sh > logs/phase1d_gpu.log 2>&1 &
set -euo pipefail
cd /root/autodl-tmp/idea
R=/root/autodl-tmp/idea
log() { echo "[$(date -Is)] $*"; }
scripts/run.sh -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { log "no GPU; abort"; exit 1; }

log "START gold (bank index 6, n_rep 20, 300 points)"
scripts/run.sh 11_replay.py --engine async --bank 6 --points live --seeds 0-99 --subset_file $R/eval/gold_points.json \
    --n_rep 20 --out_root $R/data/replay_gold > logs/1d_gold.log 2>&1
log "END gold"

for k in 1 2 3 4 5 6 7; do
    log "START siblings k=$k"
    scripts/run.sh 10_rollout.py --seeds 0-99 --siblings $k-$k > logs/1d_siblings_k$k.log 2>&1
    log "END siblings k=$k"
done

log "START peek-state hidden (bf16)"
scripts/run.sh 24_peek_states.py --part hidden > logs/1d_peek_hidden.log 2>&1
log "END peek-state hidden"

log "START peek-state scoring (fp32 hfprefix)"
M0_HF_DTYPE=fp32 scripts/run.sh 24_peek_states.py --part score > logs/1d_peek_score.log 2>&1
log "END peek-state scoring"

if scripts/run.sh 25_w1_judge.py > logs/1d_w1_judge.log 2>&1; then
    log "W1-J done"
else
    log "W1-J not run (see logs/1d_w1_judge.log; needs logs/W1J_APPROVED)"
fi
log "PHASE 1D GPU QUEUE COMPLETE"
