#!/bin/bash
# Phase 1D GPU queue, part 2 (D-047): replaces the post-gold steps of run_phase1d_gpu.sh.
# Siblings run as ONE 700-trajectory batch (--siblings 1-7) instead of 7 runs of 100: per-request seeds depend only on
# (task seed, turn, k), so the design is unchanged; the larger lockstep batch keeps the GPU fed.
# Usage: nohup scripts/run_phase1d_gpu2.sh <gold_replay_pid> > logs/phase1d_gpu2.log 2>&1 &
set -euo pipefail
cd /root/autodl-tmp/idea
log() { echo "[$(date -Is)] $*"; }
GOLD_PID=${1:?gold replay pid}
log "waiting for gold replay pid $GOLD_PID"
while kill -0 "$GOLD_PID" 2>/dev/null; do sleep 60; done
grep -q "^DONE 300 points" logs/1d_gold.log || { log "gold did not finish cleanly; abort"; tail -5 logs/1d_gold.log; exit 1; }
log "END gold: $(grep '^DONE' logs/1d_gold.log)"

log "START siblings k=1-7 (one batch, 700 trajectories)"
scripts/run.sh 10_rollout.py --seeds 0-99 --siblings 1-7 > logs/1d_siblings.log 2>&1
log "END siblings: $(tail -1 logs/1d_siblings.log)"

log "START peek-state hidden (bf16)"
scripts/run.sh 24_peek_states.py --part hidden > logs/1d_peek_hidden.log 2>&1
log "END peek-state hidden"

log "START peek-state scoring (fp32 hfprefix)"
M0_HF_DTYPE=fp32 scripts/run.sh 24_peek_states.py --part score > logs/1d_peek_score.log 2>&1
log "END peek-state scoring"

log "START W1-J"
scripts/run.sh 25_w1_judge.py > logs/1d_w1_judge.log 2>&1
log "END W1-J: $(tail -1 logs/1d_w1_judge.log)"
log "PHASE 1D GPU QUEUE COMPLETE"
