#!/bin/bash
# Phase 1C GPU chain (CLAUDE.md §7 Phase 1C, RULING 2026-10-23 §7.8). Every step is resumable (finished points are
# skipped), so after an interruption just re-run this script.
# Banks 7-10 (--bank 6..9) are the pre-registered expansion triggered by 1C.0 (D-023); they run last so the
# researcher can still override D-023 before they start.
set -euo pipefail
cd /root/autodl-tmp/idea
R=scripts/run.sh
step() { echo "[$(date -Is)] START $*"; "$@"; echo "[$(date -Is)] END $*"; }
step $R 11_replay.py --engine async --bank 1 --points D --seeds 0-49   > logs/1c_bank2_s000-049.log 2>&1
[ -f data/rollout/s099.json ] || step $R 10_rollout.py --seeds 50-99 > logs/1c_rollout_s050-099.log 2>&1
step $R 11_replay.py --engine async --seeds 50-99                      > logs/1c_bank1_s050-099.log 2>&1
step $R 11_replay.py --engine async --bank 1 --points D --seeds 50-99  > logs/1c_bank2_s050-099.log 2>&1
for b in 2 3 4 5; do
  step $R 11_replay.py --engine async --bank $b --points live --seeds 0-99 > logs/1c_bank$((b+1))_live.log 2>&1
done
if [ -f logs/STOP_EXPANSION ]; then echo "STOP_EXPANSION present: banks 7-10 skipped"; exit 0; fi
for b in 6 7 8 9; do
  step $R 11_replay.py --engine async --bank $b --points live --seeds 0-99 > logs/1c_bank$((b+1))_live.log 2>&1
done
echo "[$(date -Is)] PHASE 1C CHAIN COMPLETE"
