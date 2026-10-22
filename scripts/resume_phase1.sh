#!/bin/bash
# Resume the interrupted pilot, then finish the same frozen 50-trajectory replay.
set -euo pipefail
cd /root/autodl-tmp/idea
echo "Pilot resume started: $(date -Is)"
bash scripts/run.sh 11_replay.py --seeds 0-4 --workers 20 --max_concurrent 600 > logs/11_replay_pilot_resume.log 2>&1
echo "Pilot completed: $(date -Is)"
echo "Full replay resume started: $(date -Is)"
bash scripts/run.sh 11_replay.py --seeds 0-49 --workers 20 --max_concurrent 600 > logs/11_replay_full.log 2>&1
echo "Full replay completed: $(date -Is)"
