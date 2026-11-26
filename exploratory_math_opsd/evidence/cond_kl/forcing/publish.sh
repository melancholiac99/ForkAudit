#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python render.py
git add -- REPORT.md results.json "$1.json"
if [[ $1 == secondary ]]; then git add -- reflection_ids.json; fi
git -c user.name=ForkAudit -c user.email=forkaudit@localhost commit -m "Report post-hoc math forcing $1"
ref=$(git rev-parse HEAD)
if ! (cd /mnt/data/ForkAudit/s26-code && source env/verl_agent.sh && python .runtime/runs/shop_gap26_20261123/first_command_replay/publish_tree.py "$ref") >"/mnt/data/ForkAudit/math-opsd/run/cond-kl/forcing/published-$1.log"; then
    echo "Remote publication pending for local commit $ref" >&2
fi
