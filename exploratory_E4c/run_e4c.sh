#!/usr/bin/env bash
# E4c queue (PREREG_E4c §5). Run from exploratory_E4c/ in the m0 environment, after the pre-registration commit.
# Resumable: e4c_eval.py skips episode files that already exist.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/logs

python test_e4c.py | tee data/logs/cpu_tests.log

# Gate S0 (plumbing only, random policy, 2 games): no success rates are read.
rm -rf /tmp/e4c_dry
python e4c_eval.py --split valid_unseen --conds R X --dry --limit 2 --out /tmp/e4c_dry > data/logs/s0_dry.log 2>&1
python s0_checks.py dry /tmp/e4c_dry | tee data/logs/s0_dry_check.log
# Gate S0 (task completion conditions; reported, not a stop): does any PDDL goal mention examine / inventory / look?
python s0_checks.py goals | tee data/logs/s0_goals.log

python e4c_eval.py --split valid_unseen --conds base P R PR X > data/logs/eval_unseen.log 2>&1
python e4c_analyze.py --splits valid_unseen | tee data/logs/primary.log       # the reading is final here

python e4c_eval.py --split valid_seen --conds base P R PR X > data/logs/eval_seen.log 2>&1
python e4c_analyze.py | tee data/logs/final.log
