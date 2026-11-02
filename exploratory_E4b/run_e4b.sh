#!/usr/bin/env bash
# E4b queue (PREREG_E4b §6). Run from exploratory_E4b/ in the m0 environment, after the pre-registration commit.
# Order puts the primary 2x2 on valid_unseen first; seed-major so a crash leaves whole seeds. Resumable: finished
# adapters and evaluated episodes are skipped (e4_07 resumes per episode file).
set -euo pipefail
cd "$(dirname "$0")"
LR=5e-06
PRIMARY="Azero Agap Dzero Dgap"
mkdir -p data/logs

python test_e4b.py
python e4b_pairs.py | tee data/logs/pairs.log
python - <<'EOF'
import json
want = json.load(open("e4b_gate.json"))["expected_pairs_sha256"]
got = json.load(open("data/pairs/manifest.json"))["sha256"]
assert got == want, {"expected": want, "got": got}
print("pairs SHA256 match the pre-registered gate")
EOF

train() {  # arm seed
  [ -f "data/adapters/$1_s$2/adapter_model.safetensors" ] && return 0
  python e4b_run.py train --arm "$1" --seed "$2" --lr "$LR" > "data/logs/train_$1_s$2.log" 2>&1
}

# Stage 1: primary 2x2, valid_unseen (+ base re-run once)
for s in 0 1 2; do
  for a in $PRIMARY; do train "$a" "$s"; done
  base_flag=$([ "$s" = 0 ] && echo "" || echo "--no_base")
  python e4b_run.py eval --split valid_unseen $base_flag --adapters Azero_s$s Agap_s$s Dzero_s$s Dgap_s$s \
    > "data/logs/eval_unseen_s$s.log" 2>&1
done
python e4b_analyze.py --splits valid_unseen | tee data/logs/primary.log      # primary reading is final here

# Stage 2: Bfac-gap (secondary)
for s in 0 1 2; do train Bfacgap "$s"; done
python e4b_run.py eval --split valid_unseen --no_base --adapters Bfacgap_s0 Bfacgap_s1 Bfacgap_s2 \
  > data/logs/eval_unseen_bfac.log 2>&1

# Stage 3: valid_seen replication for every model
ALL=$(for a in $PRIMARY Bfacgap; do for s in 0 1 2; do printf "%s_s%s " "$a" "$s"; done; done)
python e4b_run.py eval --split valid_seen --adapters $ALL > data/logs/eval_seen.log 2>&1

python e4b_analyze.py | tee data/logs/final.log
