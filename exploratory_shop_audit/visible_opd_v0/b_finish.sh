#!/usr/bin/env bash
set -eu
source env/verl_agent.sh
export OMP_NUM_THREADS=2
r=.runtime/runs/shop_visible_opd_content_b_20261120
until test -f "$r/score_finished.json"; do
    if ! systemctl is-active --quiet forkaudit-content-b-data; then
        python - <<'PY'
from exploratory_shop_audit.visible_opd_v0.b_collect import RUN
from exploratory_shop_audit.grpo_forced_audit.common import read_rows,save
import json
cost=read_rows(RUN/'b_cost.jsonl')
assert any(x['stage']=='three_view_scoring' and not x['pilot'] for x in cost)
keys=set()
for p in RUN.glob('tape_*.jsonl'):
    for l in p.open():
        v=json.loads(l);keys.add((v['goal'],v['step']))
assert len({k[0] for k in keys})==512 and len(keys)==len(list((RUN/'features').glob('*.pt')))
save(RUN/'score_finished.json',dict(tasks=512,steps=len(keys)))
PY
    else sleep 5; fi
done
pids=()
for k in 0 1 2 3; do
    if ! test -f "$r/fit_$k.json"; then CUDA_VISIBLE_DEVICES=$k python -m exploratory_shop_audit.visible_opd_v0.b_fit "$k" >"$r/fit_$k.log" 2>&1 & pids+=($!); fi
done
bad=0
for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
test "$bad" = 0
CUDA_VISIBLE_DEVICES=0 python -m exploratory_shop_audit.visible_opd_v0.b_ridge > "$r/ridge.log" 2>&1 &
ridge=$!
OPENBLAS_NUM_THREADS=1 python -m exploratory_shop_audit.visible_opd_v0.b_metrics
wait "$ridge"
test -f "$r/equivalent_pool.json"
CUDA_VISIBLE_DEVICES=0 python -m exploratory_shop_audit.visible_opd_v0.b_bias
