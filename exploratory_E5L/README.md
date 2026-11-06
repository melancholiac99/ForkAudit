# E5-L: zero-replay-credit learning value in HotpotQA Search Agent

**Implementation only — not a completed experiment or frozen preregistration.** New folder does not change the E5-S/C/K preregistration or E4(D) results.

Design and limitations: [DESIGN_DRAFT.md](DESIGN_DRAFT.md). Existing environment/rollout machinery: [`../exploratory_E5`](../exploratory_E5). Reused DPO implementation: [`../exploratory_E4/e4_06_train.py`](../exploratory_E4/e4_06_train.py).

## Steps

All commands run from repo root. Dependencies and model/retriever setup are the same as E5 (`exploratory_E5/README.md`). `HOTPOT_TRAIN` must be **HotpotQA train**, not E5-S's dev question list; `HOTPOT_DEV` must be **HotpotQA dev distractor**. Do not use dev for DPO examples. `WIKI18` is wiki-18.jsonl. `RETRIEVER_URL`, `SEARCH_CACHE`, and `M0_MODEL` follow E5.

```bash
python -m unittest discover -s exploratory_E5L -p 'test_e5l.py' -v

# 0. Freeze disjoint train/pilot/eval tasks before collecting any training result.
python exploratory_E5L/e5l.py freeze \
  --hotpot-train "$HOTPOT_TRAIN" --hotpot-dev "$HOTPOT_DEV" --wiki18 "$WIKI18" \
  --exclude-eval exploratory_E5/tasks_search.json  # omit if E5-S tasks not frozen

# 1. Independent pilot; shares the E5-S code and deterministic retrieval cache.
for stage in rollout; do
  python exploratory_E5/e5_run.py "$stage" --env search \
    --tasks exploratory_E5L/tasks_pilot.json --out_root exploratory_E5L/data/pilot_measure
done
python exploratory_E5/e5_run.py bank --env search --b 1 \
  --tasks exploratory_E5L/tasks_pilot.json --out_root exploratory_E5L/data/pilot_measure
python exploratory_E5/e5_run.py peek --env search \
  --tasks exploratory_E5L/tasks_pilot.json --out_root exploratory_E5L/data/pilot_measure
python exploratory_E5L/e5l.py gate --pilot

# Stop here. Review pilot gate and DESIGN_DRAFT; freeze approved thresholds before formal GPU results.

# 2. Full training questions, after approval.
python exploratory_E5/e5_run.py rollout --env search \
  --tasks exploratory_E5L/tasks_train.json --out_root exploratory_E5L/data/train_measure
python exploratory_E5/e5_run.py bank --env search --b 1 \
  --tasks exploratory_E5L/tasks_train.json --out_root exploratory_E5L/data/train_measure
python exploratory_E5/e5_run.py peek --env search \
  --tasks exploratory_E5L/tasks_train.json --out_root exploratory_E5L/data/train_measure
python exploratory_E5L/e5l.py gate
python exploratory_E5L/e5l.py pairs
python exploratory_E5L/e5l.py preflight --model "$M0_MODEL"

# 3. GPU DPO, every group x 3 independent training seeds.
for arm in Szero Sgap Nzero Ngap Smixed Nmixed; do
  for seed in 0 1 2; do
    python exploratory_E5L/e5l.py train --arm "$arm" --seed "$seed" \
      --model "$M0_MODEL" --approved
  done
done

# 4. Held-out dev evaluation, 3 rollouts/task. Can resume per tag.
python exploratory_E5L/e5l.py eval --tag base --model "$M0_MODEL" --approved
for arm in Szero Sgap Nzero Ngap Smixed Nmixed; do
  for seed in 0 1 2; do
    python exploratory_E5L/e5l.py eval --tag "${arm}_s${seed}" \
      --model "$M0_MODEL" --approved
  done
done
python exploratory_E5L/e5l.py analyze
```

Note: if E5-S frozen tasks do **not** exist, omit `--exclude-eval` entirely; the script automatically excludes any E5-S task lists found in `exploratory_E5/`. With E5-S tasks present, extra explicit exclusions are redundant. Do not edit or overwrite those original task lists.

**Hard stops:** failed E5 refeed consistency, too few matched labelled pairs (`N<256`), missing bank/peek, malformed raw draft responses, and overly long tokenizer input. A 4-action search budget may yield very few informative states or labels. If a gate fails, **report it and revise the unrun design explicitly**, not by quietly reclassifying outcomes or repeating examples.

### Data outputs (gitignored by parent `.gitignore` may not cover the new folder)

`exploratory_E5L/tasks_*.json` are frozen task specifications containing gold answers; `exploratory_E5L/data/` holds bulky replay inputs, adapter checkpoints, per-episode outcomes and manifests. These should stay **out of Git**, with only hashes, schema and aggregated reports committed when appropriate. `.gitignore` in this folder blocks them by default. The driver leaves `E5`'s existing state and frozen predictions unchanged.

### Execution validation

The Python CPU unit tests cover the new labels, shallow credit classes, paired sample draw, source consistency checks and HotpotQA train/dev separation. **They do not prove equivalence to the production retriever, run a vLLM engine, or establish the paper's hypothesis**; server preflight, pilot and formal evaluation are still required.
