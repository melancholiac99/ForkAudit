# Hindsight next three steps: run-first implementation

**Status:** code + 7 targeted CPU tests; **no new GPU data or H-LN training result**. Branch `research/hindsight-judges`; no modifications to frozen E1/E1c, E4 or `main`.

## Step 1. Do the mechanism test first (386 original P4 forks)

`exp3_action_diagnostic.py` takes the frozen P4 cohort, original Student written responses and original five candidate actions. It scores action continuations under **M / L / LN / F × Student current Think keep/drop**, with both frozen 7B and 32B. `M` is the exact decision-time ALFWorld prompt, so 7B-M is the student reference at each corresponding prefix. L and LN are **leakage-guarded**: the target turn's *response and executed-action label* are masked in the privileged copy (but its environment observation and final outcome remain). Consequently L vs LN isolates **other-turn Think**, and keep vs drop isolates **the current Student prefix**. These are *not* E1c's original unguarded L/LN.

The diagnostic measures (a) Teacher-minus-7B-Student **action-sequence log-prob and five-arm log-odds**, (b) whether the change scales with **independent q3456 action quality**, (c) expected independently replayed choice value, and (d) harmful factual corrections where the factual action is worse than the preregistered alternative. This is not full-vocabulary token KL or proof of beneficial RL gradients. Historical rollout token logprobs are not strictly needed: reference student action logprobs are recomputed from frozen 7B using the **same action prefix and tokenizer**. Missing/unparseable/no-Think points are logged and excluded, not silently synthesized.

```bash
python -m unittest discover -s hindsight_judges -p test_exp3.py -v
python hindsight_judges/exp3_action_diagnostic.py prepare

# Fast exploratory check first: first 64 point IDs, not a publication estimate.
python hindsight_judges/exp3_action_diagnostic.py score --name 7b --limit 64 \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct
python hindsight_judges/exp3_action_diagnostic.py score --name 32b --limit 64 \
  --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct
python hindsight_judges/exp3_action_diagnostic.py analyze --limit 64 \
  --arms /path/to/exact/original/exploratory_arms.parquet

# Full evidence (same commands with no --limit):
python hindsight_judges/exp3_action_diagnostic.py score --name 7b --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct
python hindsight_judges/exp3_action_diagnostic.py score --name 32b --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct
python hindsight_judges/exp3_action_diagnostic.py analyze --arms /path/to/exact/original/exploratory_arms.parquet
```

Report: `hindsight_judges/results/action_diagnostic/analysis_first64.json` or `analysis_full.json`. The frozen P4 / E1c preparation files must be present; the exact original deep-discovery arms parquet is hash-checked against P4 E2. Recompute M and privileged conditions with the SAME model precision within each model; do not subtract 7B/32B logprobs without acknowledging they are different teachers. The 32B scorer uses the verified E1c FP16 MATH chunked backend; 7B uses the **E1-verified full-FP32 recovery** of that frozen 7B scorer (both gate against direct candidate scoring).

**Decision rule:** If current Think makes the Teacher systematically *less* corrective on bad factual actions and less aligned with independent Q, the blind-credit method is directly motivated. A change in logprob alone is insufficient. If the effect is absent, do not claim the mechanism or treat blind credit as established.

## Step 2. H-LN: reuse E4 DPO and its held-out ALFWorld evaluation

`exp3_hln_dpo.py` reads original E4 judge requests and point records. It re-scores both **L** and **LN** using the same 32B FP16 scorer, keeps all E4 actions/observations/candidates fixed, and constructs E4-format DPO training pairs. The two natural arms **H_re / H_LN** have the same number of pairs and the same original E4 hyperparameters/seed. An extra common-point pair set is saved for later checking, but does not need to be trained now. The original tracked E4 H arm is untouched.

The E4 `data/points/` and `data/judge/requests.jsonl.gz` are not individually Git-tracked. They ARE available in the published E4 nonweight archive or on the original E4 server. To restore from the archive:

```bash
cd exploratory_E4/results_archive
cat e4_nonweight_results.tar.gz.part* > /tmp/e4_nonweight_results.tar.gz
mkdir -p /tmp/e4_results
tar -xzf /tmp/e4_nonweight_results.tar.gz -C /tmp/e4_results
cd ../..
```

On original E4 server, omit `--e4-data /tmp/e4_results/data` to use `exploratory_E4/data` directly. To use restored data:

```bash
python hindsight_judges/exp3_hln_dpo.py prepare --e4-data /tmp/e4_results/data

# Optional limited GPU pilot (separate from full score output):
python hindsight_judges/exp3_hln_dpo.py score --limit 100 \
  --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct

# Full score, pair construction, then matched-budget original E4 DPO training:
python hindsight_judges/exp3_hln_dpo.py score \
  --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct
python hindsight_judges/exp3_hln_dpo.py pairs --e4-data /tmp/e4_results/data
for seed in 0 1 2; do
  for arm in H_re H_LN; do
    python hindsight_judges/exp3_hln_dpo.py train --e4-data /tmp/e4_results/data \
      --arm "$arm" --seed "$seed" --lr 2e-5
  done
done

# Use the same E4 ALFWorld evaluation harness and paired game-bootstrap:
export M0_MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
python hindsight_judges/exp3_hln_dpo.py evaluate --e4-data /tmp/e4_results/data --split valid_unseen
python hindsight_judges/exp3_hln_dpo.py analyze  --e4-data /tmp/e4_results/data --split valid_unseen
```

Main result: `hindsight_judges/results/hln_dpo/hln_dpo_result.json`, with H_LN minus H_re paired 95% CI. A larger natural-pair advantage is not by itself a proof of OPD mechanism; this is **DPO**, so it serves as the fastest training bridge. H_re is re-scored on the same FP16 execution as LN; do not compare E4's original BF16 logits directly against these.

## Step 3. Smallest viable rationale-blind credit module

`rationale_blind_credit.py` implements the proposed **bounded positive reward-advantage weight**. Compute teacher evidence effect from the *same no-Think prefix*: `delta = log P_teacher(action | M+F) - log P_teacher(action | M)`, where both are action-continuation scores with M/F as the two privilege conditions. (For code, these correspond to `F/drop` and `M/drop`; use a single teacher model for the subtraction.) Then:

`w = clip(exp(alpha * sign(A_reward) * delta), 0.5, 2.0)`

The PPO/GRPO reward advantage **A_reward** is computed from actual environment rewards; `A_reward * w` always retains its sign. The `weighted_ppo_policy_loss` helper applies this sign-preserving weight to **all Student response tokens**, including Think. This is a minimal **method candidate and implementation primitive**, **NOT** a completed OPD/RL experiment or an algorithm with established efficacy. No full GRPO agent pipeline is claimed in these three steps.

```bash
python hindsight_judges/rationale_blind_credit.py selftest
```

## Read results without overinterpreting them

If 7B/32B bad-action correction improves with Think removal **and** the H-LN DPO students improve on held-out games, this supports moving to reward-based Rationale-Blind Agent training. If one or both fail, revise the mechanism before paying for the final RL run. Prior E1c L→LN improvement was a **Judge** result, not an OPD training gain. Existing E4 A versus H is relevant but is **not** a Think-only causal comparison.
