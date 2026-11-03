# Hindsight Experiment 2 — synchronous, full-vocabulary on-policy distillation

**Status: code prepared and CPU-tested; no GPU train, numerical gate, or ALFWorld held-out outcome has been run in this development environment.** This is independent exploratory work on `research/hindsight-judges`; original E1/E1c/P4/E4 results remain read-only.

## Scientific question and limitations (read before running)

E1c observes that deleting the Student's visible historical `<think>` changes a 32B hindsight judge's executed-action preference and independent replay-referenced decision value. Whether that makes *actual OPD gradients* better must be tested by training and held-out success, not Teacher logprob alone.

**Critical distinction**: to avoid giving the Teacher the target response verbatim, `L_guarded` removes the **supervised turn's** reasoning from the duplicate future log, while `LN_guarded` removes **all turns'** reasoning from that future log. Both still receive the Student's already generated *causal prefix* during next-token scoring, and both see the same factual actions, observations, and outcome. Thus the controlled contrast `L_guarded` vs `LN_guarded` isolates **other-turn reasoning in privileged evidence**, *not* decision-t rationale that remains in the current-token prefix. If the localization study finds decision-t reasoning explains most of E1c's L→LN shift, this training comparison alone CANNOT validate a remedy for that dominant mechanism; design a credit-level or alternative teacher correction before claiming one. The nominal `L_guarded` vs `LN_guarded` comparison still provides a useful bounded diagnostic.

Additional hindsight logs reveal the factual **action** in both privileged conditions. That is a matched source of answer copying, not resolved by rationale removal. Neither a rising Student token logprob nor smaller Teacher KL proves improvement. Only held-out ALFWorld success supports an end-to-end learning claim.

## Conditions and optimization

| Arm | Teacher receives | Student receives | Training loss |
|---|---|---|---|
| `M_native` | **Exactly** the original decision-time prompt | Original prompt | Full-vocabulary token KL |
| `M` | Same wrapper as privilege arms but future unavailable | Original prompt | Same |
| `L_guarded` | Complete factual episode; `<think>` at supervised turn stripped in duplicate copy | Original prompt | Same |
| `LN_guarded` | Same episode, all visible historical `<think>` stripped in duplicate copy | Original prompt | Same |

Qwen2.5-32B Teacher is frozen under the *existing E1c precision configuration* (FP16 transformer / FP32 head, SDPA MATH, chunked prefill, frozen checkpoint hashes). Qwen2.5-7B Student trains LoRA (r16, alpha32, dropout 0.05, all projections) and generates full ALFWorld episodes with the **same trainable model instance**, rather than a stale external actor.

- Both models are resident in the same process on **different devices** (`cuda:0` Student, `cuda:1` Teacher). GPU memory budget does not change statistical design.
- At optimizer update `u`: select a fixed, outcome-blind batch of E4 TRAIN games → sample whole episodes using the current 7B Student and temperature 0.7 → select a fixed, outcome-blind subset of visited decisions → score Teacher distributions with that episode's full future → **exactly one** optimizer step averaged over selected decisions → sample brand-new episodes with the updated Student at `u+1`. Each episode is used for at most one optimizer step. There is no multi-epoch replay over stale Student responses.
- For every sampled response, next-token Teacher log probabilities are calculated at **the exact generated token IDs** conditioned on the *previous* Student token prefix; the target token never appears in its own prefix. We never rely on re-tokenizing generated response text. Teacher logprobs for the **full vocabulary** are passed to Student loss (no top-k normalization). Default `reverse` KL = `KL(Student || Teacher)`; `--kl-direction forward` is the specified directional ablation.
- All sampled `<think>` and `<action>` token positions participate in KL gradients. Per-class KL is recorded as descriptive mechanism analysis, **not** a correctness metric.
- A separate numerical gate compares full-vocabulary distributions from chunked vs one-pass Teacher scoring before any training step; total-variation tolerance is 0.005. The formal experiment also requires a reviewed Think localization analysis (386 points) as a scope gate.
- Each optimizer update has atomic adapter checkpoint, optimizer state, RNG state, full episode records, selected responses, hash-bound token diagnostics and SHA manifest. Resume refuses mismatched configurations, nonconsecutive checkpoints and partial output.

## CPU tests

```bash
python -m unittest discover -s hindsight_judges -p test_exp2_online.py -v
python hindsight_judges/exp2_online.py selftest
```

These test token KL direction/gradients, deterministic task schedule, matched L/LN evidence, anti-leakage at supervised turn, missing/unfinished `<think>` diagnostics, train/held-out disjointness, success-independent decision selection and game+training-seed paired bootstrap. They **do not** exercise GPU kernels, Hugging Face sampling, or ALFWorld without the real server.

## Mandatory preceding Think localization

Run the already-published localization code **first**:

```bash
python hindsight_judges/think_localization.py prepare
python hindsight_judges/think_localization.py score \
  --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct
python hindsight_judges/think_localization.py analyze \
  --arms /path/to/exact/original/exploratory_arms.parquet
```

Inspect `hindsight_judges/results/think_localization/analysis.json` for `DROP_T` and `KEEP_T` comparisons with frozen L/LN. The reviewer should decide whether other-turn rationale is a plausible training intervention. This is a scientific gate, not a programmable success threshold.

## Full online training (no computational shortcuts)

Every formal arm/seed starts from exactly the same frozen 7B checkpoint, with the same training-game and rollout-seed schedule and the same number of updates. Run each job with **two isolated GPUs**. For independent jobs use distinct output directories.

```bash
# First, actual two-GPU plumbing only: one fresh on-policy batch and one update.
CUDA_VISIBLE_DEVICES=0,1 python hindsight_judges/exp2_online.py train \
  --condition LN_guarded --seed 0 --n-updates 1 \
  --games-per-update 2 --max-decisions-per-update 4 \
  --output /path/to/hindsight_online_smoke/LN_guarded_s0

# Formal, after localization result has been inspected and its scope accepted.
for c in M_native M L_guarded LN_guarded; do
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0,1 python hindsight_judges/exp2_online.py train \
      --condition "$c" --seed "$seed" \
      --student /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
      --teacher /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
      --student-device cuda:0 --teacher-device cuda:1 \
      --n-updates 100 --games-per-update 8 --max-decisions-per-update 16 \
      --max-steps 50 --max-new-tokens 512 --max-student-len 6144 \
      --lr 2e-5 --kl-direction reverse \
      --localization-analysis hindsight_judges/results/think_localization/analysis.json \
      --scope-reviewed \
      --output "/path/to/hindsight_online/${c}_s${seed}"
  done
done
```

This commits to **100 distinct on-policy optimizer updates per run**, each based on complete episodes from the current Student, not a fixed 512-decision cached corpus. The four conditions, three Student seeds and evaluation budgets are independent. To study KL direction, rerun the *entire* matched comparison with `--kl-direction forward` under new output directories; do not compare unmatched settings or extend a prior run's `--n-updates` in place.

## Held-out evaluation (same HF policy sampling backend as training)

This uses the **same** `sample_episodes` generation function and ALFWorld EnvPool as training, with no Teacher. Choose checkpoints or hyperparameters on the `tune` split, **never** on `valid_unseen`. The protocol's primary comparison uses the final checkpoint from fixed 100 updates, without test-set model selection.

```bash
for c in M_native M L_guarded LN_guarded; do
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 python hindsight_judges/exp2_online.py eval \
      --student-device cuda:0 --student /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
      --adapter "/path/to/hindsight_online/${c}_s${seed}/step_00099" \
      --eval-split valid_unseen --eval-reps 3 --eval-seed 20261103 \
      --max-steps 50 --max-new-tokens 512 --max-student-len 6144 \
      --eval-output "/path/to/hindsight_eval/${c}_s${seed}"
  done
done
python hindsight_judges/exp2_online_report.py \
  --eval-root /path/to/hindsight_eval --seeds 0 1 2 \
  --out /path/to/hindsight_eval/analysis.json
```

The independent report computes primary `LN_guarded - L_guarded` held-out success (percentage points), secondary contrasts versus M and M_native, and a **paired two-level bootstrap over games and training seeds**; evaluation repetitions remain nested in a game/seed cell. Use the repo's E4 `e4_07_eval.py` vLLM benchmark as an optional independent evaluation-backend robustness check, not as a replacement for the matched HF generator.

## Interpretation and next step

A positive LN-over-L held-out success difference would support that *other-turn* Student rationales in privileged Teacher evidence can harm on-policy reasoning distillation. It would **not** prove that the current decision's rationale is removed from Teacher autoregressive context, nor establish a complete solution to self-rationale hindsight credit bias. If `DROP_T` accounts for most of the observed E1c effect, the next algorithm should connect a rationale-independent action credit evaluator to the Think-token learning signal, while preserving the Student's actual reasoning prefixes. That needs its own training contrast; this implementation does not mislabel it as completed.
