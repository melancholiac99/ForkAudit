# Hindsight: Think localization and Experiment 2

**Status: implemented / CPU tests passed / no GPU or student-learning result yet.** Independent exploratory work on `research/hindsight-judges` only. Do not modify E1/E1c, P4, E4 or `main`. E1c is completed; its 32B FP16 numerical baseline is reused without cross-dtype subtraction.

## Research questions

1. **Localization:** Is full-log hindsight bias predominantly caused by the rationale at decision `t`, by *other turns'* rationales, or by rationale content? Using the original 386 E1c points and five candidate actions, compare old frozen `L` / `LN` with the new `DROP_T` (remove only `t`), `KEEP_T` (keep only `t`) and exploratory `SWAP_T` (replace `t` rationale with an unrelated trajectory's rationale). Only written `<think>` spans change; each historical action, environmental observation and terminal outcome remains fixed. Donor assignment is deterministic and comes from another original trajectory. These are intervention effects, **not** proof of a psychological self-rationalization mechanism. Assess factual-action log odds **and** independently replay-referenced expected choice value; use trajectory-cluster bootstrap.
2. **Training:** Does rationale isolation in **Teacher's additional hindsight input** improve real Student learning while keeping `<think>` gradients? Compare three separate *trained students*: `M`, `L_guarded`, `LN_guarded`. Every condition teaches the same Student-generated response token positions, including all `<think>` tokens. The model under training never receives privileged future logs. No direct credit weighting is included in this first implementation.

**Crucial anti-leakage guard:** `L_guarded` removes `<think>` at the **currently supervised turn** from the duplicated privileged episode, but keeps think from other turns; `LN_guarded` removes all visible think from the privileged episode. Both retain the current action and environment outcome. Otherwise `L` could reveal the exact target text of the token being predicted, invalidating a direct OPD comparison. This differs intentionally from E1c's unguarded `L` and must be labelled accordingly.

**Training objective:** on-policy-prefix **forward KL approximation** using the normalized top-`k` Teacher distribution, evaluated on prefixes visited by a Student rollout; the Teacher is frozen Qwen2.5-32B-Instruct and the Student is Qwen2.5-7B-Instruct with LoRA. This is *not* full-vocabulary KL, reverse KL, or reinforcement learning. Require mean captured teacher top-k probability mass >=0.90 or stop and increase `--topk`. One round's data is on-policy at collection time. To remain on-policy in later rounds, collect fresh rollouts from each resulting adapter; never reuse the initial rollouts indefinitely.

## CPU checks

```bash
python -m unittest discover -s hindsight_judges -p test_exp2_pipeline.py -v
python hindsight_judges/think_localization.py selftest
python hindsight_judges/exp2_opd.py selftest
```

## Localization: run after reviewing protocol

```bash
python hindsight_judges/think_localization.py prepare
python hindsight_judges/think_localization.py score \
  --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct
python hindsight_judges/think_localization.py analyze \
  --arms /path/to/original-matching/exploratory_arms.parquet
```

Inputs: existing frozen E1c `results/exp1c_p4/{requests.jsonl.gz,manifest.json,audit.json,32b/scores.jsonl,32b/run.json}`, P4 tracked archives, E1 point metadata and original exact deep-discovery arms. This does **not** need the original P4 bank-1 replay directory again. Scores: `hindsight_judges/results/think_localization/32b/scores.jsonl`; analysis: `.../analysis.json`. 386 × 3 = 1,158 new 32B score jobs, plus frozen L/LN reused only if they are from the validated E1c run. Requires the original 32B checkpoint; no silent truncation; numerical gate before scores. No GPU executed by this code publication.

## Experiment 2: first on-policy round

Same frozen E4 TRAIN game split across all three arms. **Do not use the original P4 386 experimental points as a training/evaluation test set.** Collect each training round with E4's original vLLM sampling / ALFWorld harness. The first round is a shared base rollout; subsequent rounds use independently collected samples from each adapter.

```bash
# 1. Collect base student trajectories (the same episodes for all three arms).
python hindsight_judges/exp2_opd.py rollout \
  --rollout-out /path/to/hindsight_exp2/round0_rollouts --max-games 400 --reps 1

# 2. Freeze a deterministic set of 512 Student-visited decisions.
python hindsight_judges/exp2_opd.py prepare \
  --rollouts /path/to/hindsight_exp2/round0_rollouts \
  --out /path/to/hindsight_exp2/round0 --max-examples 512

# 3. Sequentially release 7B from VRAM, load frozen 32B Teacher, score all 3 contexts.
for c in M L_guarded LN_guarded; do
  python hindsight_judges/exp2_opd.py teacher \
    --out /path/to/hindsight_exp2/round0 --condition "$c" --topk 256
 done
python hindsight_judges/exp2_opd.py audit --out /path/to/hindsight_exp2/round0

# 4. Three separate Student LoRA training jobs, equal data/budget/seed.
for c in M L_guarded LN_guarded; do
  python hindsight_judges/exp2_opd.py train \
    --out /path/to/hindsight_exp2/round0 --condition "$c" \
    --save /path/to/hindsight_exp2/adapters/${c}_s0 --epochs 1 \
    --grad-accum 16 --seed 20261103
 done

# 5. Real held-out ALFWorld success; use E4's validated evaluation pipeline.
# E4's e4_07_eval.py resolves adapter NAMES beneath exploratory_E4/data/adapters.
# Symlink/copy the three adapters there, then:
python exploratory_E4/e4_07_eval.py --split valid_unseen \
  --adapters M_s0 L_guarded_s0 LN_guarded_s0 --reps 3
```

For a second round, call `exp2_opd.py rollout --base-adapter /path/to/adapter --rollout-out /new/rollouts`, then use a **new `--out`** and train again with `--base-adapter` pointing at that arm's previous adapter. Keep conditions and compute matched. Do not point `prepare` at validation rollouts; it validates the game files against the frozen E4 TRAIN list. Report game-level paired bootstrap confidence intervals on held-out successes and multiple seeds; token-level Teacher logprob / KL summaries are **diagnostics only** and cannot establish gradient correctness.

## Caveats / decision points

- E1c's L → LN value improvement does **not** automatically imply training improvement. The concrete OPD contexts are guarded, not byte-identical to E1c L/LN.
- The Teacher's privileged episode still includes the actually executed action/observation, so action leakage/provenance bias is not removed; both privileged conditions are matched on this information.
- All three teacher conditions are evaluated on the *same response token IDs*, but 32B's tokenization must exactly match 7B's or scoring refuses. The metadata/hashes prevent silent resume on a different dataset.
- Top-k normalized CE changes the original teacher distribution; `--min-top-mass` enforces fidelity. This is an implementable first-round OPD baseline, **not** a claimed novel method until training outcomes and comparisons exist.
- Current implementation uses Qwen2.5-32B in memory for Teacher and Qwen2.5-7B for Student in separate sequential processes. It assumes matching frozen model weights and E1c precision/backend; numerical controls should be inspected on the actual GPU before claiming reproducibility.
- Do not report an improvement until `valid_unseen` is independently evaluated. No GPU training or held-out evaluation was run in this development environment.
