> **CTHA v2 (2026-11-12):** This page documents the superseded v1 prototype.
> For current commands, task-type-heldout tests, full-rollout gradients and the GiGPO-style baseline, read [README_V2.md](README_V2.md).
> Do not reuse v1 run directories for v2.

# CTHA: Cross-Task Habit Advantage for online Agent RL

**Status (2026-11-12):** executable *experimental* implementation on the separate `credit-grok` branch, **not a validated GPU result**. GitHub Actions checks Python syntax, synthetic rollouts/advantage preparation and command construction on CPU only. Run one small server smoke before committing GPU budget. E4/E4b/E4d/E4e protocols/data are unchanged.

**Research objective:** in sparse-reward multi-turn ALFWorld, can a low-dimensional estimate of *reusable habit-level* return contrasts help online policy optimization when every rollout in an individual task group fails? The method is a predictive auxiliary signal, NOT an unbiased per-action advantage and NOT a demonstrated "projection law."

## Algorithm implemented

Each training iteration has a single frozen behavior policy (base model or the previous iteration's LoRA). A vLLM worker runs **K full 50-step-cap episodes per selected train game** and saves raw generated **completion token IDs**, student-visible prompts, legal actions and terminal success. The worker exits before starting the single-GPU PyTorch LoRA update (no simultaneous vLLM/trainer residency).

- **GRPO episode baseline:** `A^E_i=(R_i-mean_g R)/(std_g R+1e-6)`, or exactly zero for all-zero/all-one groups. This episode-level advantage is attached to the first `horizon` steps (the same horizon used in all arms for fair compute).
- **Habit features:** only `prompt, candidate action, task, admissible actions`: examine, goto, revisit within the visible two-step window, goto outside it, open, take task target, take non-target, put and related types. These are intentionally small/interpretable. No simulator state or future feedback in a step feature.
- **Reward contrast regression:** for each episode sum these features over the first `horizon` steps. Within each `(iteration, game)` group, center both episode feature counts and terminal rewards. Fit ridge `beta` on multiple other **games** (current round and a rolling window of prior rounds). Entire focal game ID is excluded from its beta fit, even if visited in earlier iterations. All-fail and all-success groups contribute no regression contrast.
- **Conservative gate:** default requires (a) at least 8 mixed-reward training groups, (b) coefficient sign stability across four game partitions, (c) positive out-of-game R² for within-game reward contrasts. The latter checks predictive validity, **not causal value**. Failed checks yield zero auxiliary advantage and fall back to ordinary GRPO.
- **Local action comparison:** `q_beta(h,a)=beta^T phi(h,a)`. Subtract the uniform mean over *legal* actions at that Student prompt; this is an action-independent baseline, **not** the current policy expectation. In an **all-failed task group only**, add `omega * clip(A^H/std(A^H), -2,+2)` to episode advantage. This is intentionally *shaped* policy optimization, not a claim of unbiased advantage.
- **Policy update:** single-GPU, temperature-matched LoRA PPO/GRPO-style clipped surrogate (per generated completion token), old log-probs computed from the exact frozen sampled actor/adapter. One policy-update phase per rollout batch; 0 KL/reference coefficient in v1. No DPO and no hidden oracle reward. Optional `--action-only` requires exact token-offset roundtrip and refuses unsafe masking.
- **Evaluation:** optionally, a fixed **tune** cohort only. The `valid_unseen` split is reserved for final comparison after hyperparameter selection.

The code does **not** reproduce the full distributed verl-agent GiGPO stack or implement GiGPO's anchor-state advantage. It implements an *independent single-GPU online GRPO baseline* and a CTHA augmentation, to test the core claim before adding distributed training. "Fixed" is a small hand-coded rule-reward control, not an exact reproduction of CovGPO.

## Server setup

From the existing ForkAudit checkout on your GPU server:

~~~bash
git fetch origin credit-grok
git checkout credit-grok
git pull --ff-only origin credit-grok

# Use the existing server environment containing PyTorch, transformers, PEFT,
# vLLM, ALFWorld, TextWorld and numpy. The existing E4 vLLM/LoRA setup is a
# good starting point. Avoid blindly upgrading these packages.
export M0_MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct

python -m unittest discover -s exploratory_ctha -p 'test_ctha.py' -v
python exploratory_ctha/doctor.py --model "$M0_MODEL" --require-gpu
~~~

If running with a **1.5B** model, pass its local directory via `--model` to **every** arm. Do not use a 7B LoRA adapter on a 1.5B base model.

### A. Minimal real-GPU smoke (start here)

Run on a few train games; this **does** perform sampling and policy updates. Use unique run directories and keep this smoke separate from main results.

~~~bash
mkdir -p /root/autodl-tmp/ctha
python exploratory_ctha/run.py \
  --run-dir /root/autodl-tmp/ctha/smoke_grpo \
  --model "$M0_MODEL" --mode grpo \
  --rounds 1 --tasks-per-round 4 --reps 2 --horizon 2 \
  --min-mixed 2 --workers 8 --chunk-games 4 --eval-every 1 --eval-games 6

python exploratory_ctha/run.py \
  --run-dir /root/autodl-tmp/ctha/smoke_ctha \
  --model "$M0_MODEL" --mode ctha \
  --rounds 1 --tasks-per-round 4 --reps 2 --horizon 2 \
  --min-mixed 2 --workers 8 --chunk-games 4 --eval-every 1 --eval-games 6
~~~

The first smoke may have zero active habit coefficients (insufficient mixed-reward groups). That is expected and **does not** establish whether CTHA helps. The smoke verifies CUDA/vLLM and PEFT checkpoint handoff. Look for `round_000/rollouts/summary.json`, `prepared/advantage_report.json`, `adapter/train_log.json` and `eval_tune/summary.json`.

**Do not start a long experiment until the first rollout and policy-update GPU smoke succeed.** CPU CI cannot validate the versions of vLLM / Transformers / PEFT or the GPU memory footprint on your machine.

### B. Main paired comparison

Run baseline and proposed method with **identical** dataset partitioning, task/sample seeds, task counts, K, temperature, LoRA settings and evaluation cadence. Each run is sequential on a single GPU.

~~~bash
COMMON="--model $M0_MODEL --rounds 12 --seed 0 --tasks-per-round 16 --reps 4 --horizon 8 --eval-every 2 --eval-games 40"

python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha/grpo_s0 \
  --mode grpo $COMMON

python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha/ctha_s0 \
  --mode ctha $COMMON

python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha/fixed_s0 \
  --mode fixed $COMMON
~~~

These commands are for one seed/pilot; **repeat multiple seeds for any performance claim**. Increase `--rounds` on a completed run to continue (all other flags and code hashes must remain identical). To resume after interruption, reissue the original command: completed episodes are checked for provenance identity and skipped, prepared data and finished adapters are verified. Changed hyperparameters require a **new run directory**.

If your server has less GPU memory, lower `--gpu-util`, `--tasks-per-round`, `--chunk-games` and `--horizon`; **use the same settings for each arm**. With near-zero successes throughout, within-task return regression cannot learn; use a more competent initialization or a separate curriculum instead of claiming invented dense rewards.

### C. Required ablations and report

~~~bash
# Keep COMMON identical; one run directory per mode.
python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha/ctha_nogate_s0 \
  --mode ctha_nogate $COMMON
python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha/ctha_all_s0 \
  --mode ctha_all $COMMON
python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha/ctha_nocenter_s0 \
  --mode ctha_nocenter $COMMON

python exploratory_ctha/summarize.py \
  --run grpo=/root/autodl-tmp/ctha/grpo_s0 \
  --run ctha=/root/autodl-tmp/ctha/ctha_s0 \
  --run fixed=/root/autodl-tmp/ctha/fixed_s0 \
  --out /root/autodl-tmp/ctha/learning_curves.csv --threshold 0.80
~~~

Primary metric: environment interactions (training steps; evaluations excluded) to a **fixed tune success threshold**. Also report success-vs-env-steps curves, all-failed group share, habit coefficients per round, GPU wall-clock, parse validity and *final* `valid_unseen` performance after selection. Tune evaluation may be expensive; count its cost in total compute even though it is excluded from training interaction curves.

If comparing to a published GiGPO 93% model, clarify that it uses a different training procedure/checkpoint. A strictly controlled comparison requires running the GiGPO baseline under the **same** training budget, not merely displaying its checkpoint success rate beside this run.

## Statistical limitations and safety rails

1. Within-group behavior/reward correlation is not causal action credit. Group centering removes some *task difficulty* but not within-trajectory confounding.
2. Habit coefficients can change as the policy changes. Refitting on fresh on-policy batches is necessary; cross-task transfer may still be poor.
3. There is **no reward signal from which to learn** if all training groups remain all-failed. The algorithm falls back to zero GRPO/CTHA advantages, not an invented improvement.
4. Each source trajectory's early-step features may include post-action consequences indirectly through subsequent observations. Only the feature at a given decision is student-visible; this does not make the learned return decomposition an unbiased local causal effect.
5. Source data used to fit coefficients is always **train** only; group heldout is by game identity. Do not copy E4 valid_seen / valid_unseen rollouts into a CTHA run directory.
6. Completing CPU tests proves file/protocol correctness only. Real inference, LoRA training and multi-round checkpoints must be exercised with the server smoke commands.

## File map

- `habits.py`: observable features, within-task ridge, game-heldout stability and out-of-game R² gate, GRPO and shaped advantages.
- `rollout.py`: fixed-policy resumable full ALFWorld episodes with exact vLLM token IDs.
- `prepare.py`: safe train-only batch construction and source hashes.
- `train.py`: frozen-actor old logprobs, temperature-matched per-token clipped loss, LoRA update/checkpoint.
- `run.py`: single-GPU alternating orchestration, resume integrity, fixed train/tune game selection.
- `summarize.py`: comparable learning curve records and interactions-to-threshold.
- `doctor.py`: server environment preflight.
- `test_ctha.py`: synthetic CPU tests.

No experiments have been run on the user's GPU server as part of this commit. Results must be collected before any training-performance statements.
