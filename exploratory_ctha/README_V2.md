# CTHA v2 — task-type-held-out online agent RL

**2026-11-12, experimental implementation.** The v1 notes in README.md are
historical; use **this** guide for current commands. Changes are isolated to
`exploratory_ctha/` on `credit-grok`; frozen E4/E4e protocols and data are untouched.
**CPU CI is not a GPU training validation.** No model gain is claimed yet.

## What changed since v1

1. The first `--feature-window` decisions (default 8) produce a *rate-normalized*
   trajectory feature vector, but **the base online RL gradient uses every step**
   (`--train-horizon 0`). The optional habit bonus is limited separately by
   `--aux-horizon 8`. This prevents the original first-eight-steps-only update.
2. Deterministic, signed-hashed **generic action/observation interaction features**
   (128 hash bins + generic scalars), separate **task-progress features**, and
   controls `action`, `manual`, `progress`, `both`.
   Neither feature family uses the action's future observation or episode reward.
   The two families are semantically distinguishable, not mathematically
   orthogonal. A causal claim about habits requires the progress-only control.
3. Each iteration uses current train trajectories and a window of previous
   **train** iterations. Ridge regression is group-centered within
   `(iteration,game)`, and only mixed-reward task groups contribute contrasts.
   Each target task **type** is predicted from all *other* task types.
4. **P0 predictive gate:** accuracy in ordering success-vs-failure rollouts within
   a held-out task type is compared against models fitted on labels permuted
   inside *training* task groups. Game-cluster bootstrap of the accuracy
   difference must have a positive lower 95% bound; additionally at least
   `--min-mixed 20` mixed groups and `--min-pairs 200` comparison pairs
   are required. Failing the gate produces **zero habit reward**. No baseline
   success is invented for all-fail groups.
5. **GiGPO-style anchor advantage**: `--base-advantage gigpo` adds (a) episode
   reward minus same-task mean and (b) reward-to-go minus the mean of all steps
   sharing the exact `obs_before` *within that task*. With sparse terminal
   reward and `--gamma 1`, per-step return equals the episode outcome.
   Singleton anchors contribute zero local advantage.
   This is a **single-GPU approximation of the Eq.3/6/7/8 structure** in
   `langfengQ/verl-agent/gigpo/core_gigpo.py`, not a reproduction of the
   reference distributed verl-agent code, its anchor extraction, or training
   results. Ordinary `--base-advantage grpo` is also supported.
6. Only all-failed task groups receive the auxiliary habit signal by default;
   mixed/all-success groups use their original GRPO/GiGPO-style signal.
   `--omega` stays constant until `--anneal-start`, then decays linearly
   to zero at `--anneal-end`. No auxiliary gradient after annealing.
7. A **frozen-policy collection mode** lets you execute P0 before RL: it
   collects rollout data from the *same* model and never trains the actor.

This is a reward-shaping / reward-decomposition **hypothesis**, not an unbiased
step-action value or evidence of the "projection law".

## Server launch

Start from the original server repo checkout. Use its ALFWorld and vLLM
environment; do not install or change GPU libraries based solely on CPU CI.

~~~bash
cd /root/autodl-tmp/idea
git fetch origin credit-grok
git checkout credit-grok
git pull --ff-only origin credit-grok

export M0_MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct

python -m unittest discover -s exploratory_ctha -p 'test_ctha*.py' -v
python exploratory_ctha/doctor.py --model "$M0_MODEL" --require-gpu
~~~

### P0: one frozen policy, training tasks only

Collect multiple task/episode contrasts without updating the policy. These
tasks come from the E4 **train** split; `valid_unseen` is NOT accessed.

~~~bash
python exploratory_ctha/run.py \
  --collect-only --run-dir /root/autodl-tmp/ctha_v2/p0_base \
  --model "$M0_MODEL" --rounds 6 \
  --tasks-per-round 32 --reps 4 --eval-every 0 \
  --workers 16 --chunk-games 16

python exploratory_ctha/p0_audit.py \
  --run-dir /root/autodl-tmp/ctha_v2/p0_base \
  --out /root/autodl-tmp/ctha_v2/p0_report.json \
  --feature-window 8 --min-pairs 200 --min-mixed 20
~~~

Examine `manual`, `action`, `habit`, `progress`, and `both`
out-of-type ranking accuracies against their shuffled-label controls.
Also inspect `both_minus_progress` and CIs. Accuracy 0.64 in a previous
mixed-policy **validation** diagnostic is not a promised result here.
The frozen base model may yield too few mixed groups; report insufficient
signal instead of promoting a weak predictor.

### GPU integration smoke, distinct from scientific results

Start small to validate vLLM→token IDs→PEFT update→new adapter→tune evaluation.
`--train-horizon 2` is smoke-only, NOT the main objective.

~~~bash
python exploratory_ctha/run.py \
  --run-dir /root/autodl-tmp/ctha_v2/smoke \
  --model "$M0_MODEL" --mode ctha --base-advantage gigpo \
  --rounds 1 --tasks-per-round 4 --reps 2 \
  --feature-window 2 --train-horizon 2 --aux-horizon 2 \
  --min-mixed 2 --min-pairs 2 --permutations 2 --bootstraps 20 \
  --workers 8 --chunk-games 4 \
  --eval-every 1 --eval-games 6
~~~

A smoke may correctly produce **zero** auxiliary signal because held-out
task-type validation requires at least three types and mixed outcomes.
Do not interpret that as a research result.

### P1: main paired comparison

The training trajectory is full length: `--train-horizon 0`. Keep sample
budget, model initialization, game ordering, K, and update hyperparameters
identical. These commands are one-seed pilots; repeat several seeds before
claiming a gain.

~~~bash
COMMON="--model $M0_MODEL --rounds 12 --seed 0 --tasks-per-round 16 --reps 4 --feature-window 8 --train-horizon 0 --aux-horizon 8 --eval-every 2 --eval-games 40"

# Episode-level GRPO
python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha_v2/grpo_s0 \
  --mode grpo --base-advantage grpo $COMMON

# Single-GPU GiGPO-style anchor grouping baseline
python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha_v2/gigpo_s0 \
  --mode grpo --base-advantage gigpo $COMMON

# Fixed-rule bonus control: same GiGPO-style base
python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha_v2/fixed_s0 \
  --mode fixed --base-advantage gigpo $COMMON

# Proposed generic habit credit: same GiGPO-style base
python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha_v2/ctha_s0 \
  --mode ctha --family habit --base-advantage gigpo $COMMON

# Progress-only alternative. Do NOT claim habit gain without this comparison.
python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha_v2/progress_s0 \
  --mode ctha --family progress --base-advantage gigpo $COMMON
~~~

The default cold start may need multiple iterations before mixed group/pair
thresholds are met. `advantage_report.json` records gate reason, mixed pair
counts, task-type coefficients, auxiliary fraction and effective omega. If no
auxiliary advantage ever activates, CTHA **has not been tested**; increase
training-task coverage or use a compatible stronger initial model, without
changing the evaluation split.

### P2: ablation controls

~~~bash
python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha_v2/ungated_s0 \
  --mode ctha_nogate --family habit --base-advantage gigpo $COMMON

python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha_v2/both_s0 \
  --mode ctha --family both --base-advantage gigpo $COMMON

python exploratory_ctha/run.py --run-dir /root/autodl-tmp/ctha_v2/no_center_s0 \
  --mode ctha_nocenter --family habit --base-advantage gigpo $COMMON

python exploratory_ctha/summarize.py \
  --run grpo=/root/autodl-tmp/ctha_v2/grpo_s0 \
  --run gigpo=/root/autodl-tmp/ctha_v2/gigpo_s0 \
  --run ctha=/root/autodl-tmp/ctha_v2/ctha_s0 \
  --run progress=/root/autodl-tmp/ctha_v2/progress_s0 \
  --out /root/autodl-tmp/ctha_v2/learning_curve.csv --threshold 0.80
~~~

Primary analysis: **training environment steps to fixed tune success threshold**,
tune success vs cumulative train steps, and wall-clock/GPU budget. Evaluation
episodes do not contribute to policy training. Tune is for development, not
final claims; once hyperparameters are fixed, evaluate on valid_unseen without
re-fitting coefficients.

## Rules for valid comparisons

- Policy v2 is **not** reproducibly identical to the official GiGPO code:
  comparison to original GiGPO must identify the difference. Full reference
  verl-agent integration, P3 randomized causal rule validation and P4 second
  environment are *not yet implemented* in this package.
- No P0 or online fitting from 28-model E4 validation traces. Only the train
  games recorded in `round_*/rollouts/` are read by v2.
- Within-game episode contrasts are **correlational**. A positive pair-ranking
  score or rule weight does not prove the action caused reward.
- Progress and generic action features may correlate; separate P0 and online
  feature ablations are necessary to attribute any improvement.
- Adjacent samples inside one game are correlated; report game-level intervals
  across independent seeds. Study all-failed-group *counts* per iteration:
  the auxiliary window naturally shrinks as policy improves.
- The v2 training fingerprint includes source file SHA256 and all
  hyperparameters. To change features, thresholds or optimizers, use a **new**
  run directory. Existing v1 run directories must not be reused.
- Runtime: full 50-step rollout updates are materially more expensive than v1's
  first-8-step-only updates. Validate GPU memory and time on the smallest smoke,
  then scale task counts; do not start all arms before smoke success.

## Files

`features_v2.py` — hashed interaction and progress features;
`habits_v2.py` — type-heldout regression, shuffle-null, anchor and shaped advantages;
`p0_audit.py` — frozen-policy P0 diagnostic;
`prepare.py`, `run.py` — v2 batch preparation/alternating training;
`rollout.py`, `train.py` — vLLM and PEFT/PPO;
`test_ctha_v2.py` — v2 CPU checks.
