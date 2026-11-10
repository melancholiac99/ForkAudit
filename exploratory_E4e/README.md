# E4e: staged failure funnel and four-arm recovery audit

**Status:** exploratory engineering follow-up; not part of any E4/E4b/E4c/E4d lock or pre-registered test. Do not overwrite existing reports.

Purpose: before building an online GRPO loop, quantify where complete episodes stall and check whether the specific *wrong held object* problem can be improved by giving the student explicit carrying information, enforcing a legal put action, or both. No training happens in this module.

## 0. CPU sanity checks

From the repository root:

```bash
python -m unittest discover -s exploratory_E4e -p 'test_e4e.py' -v
python -m py_compile exploratory_E4e/*.py
```

These tests are lightweight mocks, not evidence that the real ALFWorld / vLLM environment is installed or that the rules recover tasks.

## 1. Funnel existing rollout episodes (no GPU)

The E4 summary JSON does **not** include individual observations. Run the following against the server checkout **with its original** `s{seed}_r{rep}.json` episode files. Raw episodes may be absent from the published branch; `funnel.py` raises rather than silently fabricating them.

```bash
python exploratory_E4e/funnel.py   --source base=exploratory_E4/data/rollout/valid_unseen/base   --source aprime=exploratory_E4/data/rollout/valid_unseen/Aprime_s0   --source aprime=exploratory_E4/data/rollout/valid_unseen/Aprime_s1   --source aprime=exploratory_E4/data/rollout/valid_unseen/Aprime_s2   --compare base aprime --out /tmp/e4e_funnel.json
```

The funnel is **task-specific**:

- ordinary pick-and-place and lamp-look: found → picked → completed;
- clean / heat / cool: found → picked → processed → completed;
- pick-two: found/picked require **two distinct target instances** → completed.

`completed` is always the environment's verified `won`; for lamp-look it is **not** incorrectly called a placement. `found`, `picked` and `processed` are reconstructed conservatively from action feedback; a success implies prerequisites. Missing milestone text in a failed rollout can be a detector miss, not a real task failure. The report contains `unrecognized` tasks and the observed first missing milestone; **the first-missing counts are not causal recovery budgets or theoretical success-rate gains**.

Check stage parser sensitivity manually on samples of both successful and failed episodes before relying on aggregate results. In particular, verify task-specific ALFWorld response wording and `pick_two` behavior. Never discard unrecognized tasks when stating aggregate rates.

## 2. Run the same funnel on an existing public RL policy

Use a *locally available, compatible* full checkpoint (for example a public GiGPO 7B checkpoint) and use exactly the same games, prompt parser and step cap as the base run. It must first be checked for architectural and vLLM compatibility. No training required:

```bash
python exploratory_E4e/four_arm_eval.py --split valid_unseen   --policy-tag gigpo7b --model-path /path/to/local/gigpo7b   --conditions base --reps 3 --workers 20
python exploratory_E4e/funnel.py   --source gigpo=exploratory_E4e/data/rollout/valid_unseen/gigpo7b/base   --out /tmp/e4e_gigpo_funnel.json
```

Compare stage-conditional rates, not just a single success number. Do not interpret an RL-vs-DPO difference as a causal effect of RL unless the training configurations are matched.

## 3. Whole-game 2×2 (inference only)

Two factors: append a state line constructed from **agent-visible past observations**; and enforce one **currently admissible** `put/move` action for a known non-target held object. The override consumes a real environment step and never invents a command. If carrying status is unknown or no legal put is offered, it does nothing.

```bash
python exploratory_E4e/four_arm_eval.py --split valid_unseen   --policy-tag base --conditions base state hard both   --reps 3 --workers 20
# For an E4 Aprime LoRA checkpoint, if its adapter is present locally:
python exploratory_E4e/four_arm_eval.py --split valid_unseen   --policy-tag aprime_s0 --adapter exploratory_E4/data/adapters/Aprime_s0   --conditions base state hard both --reps 3 --workers 20
```

- `base`: original prompt and model sampling.
- `state`: add explicit carrying information (or 'unknown' when not established).
- `hard`: maintain original prompt, but substitute a legal release command whenever known to carry a non-target.
- `both`: both interventions.

Generated paths are `exploratory_E4e/data/rollout/{split}/{policy-tag}/{condition}/s{seed}_r{rep}.json` and `summary.json`. The outputs contain the sampled model action, actual executed action, `rule_triggered`, `overridden`, held state and the full feedback log. Repeated runs are resumable; **do not reuse a policy tag after changing checkpoint weights or configuration**, because completed episodes are skipped.

Summarize each arm with the CPU funnel:

```bash
python exploratory_E4e/funnel.py   --source base=exploratory_E4e/data/rollout/valid_unseen/base/base   --source state=exploratory_E4e/data/rollout/valid_unseen/base/state   --source hard=exploratory_E4e/data/rollout/valid_unseen/base/hard   --source both=exploratory_E4e/data/rollout/valid_unseen/base/both   --compare base hard --out /tmp/e4e_four_arms.json
```

**Critical limitations:** `hard` uses a deterministic, fully tracked observation-history rule and is **not** a deployable model skill. Legal put commands may put an object in an inconvenient place; this is intentional (no oracle selection of destination). Feedback-derived carrying state may be incomplete, so intervention coverage and false positives must be audited. Use original common-random-number seeds for comparability, but report paired game-level intervals; identical seeds do not guarantee identical stochastic trajectories across different prompts.

For method selection, base vs state tests information exposure; base vs hard tests a rule intervention; hard vs both tests whether the two are complementary. Check unnecessary release rate and total override count. A statistically weak or non-positive whole-game effect should **not** be interpreted as proof the behavior is unimportant.

## 4–5. Conditional next steps (not implemented here)

Only if the whole-game intervention produces a meaningful recovery signal: generate **train-split only** corrective preference pairs (`correct put` preferred over `factual wrong action`), with matched negative examples where release is not appropriate. Reuse E4's DPO path, then compare standard GRPO from original and corrective initializations. Keep valid splits evaluation-only, avoid using 53 validation-state analyses as training data, and use an existing validated online RL implementation rather than writing a new optimizer.

This module does not claim any new success rates or run a GPU evaluation on its own.
