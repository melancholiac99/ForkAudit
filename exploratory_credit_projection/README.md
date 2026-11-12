# Counterfactual credit projection audit (exploratory, 2026-11-12)

**Question:** in E4's five-arm executed replays, are large measured action gaps less predictable from what the Student saw? This is a mechanism audit, **not a proof** of "Decisive Steps Teach the Least".

## Reproduce on CPU (no model/GPU)

From the repository root with numpy, scipy and scikit-learn installed:

~~~bash
python -m unittest discover -s exploratory_credit_projection -p 'test_audit.py' -v
python exploratory_credit_projection/audit.py --source e4 --out /tmp/credit_projection_e4
~~~

If the original E4/data/points and E4/data/replay directories are available, the script reads them. Otherwise it **streams the existing five-part archive** at exploratory_E4/results_archive, verifies each part's SHA256, and reads only decision-point and replay JSON members. It does not extract the 1.1GB tree. The original dataset contains 6,016 counterfactually defined points (400 train games), 5 arms and 3 binary continuations per arm; the script checks files and reports actual counts rather than inventing results. No validation episodes enter training.

Run with --source m0 only if the original **unpublished M0 data/replay files** have been restored locally alongside the tracked prompts. These raw M0 replays are not in credit-grok. The script fails explicitly without them. E4 is the primary fully archived source.

Outputs (at --out): summary.json, REPORT.md, points.csv, replicate_holdout.csv. Results default to the gitignored exploratory_credit_projection/results_local directory. A GitHub Actions workflow runs the synthetic tests and the archived E4 CPU audit and uploads its outputs as an artifact when the workflow is enabled.

## What is measured

1. The prediction input is the **original Student decision prompt plus each candidate action**, with no hidden state, peek, future result, full history, game ID or outcome features. An action-only model is a negative control. Note: the original prompt may already contain available actions; this is allowed because the Student saw them.
2. GroupKFold leaves out entire games and refits TF-IDF vocabulary and a ridge regression (fixed alpha=10) within each training fold. Predictions Qhat(h,a) are out-of-fold and clipped to [0,1]. Compare held-out MSE with a train-fold mean baseline and the action-only baseline. Failed generalization may reflect a weak predictor, not intrinsically unlearnable credit.
3. The main diagnostic selects the observed (3-run) top and bottom actions and reports observed gap, signed projected difference and positive clipped predicted/observed gap ratio. **The ratio is not a true realizability proportion**. Bins preserve identical gaps. With 3 binary outcomes per arm, the max-minus-min gap has only 4 possible values (0, 1/3, 2/3, 1), so 10 distinct deciles cannot be obtained without arbitrary tie-breaking.
4. An independent-replicate negative control selects best/worst on n-1 outcomes per arm, then evaluates their gap using the held-out outcome. Rotate the holdout across 3 repetitions. This detects **winner's curse / rollout sampling noise**, but is noisy with only three replays.
5. A game-cluster bootstrap summarizes the association of observed gap with clipped ratio. **The gap is in the denominator of the ratio**, so an apparent downward slope is partly mechanical. Inspect the signed projected difference and independent-replicate gaps before drawing substantive conclusions.

## Interpretation and limits

- **Suggestive:** high-gap bins have small projected advantage even after independent-replicate checks suggest that the gap is real, and prompt models perform meaningfully above baselines.
- **Against the proposed trend:** high-gap advantages are consistently predictable from Student prompts and persist in independently evaluated outcomes.
- **Inconclusive:** sparse all-zero rewards dominate, predictor performance is poor, or high gaps shrink toward zero on holdout. Such results do not separate hidden-state aliasing from model capacity and sampling noise.

This experiment cannot directly compute EVPI, since there are no matched samples of different hidden simulator states conditional on the **same Student information and action set**. It does not test whether projected preferences improve DPO. Do not claim a causal explanation of the original E4 negative results from this diagnostic alone.

All changes live under exploratory_credit_projection/ and its independent workflow; frozen E4/E4e data, preregistrations, training scripts and reports remain untouched.
