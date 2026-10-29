# Actual Hindsight Teacher probe

**Completed.** See the [Chinese results and research judgment](../../reports/teacher_probe_results_2026-10-29.md).
The main probe did not support systematic harm from hindsight. A post-main
explicit-location diagnostic on all 24 mechanism points improved native Fork
recommendations; its separate protocol, inputs and results are in `explicit_clue/`.
Main aligned-feedback gains were small and concentrated in a few cases.

This exploratory experiment tests actual Qwen2.5-7B-Instruct action preferences,
following the earlier outcome-derived preference-transfer analysis. It uses one
GPU for teacher-forced likelihoods and existing continuation outcomes for
evaluation; it does not train a student or generate new environment rollouts.

Read `PROTOCOL.md` for the fixed sample, conditions and interpretation rules.
`NUMERICAL_NOTE.md` documents pre-run numerical checks and implementation changes.

## Reproduce on the research server

The source replay/trajectory files and the earlier `prepared_v1` bundle are
required. Run these sequentially from the repository root, with a clean target
directory for a fresh preparation:

```bash
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false /root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/teacher_probe_2026_10_29/prepare.py
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false /root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/teacher_probe_2026_10_29/score.py
OMP_NUM_THREADS=4 /root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/teacher_probe_2026_10_29/analyze.py
```

Preparation refuses to overwrite inputs. Scoring resumes only saved requests
whose hashes match. Evaluation requires all 576 unique contexts to finish.

## Conditions

- S: unchanged student prompt.
- H: full factual hindsight, including every actual response and environment
  observation plus the factual terminal reward.
- A: H plus five correctly aligned one-step branch observations.
- X: same as A with a fixed derangement of alternative feedback; factual stays.

The student's actual thinking prefix and action opener are held fixed across
conditions. Five complete candidate commands, closing delimiter and EOS are
scored. Probabilities are normalized inside this fixed set. All teacher conditions
share the same instruction to evaluate the original student, and none receives
counterfactual success rates, old regime labels, or hidden environment facts.

## Files

- `requests.jsonl`: model inputs and exact token sequences (no replay outcomes).
- `preparation.json`, `input_manifest.json`, `point_audit.json`: source and format
  checks; 144 points, 74 distinct tasks across two disjoint cohorts.
- `scores.jsonl`: actual model log probabilities, candidate distributions and cost.
- `numerical_check.json`: cached/direct candidate-distribution checks.
- `run.json`: execution cost, peak GPU memory, software and precision.
- `score_main_snapshot.py`: exact main-run scoring implementation, verified
  against the hash saved in `run.json`; the current runner adds a directory option.
- `summary.json`, `values.csv`, `behaviors.json`, `cases.json`: paired evaluation.
- `teacher_results.png` / `.svg`: teacher-value comparison.

The 24 mechanism points have native/completed continuation outcomes. Historical
main contains 120 points, with independent bank-5/6 evaluation for 80 Live points
(63 with all candidates legal). Dead main points have no such independent bank;
their teacher behavior is retained, and missing outcomes are not imputed.

The probe evaluates action supervision **after the realized thought is fixed**.
It cannot establish or refute the effectiveness of full token-level OPSD, whose
supervision can also change the earlier thought. Expected success here is one
candidate choice followed by an unchanged continuation policy, not measured
end-to-end performance of a trained model.
