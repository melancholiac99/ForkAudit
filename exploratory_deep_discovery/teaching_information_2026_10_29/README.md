# Information conditions and transferable action preferences

2026-10-29, exploratory CPU reanalysis. No teacher inference, training, or new
environment interaction. Frozen source data and old labels are unchanged.

Run from the project root:

```bash
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/teaching_information_2026_10_29/analyze_transfer.py
```

The script reads all 3,840 raw records from the completed five-arm information
intervention (24 different tasks, 8 per original stratum, 5 actions, 2 conditions,
16 repeats). It verifies every binary outcome against the compact outcome export
and records source SHA-256 hashes in `input_manifest.json`.

For each task it splits the 16 repeat indices into 8 fitting and 8 evaluation
indices, keeping all actions and both information conditions in the same block.
It estimates separate native and information-completed success vectors, maps
each to a candidate selection distribution, and evaluates BOTH distributions
on the same held-out native and completed outcomes. It averages both directions
of 256 random partitions inside each task; these are NOT additional samples.
Uncertainty is 5,000 bootstrap draws over the 24 distinct tasks (8 per stratum).

The primary illustrative mapping is softmax(Q/0.2); temperatures 0.1 and 0.5 and
uniform tie-breaking argmax are reported as sensitivities, not optimized.
These choices were written in the script before its first execution, after
the source intervention outcomes had already been examined. This remains
post-hoc exploratory analysis, not preregistration or blind confirmation.

In the original Fork stratum, native-derived preferences give 66.33% native
success, versus 42.38% for completed-derived preferences: a paired difference
of 23.94 pp [10.13, 38.08]. On completed-condition evaluation, the same rules
give 80.85% and 81.66% (difference 0.81 pp [-2.62, 4.18]). This shows an
asymmetry in the portability of these outcome-derived rules. It does NOT
establish a difference in actual teacher logits, an optimal policy, or trained
student performance. Uniform choice on native outcomes is 39.53%; the
completed-derived rule is not established to be worse than uniform.

Outputs:

- `summary.json`: task-level results and intervals, all strata and rules.
- `transfer_points.csv`: per-task cross-fitted performance and preference mass.
- `action_profiles.json`: full observed vectors, actions, centered contrasts.
- `preference_transfer.png` / `.svg`: representative profiles and transfer cost.
- `verification.json`: raw consistency, split separation, absence of new rollouts.

Interpretation limits:

1. Cue completion is added AFTER the forced action to the continuation. A
   hindsight teacher receiving evidence BEFORE it scores an action is a
   different intervention. Do not label the completed-derived rule as an
   actual Hindsight Teacher or infer that hindsight distillation is harmful.
2. The cases are selected for available true location clues and do not estimate
   population prevalence. Seven of eight original Fork points have a positive
   primary transfer penalty; the negative case is retained.
3. This is a once-only candidate selection comparison followed by the original
   continuation policy. It is not a trained policy or whole-episode gain.
4. Much of the privileged-condition flattening can follow from saturation near
   success. That is relevant to target transfer, but not a novel theorem or an
   independently established model failure mechanism.

Scientific interpretation and a proposed next test are in
`reports/teaching_information_2026-10-29.md`.
