# Deep discovery — 2026-10-27

This directory contains post-freeze exploratory analyses explicitly requested by the researcher. Frozen inputs, definitions and confirmatory outputs remain untouched. The contemporaneous B+C repair is owned by another running process; this analysis only reads its completed results.

The preceding sentence records the situation at the start of exploration. That repair was subsequently stopped at the user's request, and no completed repaired result was used.

Initial analysis choices, before computing new associations:

1. Recover five action-conditioned success vectors for banks 1–2 (6/arm), 3–4 (6/arm), 5–6 (6/arm), 3–6 (12/arm), and independent gold (20/arm). Check alignment against the frozen labels.
2. Distinguish outcome variance, between-action variance and chosen-vs-reference advantage. Use disjoint-bank covariance to estimate between-action variance without binomial plug-in inflation. Do not interpret a non-rejection as a true zero effect.
3. In Masked points selected by banks 3–6, choose alternatives using banks 3–6 and evaluate in banks 1–2; disclose banks 1–2 Live ascertainment. Independently corroborate in gold, separating unbiased and candidate strata. Also select using 3–4 and evaluate 5–6 on the full Live set without selecting by the latter outcomes.
4. Audit the semantics of target_revealed using raw peeks: target mention, new target instance versus full history, missing from the actual two-step memory, available take action, physical changes, and factual versus counterfactual observations. These are mechanistic measurements, not additions to the frozen feature model.
5. Count sibling observation matches, unique trajectories, action support, outcome diversity, observation+admissible matches and observation+two-step-history matches. Repeated visits do not count as independent outcomes.
6. Examine task type, time course, concrete continuation sequences and action alignment. Use a small number of prespecified mechanism contrasts, trajectory-cluster bootstrap intervals, and action-shuffle controls where relevant. No GPU training or new replay is started.

All newly proposed intervention results are hypotheses unless actually run and saved here. Math identities are established RL/statistics, not claimed as new theory. A frozen single-policy ALFWorld dataset cannot prove universal impossibility or policy-independent causal criticality.

## Additional mechanistic contrast (specified before inspecting its outcomes)

Compare pairs of candidate actions at the same decision point whose post-action TextWorld fact sets are identical but whose observation texts differ. In particular, compare target-mention versus non-mention arms among pairs with no fact changes, and then check exact equality of sorted added/removed fact sets. This holds simulator facts and elapsed time fixed; it does not hold action-history text fixed. Therefore it can isolate the combined information/context pathway from measured physical-state changes, but cannot by itself separate observation content from remembered action wording. Analyze independently seeded pools and concrete continuations, including previously known versus newly revealed target instances.

## Execution notes

Two early extraction attempts exited with code 137 while other project jobs were running. The user requested stopping background tasks, and all project workers/monitors were stopped. A temporary namespace/sandbox failure then prevented commands. After the user restored execution configuration, extraction resumed as one process with unnecessary token arrays discarded immediately. No new GPU job was launched. The B+C repair logs currently show exit 137; an unverified repaired number must not replace the frozen result silently.

The final extraction and four analysis/figure scripts completed. After seeing the same-facts contrast, stricter exact-text-repeat and generic-response subsets were examined; these are explicitly further post-hoc analyses, not preregistered replications. Their overlapping populations and the weaker unrestricted exact-repeat Gold result are disclosed in the report. An additional outcome-free selector analysis compares response-target, action-target and newly-seen-target rules, with all three retained. No environment intervention has been run.
