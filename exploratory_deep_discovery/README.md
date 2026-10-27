# Deep discovery exploratory artifacts — 2026-10-27

Start with [the Chinese research report](../reports/deep_discovery_2026-10-27.md).

Everything in this directory is post-freeze exploratory. No script writes to `data/`, `eval/`, the frozen documentation, or the downloaded CWGT archive. No script runs a model, starts training, or executes a new environment rollout. The only dependency added during this work was matplotlib in the existing `m0` conda environment.

## GitHub package

The review upload contains this report, all derived exploratory tables, scripts, figures, logs and the raw case files linked directly from the report. It does not include the full server-side replay, sibling, model or hidden-state archives. Full extraction requires those original inputs; the stored manifests identify them. The derived tables and result files can be inspected without downloading the full archive.

## Reproduce

Run from the project root, sequentially, using the existing environment (the system `python` lacks pandas):

```bash
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/exploratory_extract.py
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/exploratory_analyze.py
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/exploratory_mechanisms.py
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/exploratory_cases_and_figures.py
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/exploratory_cheap_selection.py
```

Dependencies: numpy, pandas, pyarrow, scipy, scikit-learn, matplotlib; project `15_regimes` and `evalcore` are imported for the frozen Newcombe interval and clustered AUC bootstrap. Extraction discards token arrays before retaining trajectories and uses one process. No parallel joblib workers are started.

## Files and provenance

- `exploratory_input_manifest.json`: original extraction commit, script hash, input hashes and counts. It includes the raw rollout/sibling/replay/peek inputs read by extraction. Prompt hashes for the stricter repeated-text follow-up are separately in `exploratory_prompt_hashes.json`.
- `exploratory_points.parquet`, `exploratory_arms.parquet`, `exploratory_continuations.parquet`: slim raw-data reconstruction, keyed by trajectory and one-based turn; arms add `j`, where 0 is factual and 1–4 preserve the frozen alternative ordering.
- `exploratory_results.json`: decomposition, masked gain, task/time, W1, sibling and continuation analyses.
- `exploratory_mechanisms.json`: same-facts contrasts, masked geometry, fixed diagnostic regressions, action alignment shuffle, CWGT count-only reproduction.
- `exploratory_followup_results.json`: exact-repeat/generic-response restrictions, effective sibling coverage, random-anchor reference, independent concentration and next-action diagnostics.
- `exploratory_cheap_selection.json`: outcome-free one-decision selectors evaluated against factual and uniform-five controls, with gold strata.
- `exploratory_mention_pairs.parquet`, `exploratory_same_facts_details.parquet`, `exploratory_masked_details.csv`: point/pair-level results.
- `exploratory_cases.json`: seven concrete cases, including negative/weak replication examples, with observations, candidate actions, counts and subsequent action frequencies.
- `figures/`: four standalone figures, each PNG and SVG. Plot labels are English for portable fonts. The report supplies interpretation and limitations.
- `artifact_manifest.json`: final scripts/results/report hashes, software versions and validation checks.

## Definitions that matter

1. `q_12`, `q_34`, `q_56`, `q_3456`, `q_gold` use respectively 6, 6, 6, 12, 20 continuations per arm. Live was selected using 1–2; labels Fork/credit used 3–6. Gold is independently seeded but is a stratified selected cohort, not a new randomly drawn population. Independent outcome noise does not make this post-hoc study confirmatory.
2. Bootstrap resamples main trajectories (one per task), keeps all their points, and estimates point-weighted means or ratios of sums. Default seed 27102026, B=4000; diagnostic logistic AUC uses B=2000. Models use GroupKFold(5), fixed C=1 and no feature search. These CIs do not adjust for choosing scientific hypotheses after seeing results.
3. `target_mention` requires the task target word followed by an instance number. `new_to_history` means an instance ID absent from all prior/current factual observations and admissible action lists; it is not semantic novelty of all properties. `new_to_prompt` compares IDs against the actual saved pre-action prompt. Repeated-text follow-ups require full observation-string containment, not only a repeated instance ID.
4. Same facts means identical sorted fact additions/removals from the same prefix. All selected pairs also have no fact changes. `exploratory_cheap_selection.py` asserts identical post-action admissible action sets and `done`/`won`. This does not separately manipulate observation text and action-memory text, and does not claim a byte-identical full engine snapshot.
5. Sibling visits and distinct trajectories are separate. History support matches two preceding observation/action pairs plus current observation and the sorted admissible set; the strictest support also matches turn. Action-list ordering and all unobserved simulator variables are not matched.
6. `take_after_first3` measures whether a target-taking command appears in continuation indices 1–3, excluding the forced action at index 0. It records attempted commands, not verified successful manipulation. Usage differences are descriptive, not causal mediation.
7. Best-alternative gain selects on one pool and evaluates that fixed action on another. No best arm is reselected on Gold. If no cheap-feature-positive arm exists, the cheap selector retains factual; otherwise it randomizes uniformly among all positive arms, including factual if eligible.
8. All gains refer to one changed decision followed by the original continuation policy on historical prefixes. They cannot be added across time or advertised as measured end-to-end policy improvement.
9. CWGT uses all 172 paired-bank turns in its local evidence file. M0 Fork/Masked labels are not transferred. The local input SHA256 is recorded in `exploratory_mechanisms.json`.

## Execution notes

Early extraction attempts and preexisting B+C repair jobs were interrupted; see `EXPLORATORY_PROTOCOL.md`. All five final scripts completed. A missing matplotlib dependency and a JSON boolean-key plotting error were fixed before the final successful figure run. Missing Gold values are excluded, not counted as failed replication; non-Live discovery rankings/gains are missing, not artificial values. Frozen B+C results have no verified completed repair and are excluded from the report's scientific evidence.
