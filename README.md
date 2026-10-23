# ForkAudit

**Does decision structure beat confidence for step-level credit in LLM agents?**
A replay-referenced three-arm audit on ALFWorld.

When an agent's forward decision is compressed into the chosen action in a log, the decision structure is discarded: the candidate set, the policy scores of the candidates, and their relative and temporal relations. Post-hoc step-level credit estimators (LLM judge, logprob ratio, scalar confidence) only see the compressed trace.

ForkAudit tests whether, against the same execution-replay reference credit (`A_replay`), the following do better than scalar confidence at identifying pivotal steps, predicting step contribution, and resisting action-fluency contamination:

- **Arm A**: scalar chosen-action confidence. This is the control arm and reproduces the CWGT baseline.
- **Arm B**: hand-built decision-structure features over the legal action set (margins, exact entropy, type structure, loops, semantic dispersion, sampled entropy).
- **Arm C**: linear probes on policy hidden states, used as the information upper bound.

This stage does no policy training and proposes no new algorithm. The deliverable is a set of reproducible, controlled measurements with confidence intervals, plus a continue / pivot / stop verdict from a pre-registered rule.

The protocol follows *Credit Without Ground Truth* (CWGT, arXiv 2608.19760). The full protocol, metrics (M1–M5) and verdict rules (R1–R5) are in [`CLAUDE.md`](CLAUDE.md).

## Status

| Phase | State |
|---|---|
| 0. Archive search, environment, template | Done. The CWGT archive was not found, so this project uses Branch B (own replay). See `DECISIONS.md` D-011. |
| 1. Rollout and replay, `A_replay` | Done and independently re-audited (D-019, D-020). |
| 2. Feature freeze, scoring, hidden states | **Not started.** `FEATURES_FROZEN.md` has not been written. |
| 3. Evaluation (M1–M5) and verdict | **Not started.** No feature-vs-`A_replay` association has been computed. |

**There are no headline results yet.** Phase 1 only establishes the reference credit and its measurability structure. Points where it differs from CWGT are reported, not adjusted (D-020).

Phase 1 snapshot (Qwen2.5-7B-Instruct, T=0.7, 50 trajectories, 2,029 decision points, 25,491 continuations):

| Quantity | Here | CWGT |
|---|---|---|
| Counterfactual defined, \|D\| | 1,617 / 2,029 (79.7%) | 86.9% |
| Pivotal among D | 718 / 1,617 (44.4%) | 30.5% |
| Factual σ_floor = 0 among D | 77.2% | 86.0% |

Independent recomputation of `A_replay`, `sigma_floor` and `crit_sd` from the raw replay files gave 0 mismatches in 2,029 points.

## Why the git history matters

The project is pre-registered. `FEATURES_FROZEN.md` must be committed **before** any feature is correlated with `A_replay`, and the verdict rules in `CLAUDE.md` §8 may not change before Phase 3. The commit history is the evidence for this. Please review the history, not just the final tree.

**Open proposal:** [`proposals/P1_L1L2_restructure.md`](proposals/P1_L1L2_restructure.md) (PROPOSED, not adopted) splits the target into state criticality (L1) and action contribution (L2). The proposal was triggered by the Phase 1 finding that all 718 pivotal points fall inside the 758 points whose 15 continuation outcomes are not all identical. It also adds a second independent replay bank with cross-fitting, a state-value baseline, HCAPO ρ_t as an audited hindsight arm, feature families, and revised verdict rules. `CLAUDE.md` is unchanged until the proposal is approved.

All deviations and open questions are logged in [`DECISIONS.md`](DECISIONS.md) (D-001 to D-021), including the ones that currently block Phase 2 (D-003, D-004, D-005, D-021).

## Layout

```
CLAUDE.md          full protocol and pre-registered verdict rules
DECISIONS.md       every choice not fixed by the protocol, with rationale
scripts/           rollout, replay, audit, review scripts (run.sh is the entry point)
data/              decision_points.parquet, rollout trajectories, frozen task list
prompts/           the exact prompt text for every decision point
eval/              audit outputs and the SHA256 manifest of the raw replay files
env/               environment pins
logs/              run logs
```

## What is not in this repo

- `data/replay/` (about 189 MB of raw replay JSON) is not committed. Every file's SHA256 is listed in `eval/phase1_input_manifest.json`, so the data can be verified against the manifest. The files are available on request. [TODO: link to an archive such as Zenodo or Hugging Face Datasets.]
- `hidden/*.npy` (Phase 2) and model weights.
- Some scripts contain absolute paths from the original server (`/root/autodl-tmp/...`). Adjust them before re-running.

## Reproducing Phase 1

Environment: `alfworld==0.4.2`, `textworld==1.7.0`, vLLM, `Qwen/Qwen2.5-7B-Instruct`, using the HCAPO ALFWorld template. See `env/` and D-009, D-013, D-014.

Note that seeded generation is not bitwise reproducible (D-015). Re-running will give statistically comparable, not identical, replays. The audit in `scripts/12_audit_replay.py` checks the stored files for internal consistency.

## Citation

[TODO: add once a preprint is available.]
