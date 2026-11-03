# Hindsight Experiment 1c — review implementation

**Status:** DRAFT / NOT FROZEN / GPU NOT RUN. This package is for your code and protocol review. Do **not** merge into `main` or use E1c judge results as a registered claim before sign-off.

Read [EXPERIMENT1C_DRAFT.md](EXPERIMENT1C_DRAFT.md) first. E1c is separate from E4c (a different experiment in `main`). Existing E1 / P4 / E4 / E4b / E4c inputs are read-only.

## Seven inputs

`M` (pre-decision factual memory), `L` (original full factual episode), `LN` (L without the Agent's *visible written* `<think>` blocks), `LT` (L cut after step t, no terminal result), `D0` (M + two anonymous full replay branches), `DE` (D0 + truthful factual provenance), `DX` (D0 + swapped provenance). E1c adds DE/DX so the causal identity effect can be measured under *identical two-branch hindsight evidence*; D0 vs L alone cannot make that inference.

## Source prerequisites (important)

On the original P4 research server, provide:

- Existing P4 `exploratory_P4/e2_requests.jsonl.gz`, `e2m_requests.jsonl.gz`, their matching preparation files and `e2.json`.
- Existing `hindsight_judges/results/exp1_p4/{pointmeta.json,manifest.json}` plus P4's tracked `data/task_list.json` (100 original task paths and game SHA256s). **Full `data/rollout/*.json` is optional:** only 50 such files are tracked on GitHub, but all factual prefix actions and observations can be recovered losslessly from the E2 L requests. The script cross-checks any original rollout JSON that is present.
- **Original P4 bank-1** `data/replay/<traj_id>/t<turn>.json`, with the complete `continuations[].actions` (this directory is **Git-ignored**, so GitHub clone alone is insufficient).
- ALFWorld/TextWorld 0.4.2 environment with the original P4 game files, pointed to by `ALFWORLD_DATA` (default `/root/autodl-tmp/alfworld_data`).
- Exact `exploratory_deep_discovery/exploratory_arms.parquet` from `research/deep-discovery-2026-10-27` (not the source `E4` replay estimates). Hash is matched to the original P4 `e2.json`.

If the old P4 bank-1 replay JSON is missing, **stop and recover it**. Missing Git-tracked rollout JSON **does not** block E1c, because the source factual prefix is recovered from the frozen E2 L requests and `data/task_list.json`. Do not infer trajectories from aggregate q values, take a different E4 sample, or ask the policy to generate replacements.

## Commands, in order

```bash
# Keep this review experiment isolated from the completed research branch.
cd /path/to/ForkAudit
python -m unittest discover -s hindsight_judges -p test_exp1c.py -v
python hindsight_judges/exp1c.py selftest

# Confirm all 386 original P4 replay records and P4 ALFWorld games exist.
python hindsight_judges/exp1c.py preflight \
    --replay-root /path/to/original-P4/data/replay

# CPU ONLY: reconstruct the bank-1 rep-0 factual/best-alternative trajectories.
python hindsight_judges/exp1c.py prepare \
    --replay-root /path/to/original-P4/data/replay \
    --arms /path/to/exploratory_arms.parquet

# CPU tokenizer gate before contemplating GPU; hard stops if any of 2702 requests >32768 tokens.
python hindsight_judges/exp1c.py lengths \
    --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct
```

**Only after the researcher explicitly approves a specific branch HEAD, and after every CPU/length gate passes**, the proposed full run is (replace the SHA placeholder with that approved full commit, not an arbitrary current HEAD):

```bash
python hindsight_judges/exp1c.py score \
    --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct \
    --approved-commit <researcher-reviewed-full-HEAD-SHA>
python hindsight_judges/exp1c.py analyze \
    --arms /path/to/exploratory_arms.parquet
```

All new outputs go to `hindsight_judges/results/exp1c_p4/` (Git-ignored) and are bound by source, checkpoint and script SHA; the experiment-one score files never change. `score` always re-scores the original `M` and `L` in the *same* run, so old RTX6000D/H800 scores are never used as baselines. The server must archive the complete run manifest, requests/audit hashes, numerical checks, scores, and analytical JSON before publishing any result.

## Review questions / known limitations

The decision cohort is fixed (386 points, 60 original trajectories). The only 32B pre-registered facts reused are from experiment one; **1c's new hypotheses are not frozen**. The core choices are: two branches not five; bank-1 replica 0 without outcome-based selection; same 50-step *episode* cap; compare full L versus two-branch D0 as a *representation comparison*, and DE versus DX as a controlled identity cue. Some dual logs could exceed the 32B tokenizer/context or H800 memory limit; no silent truncation or GPU rerun is authorized by this draft.

The CPU tests are synthetic. We have not performed a full P4 replay on this workstation, because the original P4 `data/replay` directory is not in the Git repository. **The server should run `preflight`, `prepare`, and `lengths`, then report hashes and counts for review before starting any GPU scoring.**
