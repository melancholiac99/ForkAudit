# Full-response engineering R3 and bounded baseline RL

This is a **new acceptance protocol authorized on 2026-11-13**. R2 remains
**Engineering: FAIL**. Its action-span sign failure is retained. The earlier
proposed NLL sample-selection repair is superseded without execution.

Full-response PPO, original-base KL coefficient 0.01, LoRA and AdamW settings
are unchanged. Only signed full-response probability changes block the new
direction test. Action-span changes remain visible diagnostics. All probability,
finite-gradient, sampled behavior k3 <=0.02 and clip fraction <=0.30 gates remain
blocking. The original eight responses and four updates are unchanged.

New outputs:

- `.runtime/runs/ctha_a100_full_response_r3`: new four-step probe, live/disk
  checkpoint equality, actual CPU AdamW restore, merged/unmerged HF parity and
  updated HF/vLLM parity. R2 same-config frozen-base repeat/rollout evidence is
  explicitly copied, hashed and independently rechecked; it is not relabeled.
- `.runtime/runs/ctha_a100_baseline_rl_r1`: **four rounds each** of GRPO and
  GiGPO-style on the same sixteen train tasks, four trajectories/task. Two
  independent GPUs; original base initialization, no engineering adapter.
  Full trajectories and zero-advantage rows are retained. Persistent AdamW,
  accumulation 64 and one epoch remain unchanged. No CTHA/P0/predictor is used.

Baseline/final evaluations use the same training tasks with a distinct fixed
seed namespace, four episodes/task. Report paired per-game success change and
95% game-cluster bootstrap CI (10,000 draws, seed20261113). Positive change with
lower CI >0 supports basic learning on this fixed training set only. No Tune,
unseen, CTHA efficacy or algorithm superiority inference; no outcome-driven
extensions. This replaces the earlier five-arm/twelve-round P1 draft.

```bash
source env/a100.sh
python exploratory_ctha/a100_full_response/run_workflow.py engineering
# Requires independently verified new engineering PASS and published source receipt:
python exploratory_ctha/a100_full_response/run_workflow.py pilot
```

TensorBoard port 6006 includes `full_response_r3/` runs. Model, LoRA and optimizer
binaries remain local with checksums. Code/protocol/results publish to
`credit-grok` only. No automatic shutdown or automatic CTHA launch.
