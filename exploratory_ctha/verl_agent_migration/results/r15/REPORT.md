# Real ALFWorld rollout-session integration

**Scoped engineering integration: PASS. Learning: NOT_RUN. Generalization: NOT_RUN. CTHA online efficacy: NOT_TESTED.**

R15 restores the actual R9 native checkpoint on A100 GPU1 and runs four fixed TRAIN tasks in greedy val-only mode. No optimizer updates occur. The new session stays within one vanilla multi-turn collection, then releases vLLM before HF scoring. Full-response PPO, rewards, task definitions and sampling budgets are unchanged by the adapter.

| Check | Observed result |
| --- | --- |
| Actual environment collection | 129.181 seconds |
| Native sync/wake and sleep | 1 enter, 1 exit |
| Final session state | Closed, not resident, not poisoned |
| Actual checkpoint restore | Weights, AdamW moments/steps and scheduler match |
| Real responses / tokens | 176 / 21,578 |
| Raw-score k3 / clip fraction | 0.00067348 / 0.00273427 |
| Token ESS / response-product ESS | 0.998710 / 0.837790 |
| Observed NVML peak | 58,692.94 MiB |
| Native task success | 1/4, descriptive only |

Independent NumPy recomputation matches the saved diagnostics and original k3 <= 0.02 / clip <= 0.30 gates. Greedy-output raw-score alignment is not an importance-sampling estimate of a stochastic behavior policy; R14 separately checks fresh stochastic responses. Raw token absolute errors remain diagnostic under the approved BF16 protocol.

This real-environment integration is not a paired full-rollout speed comparison: trajectories differ from older runs. The controlled fixed-request speedup is reported separately in [R14](../r14/REPORT.md). Neither run establishes learning improvement or full four-round training completion.

```bash
source env/verl_agent.sh
python exploratory_ctha/verl_agent_migration/audit_session_environment.py \
  .runtime/runs/verl_agent_session_alfworld_r15
python exploratory_ctha/verl_agent_migration/verify_evidence.py \
  exploratory_ctha/verl_agent_migration/results/r15
```

The text-split evidence contains raw score arrays, memory trace and the registered source snapshot. Full actual trajectory packet remains at `.runtime/runs/verl_agent_session_alfworld_r15/validation_batch_0001.pkl`. Model and optimizer binaries remain in the R9 checkpoint directory; their restored content fingerprints are recorded in `restore_rank_0.json`.
