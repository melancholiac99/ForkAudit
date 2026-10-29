# Information × execution repair factorial

Exploratory 2026-10-29. [Fixed protocol](PROTOCOL.md).

**Completed: 480 new + 480 explicitly reused continuations, all integrity checks
passed; main run 649.6 seconds.** [Chinese findings and interpretation](../../reports/execution_factorial_results_2026-10-29.md).
All 960 full raw traces are included in `results/all_960_continuations.tar.gz`.
The user briefly cancelled shutdown, then reauthorized it. The instance will
shut down after upload, with a three-minute fallback.

Six previously specified mechanism cases, original five actions, four conditions,
8 continuations per cell. The 480 native/completed controls reuse documented
records from the 2026-10-28 experiment; 480 repair continuations are newly run.
The original Phase 3 data and labels are unchanged.

`prepare.py` restores all 30 branches and freezes original held-object repair
targets. `run.py` uses the original Qwen/vLLM policy and environment pool, adding
at most one legal release at its first opportunity. The repair consumes a real
environment step. `analyze.py` retains complete action-value profiles, estimates
unbiased action variance, and checks every trace's step and clue accounting.

Commands on the research server, in order:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/execution_factorial_2026_10_29/prepare.py
bash exploratory_deep_discovery/execution_factorial_2026_10_29/run.sh
```

Preparation requires the original source archives. Do not overwrite the fixed
design in an existing run; use a new directory for a changed protocol. A run
refuses to mix checkpoints with different code/configuration hashes.

CPU preflight: 30 original branches, 22 flagged for an unrelated held object,
14 with immediate legal release. A 20-continuation random-policy dry test passed
budget, one-shot repair, cue presentation, and seed checks; it is not scientific
outcome data. The remaining 8 branches have no initially flagged object.

`shutdown_guard.py` is an operational tool for this user-authorized server run,
not part of the experiment. **Do not run it to reproduce the science**: it shuts
down this AutoDL instance on completion signal or at its cost deadline.
