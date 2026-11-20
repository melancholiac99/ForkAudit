# VC-OPD package preflight

Run as a module from the checkout based on credit-grok 5a076ef5a. Source env/verl_agent.sh. Link the checkout's .runtime to the verified server assets. Reserve four idle A100s. Keep run directories owner-only.

For a new independent run, use prepare RUN (validates full model/data checksums), collect RUN, score RUN and report RUN under exploratory_shop_audit.visible_opd_v0. Each accepts the same directory. prepare requires a new directory. The current registered run and parser repair are documented in SMOKE.md. Use a systemd transient service for a persistent launch, include the checkout as WorkingDirectory, and record the code commit for each phase. Run the tests with python -m unittest exploratory_shop_audit.visible_opd_v0.test_visible.

To finish the same P-pre after authorization, store the user's stage and approved additional GPU-hours in private formal_authorization.json with stage=remaining_92_P_pre, additional_gpu_hours and smoke_summary_sha256. The latter is SHA256 over the actual smoke_summary.json file bytes. Run python -m exploratory_shop_audit.visible_opd_v0.formal RUN once. The driver holds a run lock, sets a wall timer for four allocated devices, reuses the first eight frozen tasks and reads only the remaining 92 for new collection. Complete batch directories survive interruption. Report-only access to all 100 tasks follows scoring. The final packet decision is in formal_report/packet_selection.json. No downstream stage is launched by this command.

Do not treat partial shards or averaged historical baseline rewards as completed observations. An unexpected host kill requires reconciling elapsed GPU allocation before resuming. Inspect the cost ledger and GPU workers first. The runner accounts four reserved GPUs from model/environment initialization through teardown, including idle intervals. It excludes CPU-only scoring summaries.


The §16 run shop_visible_opd_v04_s16_20261120 supersedes the smoke report for current decisions. Its authorization receipt references the previous run to carry forward all formal charges. The driver measures eight tasks with new templates, then measures the remaining 92 in the same service. Keep the old data and receipts intact. The separate CPU replay_funnel, acceptance and finish_a5 commands verify native rewards, recompute conditional scores and write formal_report/REPORT.md.

At phase B, use coverage freeze CAL_SCORES RUN OUTPUT to freeze N_c, and coverage pilot FROZEN OUTPUT FILES... on the ten updates for each arm, as modules under exploratory_shop_audit.visible_opd_v0. Inputs and class coverage are documented in coverage.py. The evidence-only page restriction is common to every arm when evidence is selected. The task count, two-rollout batch and coverage rules are registered before actor smoke; the present GPU allocation ends at A.5.

## Stage B under §§17–18

Use the frozen §16 A.5 directory as the parent. The CPU-only `action_gate` module takes the parent and a new destination and verifies every saved score shard. The dated Stage B paths are defined in `b_collect.py`. The dated run uses `b_collect --pilot`, then `b_score --pilot` for the eight-task P0. Continue with `b_collect` and `b_score` on the remaining tasks after P0 passes. The private rollout tapes feed the CPU `b_equivalent_pool` feasibility enumeration and the three-view scorer. All module names here are relative to `exploratory_shop_audit.visible_opd_v0`.

After verifying 512 complete tasks and the scored feature for every step, run one `b_fit FOLD` process per GPU for folds 0–3. Run `b_metrics` and `b_ridge`, then `b_bias` and `b_verify`. The verification checks the registration commit bytes, shard provenance, model hashes, OOF partition and scores. The machine-readable outputs and counts are in [the Stage B evidence directory](evidence/content_b_20261120/). Stop at this handoff.

## Section 19 and the Stage C smoke

Use the signed Stage B run as input. With CUDA hidden run `b_ridge --section19` and `b_section19` as modules under `exploratory_shop_audit.visible_opd_v0`. The new JSON files go to a separate `section19/` directory. The recorded CPU ridge run used ten OpenMP/MKL threads.

The single-update smoke uses the dated private directory in `c_collect.py`. From the repo root with `env/verl_agent.sh` sourced, run `python -m exploratory_shop_audit.visible_opd_v0.c_collect`, then `torchrun --standalone --nproc-per-node=4 -m exploratory_shop_audit.visible_opd_v0.c_targets`, then the same torchrun invocation with `c_smoke`. After all five arms finish, run `python -m exploratory_shop_audit.visible_opd_v0.c_verify` on CPU. The actor reads the common student shard and only the selected arm’s detached target. The private score/target preparation and the verification have access to the saved views. The report records the initial and synchronized final adapter hashes for each independent step. Stage D requires a separate launch.
