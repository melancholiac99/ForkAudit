# Completed server evidence import · 2026-11-02

These source snapshots are imported **by exact existing Git blob hash** into main, without merging the source branches or replacing main's decision log.

- E4, source `server/e4-results-20261101` @ `f03b5713ce126d426087c4275772c2d5b7f53097`: full non-weight archive, compact reports, training/evaluation logs, frozen code and verification; `run/final_verification.json` passed and registered interpretation R-L-HEURISTIC preserved.
- P8, source `server/p8-preflight-20261101` @ `d24c3d0fd3b177c477d8fa2c25b15eb4f462ba25`: completed CPU audit and verified pre-sampled phase C inputs, code, failure diagnostics. **GPU phase C did not complete** (7B prior-score reproduction failed its strict TV < 0.005 gate). This import makes no P8 phase C or S5 claim.
- E4 post-hoc label composition: two files from `review/pending-20261102` @ `892b8f0ba528a0f23fd727dc757d3a8347116397`, exploratory; NOT an E4b training result.
- Original E4b motivation proposal: `proposals/E4b_zero_replay_credit_learning_value_2026-11-02.md` is historical, superseded by the separately frozen E4b preregistration.

Integration rules: retain main's E5 amendments and paper docs. Preserve main's `exploratory_E4/e4_03_judge.py` (E5 candidate-count compatibility fix), rather than older server E4 variant. Transfer E4 `e4_01_rollout.py` and `e4_06_train.py` from the independently verified H800 run. The server-side decision logs had colliding D-059/D-060/D-061 numbering; preserve both originals under `review_sources/` and add new non-conflicting entries to the root log separately.

No GPU work or E5 experiment is initiated by this commit.
