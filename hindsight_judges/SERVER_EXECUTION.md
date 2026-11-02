# Experiment 1 v2: server execution

Source freeze: `2fab12da4d24223111bf83ff46d10b6015766421`.
Local worktree branch: `research/hindsight-judges`.
GitHub destination: `research/hindsight-judges` (researcher's explicit correction).
Source data remains read-only at
`/root/autodl-tmp/idea`; model weights remain under `/root/autodl-tmp/models`.
All new requests/results are under `hindsight_judges/results/exp1_p4` here.

Before new scores, implementation is aligned with EXPERIMENT1_PROTOCOL.md:

- Shared bootstrap seed 20261102 for all metrics, including both primary scales.
- The primary >=1/3 q12 gap value check has a matching Gold subgroup. The older
  Gold gap>0 group remains available as descriptive output.
- GPU UUID, driver, Python/library versions and math flags locked across resume;
  all ten conditions for a point must come from the same model load. Partial or
  malformed blocks are preserved and rejected; no silent repair or score mixing.
- Require complete held-out q3456 for all 386 points. Validate log-probabilities,
  softmax probabilities, argmax, request and file hashes, and numerical checks.
- Verify primary identity and value statistics independently after analysis.

These are protocol-alignment and execution checks, not new scientific thresholds.
The original v2 protocol remains unchanged. The prepared requests must be byte
identical to original exp1.py output. Gold is a sensitivity subset, not an unbiased
sample of the whole population. 7B is secondary; primary conclusions use 32B.

The queue requires E4b successful final verification and an idle GPU, then runs
32B score/analyze/verify and 7B score/analyze/verify, followed by a result inventory
and publication to `research/hindsight-judges`. No E4b/P8 scientific file changes.
It never invokes old shutdown/repair watchers. A failure saves logs and stops.

The 2026-11-02 original 32B attempt stopped before writing scores at its numerical
gate. The researcher approved the recorded execution amendment in
`EXECUTION_AMENDMENT_20261102.txt`: use SDPA MATH for all 32B/7B conditions,
fingerprint it, and preserve the failed attempt separately. The original
scientific protocol and TV threshold remain unchanged. Diagnostic evidence is
under `diagnostics/numerical_gate_20261102/`.

The later optional 7B BF16 attempt failed three gates before scoring. The researcher
requested further repair; `EXECUTION_AMENDMENT_7B_FP32_20261102.txt` records a 7B-only
FP32 recovery. All 11 diagnostic checks passed. The queue preserves verified 32B
artifacts byte-for-byte and uses the separate FP32 scorer/precision verifier for
7B. Both models still use MATH, with explicitly different transformer precision.
