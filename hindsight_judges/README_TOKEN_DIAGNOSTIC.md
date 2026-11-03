# Full-response signal supplement

See [the frozen-before-scoring supplement](PROTOCOL_TOKEN_DIAGNOSTIC_20261104.md).
This complements, rather than replaces, the five-action diagnostic. No HLN, new
rollouts, OPD training, or automatic shutdown are part of these commands.

The local prepared cohorts are P4: 386 responses / 52,814 tokens; E4: 1,586 groups /
7,930 responses / 1,058,273 tokens. E4's 7B student reference is mandatory, even though
only a 32B teacher is requested for E4. Same frozen student prompt and response-token
prefix at every position; both directional full-vocabulary KLs are retained.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover \
  -s hindsight_judges -p test_token_diagnostic.py -v

python hindsight_judges/exp3_token_diagnostic.py prepare --cohort p4 \
  --out hindsight_judges/results/token_diagnostic_p4
python hindsight_judges/exp3_token_diagnostic.py prepare --cohort e4 \
  --out hindsight_judges/results/token_diagnostic_e4

# Pin each process to one card. First compute BOTH student caches with frozen 7B.
CUDA_VISIBLE_DEVICES=0 python hindsight_judges/exp3_token_diagnostic.py cache \
  --cohort p4 --out hindsight_judges/results/token_diagnostic_p4
CUDA_VISIBLE_DEVICES=0 python hindsight_judges/exp3_token_diagnostic.py cache \
  --cohort e4 --out hindsight_judges/results/token_diagnostic_e4

# P4: 7B and 32B teachers; E4: 32B teacher against the cached 7B reference.
CUDA_VISIBLE_DEVICES=0 python hindsight_judges/exp3_token_diagnostic.py score \
  --cohort p4 --name 7b --out hindsight_judges/results/token_diagnostic_p4
CUDA_VISIBLE_DEVICES=0 python hindsight_judges/exp3_token_diagnostic.py score \
  --cohort p4 --name 32b --out hindsight_judges/results/token_diagnostic_p4
CUDA_VISIBLE_DEVICES=1 python hindsight_judges/exp3_token_diagnostic.py score \
  --cohort e4 --name 32b --out hindsight_judges/results/token_diagnostic_e4

# For EACH (cohort, name) above, run analyze then verify with the same --out.
python hindsight_judges/exp3_token_diagnostic.py analyze \
  --cohort e4 --name 32b --out hindsight_judges/results/token_diagnostic_e4
python hindsight_judges/exp3_token_diagnostic.py verify \
  --cohort e4 --name 32b --out hindsight_judges/results/token_diagnostic_e4
```

The server uses `run_token_diagnostics_20261104/driver.py` to wait for the existing
candidate/Think-localization GPU jobs, pin by GPU UUID, respect cache dependencies,
run both student caches and P4 teachers on card 0, and the E4 teacher on card 1,
run analyses and verifiers, and wake the agent on failure or completion. Current
candidate results are retained. The first ten E4 scored groups trigger a throughput
notification; before that, wall-time forecasts are provisional.

Outputs: `<out>/<teacher>/full/records/*.json.gz` contain every token's two logps,
delta, reverse/forward KL and segmentation; `analysis.json`, `REPORT.txt`, and
`verification.json` contain results and audits. `student_cache/` is a LOCAL derived
FP32-hidden-state cache (~14.8 GiB combined); never publish it as scientific results
or upload it to Git. `--limit N` has separate cache/output folders for explicit pilots.

The historical text is canonically re-encoded, not the lost original generated token
IDs. E4's shared factual future may favor the factual arm; group rankings are compared
with independent three-replay values and reported only as conditional diagnostics.
Neither a log-ratio increase nor a favorable ranking establishes an effective OPD
gradient or learned policy improvement.
