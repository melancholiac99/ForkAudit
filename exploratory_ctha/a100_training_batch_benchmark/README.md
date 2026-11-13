# A100 training microbatch benchmark

Compute-only test on the original eight real responses and saved R3 engineering adapter; zero optimizer updates. The active pilot is unchanged. Microbatches2/4 preserve the full-response objective and all-row denominator.

Measured speedups: 1.0221x and 1.0350x versus the unchanged sequential implementation. Both passed the preregistered probability/loss/gradient checks. This small gain is not a deployment or end-to-end speed claim. No candidate is adopted. Gradient binaries remain local under .runtime/runs/a100_training_batch_benchmark_r1/ with hashes in each report.

The samples are an engineering subset, not a benchmark performance or learning result.
