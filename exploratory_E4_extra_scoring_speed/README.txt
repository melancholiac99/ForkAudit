Exploratory scoring speed test — requested 2026-10-31

This experiment does not modify the registered E4 scorer, protocol, scores,
cost matching, model precision or thresholds. It does not train a model.

batched.py preserves shared-prefix scoring, BF16 transformer and FP32 lm_head,
but forwards candidate suffixes in batches with explicit padding masks.
Memory guard reduces batch size for long contexts. Requested sizes: 2 and 5.

CPU validation:
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /root/miniconda3/envs/m0/bin/python benchmark.py --selftest
The tiny random Qwen2 test checks four prefix lengths and variable suffix
lengths, batch sizes 1/2/5, and TV < 0.005. It is not a real-model result.

GPU benchmark selects text-length quantiles 0, 0.5, 0.9 and 1 in each F/L
condition before looking at scores. Original and optimized timings alternate
order across two repetitions, with synchronized timing and peak memory.
Original/direct checks use shortest and longest selected requests per
condition. Every batched result must match original probabilities at
TV < 0.005. A failure stops the benchmark and restores the formal queue.
No speedup on this small selection proves full-workload speed or equivalence.

hook.py suspends only the waiting orchestration parent (PID 2683), while the
formal F worker (PID 15319) runs unchanged to completion. Once F exits and
the GPU is free, it runs the benchmark with a 30-minute timeout. Finally it
resumes the original queue, which runs L with the original scoring backend.
An independent guard resumes the queue if the hook dies unexpectedly.
Thus benchmark time never enters formal F scoring_seconds or replay timing.

Outputs:
  cpu_selftest.json — engineering CPU checks
  hook_status.json — waiting/running/complete/failed
  benchmark_progress.json — per-request measurements as they finish
  benchmark.json and REPORT.txt — numerical gates, timings and speedups
  benchmark.log and hook.log — execution diagnostics

There is no automatic production-backend replacement, publication or shutdown.
