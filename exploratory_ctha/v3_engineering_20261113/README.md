# Engineering acceptance only

The user cancelled the four-round learning pilots on 2026-11-13, before any policy update. The healthy FP32 repeat_a collector is adopted without changing its source or output. Its original controller and launcher were stopped individually; the collector and GPU worker were left running.

This separately registered orchestrator uses the unmodified `v3_fp32` collector, trainer, direction gate, probability gate and stability gate. It adds a CPU restore test of the actual probe LoRA/Adam checkpoint and independent result verification. No Step-0 Tune, four-round GRPO/GiGPO, full P1/P2 or A100 experiments run here. The fresh output directory is `/root/autodl-tmp/ctha_v3_engineering_20261113`.

After success, the queue wakes the assistant to independently review `verification.json`, `REPORT.txt`, the probability/direction evidence and checkpoint test. Run `publish.py --data /root/autodl-tmp/ctha_v3_engineering_20261113` to package evidence (including raw records, excluding model/optimizer binaries whose hashes and local paths are retained), publish only to credit-grok, and verify the remote commit. Only after successful review and publication, sync and execute `/bin/bash /usr/bin/shutdown` per the user instruction. On failure, wake for repair without silently relaxing gates or changing frozen code. Do not claim learning effectiveness.

The actual checkpoint next-step equality check uses fixed diagnostic gradients on CPU copies. It does not substitute for a future second on-policy training round. Migration to A100 requires a new hardware registration and numeric/memory checks; current hardware success is not A100 certification.
