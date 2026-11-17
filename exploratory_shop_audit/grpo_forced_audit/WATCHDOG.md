# Shopping experiment watchdog

Researcher request: “做个监控，有问题的时候唤醒你去修代码”. The user authorizes background Codex diagnosis and routine implementation repair within this experiment. Preserve the scientific protocol and all saved measurements. Driver changes/reboot or destructive recovery require user handling.

A systemd timer checks every 60 seconds. Confirm faults on two consecutive checks; allow one repair at a time, no more than one new incident every 30 minutes, and no repeated wakeup for the same continuously present fault. Detect service failure, stale heartbeat (>3 minutes), missing saved progress (>40 minutes), NVIDIA/kernel errors, memory below 8 GiB, disk below 30 GiB (critical below 20 GiB), new recorded exceptions and TensorBoard outage. GPU utilization alone never means failure. Old exceptions are baselined at installation.

The monitor writes `watchdog/status.json`, `incidents/*.json`, and enqueues a background `codex exec` invocation using the existing ChatGPT login. This is a dedicated background Codex invocation, not a guaranteed push notification or resumption of the currently open chat. It uses no experiment GPUs. Codex/account/network availability is required. The monitor records agent failures; it cannot repair a powered-off host or a broken kernel that prevents execution.

Repair instructions: read the incident and current evidence, determine whether intervention is necessary, fix only implementation errors, test, retain failed evidence and resume from verified state with saved samples. Never rewrite research outcomes, silently replace trajectories or choose favorable retries. No automatic reboot, driver changes, raw data removal, main merge or unrelated experiment changes. For partial updates/checkpoint uncertainty, stop with NEEDS_USER instead of resetting from an older checkpoint blindly. Record what was actually verified. Agent process completion is not proof the issue was fixed.

Status: `cat .runtime/runs/shop_grpo_forced_20261117_01/watchdog/status.json`.
Reports: the same folder's `repairs/<incident>/REPORT.txt` and `status.json`.
Disable: `systemctl stop forkaudit-shop-watchdog.timer`; this does not stop an already running repair. Stop that separately with `systemctl stop forkaudit-shop-repair.service` if needed. A `watchdog/DISABLED` file also suppresses monitoring.

CLI interface verified locally (0.167.2), authenticated as ChatGPT. Official non-interactive interface: https://learn.chatgpt.com/docs/non-interactive-mode . Monitoring changes operational handling only; numerical research definitions remain frozen.

Deployment verification (2026-11-17 09:10 UTC): production monitor broker dispatched synthetic incident `probe-1794906608` through systemd; background Codex returned `WATCHDOG_READY`, exit 0. No tool use or experiment change was permitted in this probe. Interrupted repair receipts block automatic repetition of the same incident. Operational unit tests cover confirmation, deduplication, recovery rearming, cooldown, disable, disk/progress thresholds and diagnostic command timeouts.
