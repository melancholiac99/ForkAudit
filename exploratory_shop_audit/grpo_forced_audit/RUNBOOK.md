# 2026-11-17 执行入口

工作区：`/root/ForkAudit-shop-20261117`，数据：`.runtime/runs/shop_grpo_forced_20261117_01`。旧工作区仍为`/root/ForkAudit`。当前协议见PROTOCOL.md，具体适配见DECISIONS.md。

CPU预处理（不调用模型）：

```bash
.runtime/shop-review-venv/bin/python -m exploratory_shop_audit.grpo_forced_audit.cli clean
.runtime/shop-review-venv/bin/python -m exploratory_shop_audit.grpo_forced_audit.screen_parallel
.runtime/shop-review-venv/bin/python -m unittest exploratory_shop_audit.grpo_forced_audit.test_experiment -v
```

`checks.preflight(run)`执行两个真实引擎的接口/恢复前检。`analysis.historical(run)`复算旧轨迹。`screen_parallel`在全部任务完成后合并结果并调用split，不能以部分结果提前划分。目录内已有CPU记录续算，不重新选择任务。

GPU流水线（先检查preflight/splits/historical，所有新增GPU工作由budget.execute记录；原24 GPU小时上限由研究者于2026-11-17明确解除，见resource_authorization.json）：

```bash
OPENBLAS_NUM_THREADS=1 .runtime/shop-review-venv/bin/python -m exploratory_shop_audit.grpo_forced_audit.campaign
```

顺序：24×8 pilot；原生GRPO更新1/恢复至10；5题E0吞吐；冻结预算；剩余E0；GRPO每10更新的贪心/随机验证；训练终点对称评测；新终点自然状态E1；CPU报告。第一轮更新前完成V的两种解码基线。脚本遇到失败返回非零，保存日志与已完成记录，不重试选择最好结果。GPU进程结束后清理本次子进程，其他GPU作业不受影响。

独立模块入口`rollout pilot/evaluate`和`audit --label base/final`会调用模型，必须经budget.execute调用。`report.final_report`、`analysis`和`audit.summarize`仅复算。原生训练checkpoint保留model/optimizer/scheduler，resume会逐项核验；原子写入的结束标志及日志决定是否完成，不能只凭目录存在。

关键证据：`live_status.json`、`cost_ledger.jsonl`、`pilot_summary.json`、`training_metrics.jsonl`、`weight_updates.jsonl`、`training/restore_rank_0.json`、`budget_freeze.json`及最终`REPORT.md`。未完成阶段不填成功。权重和完整原始数据不入Git。

当前四卡设置：训练使用GPU0/1/2/3同步数据并行、每卡独立TP1采样；独立审计/评测阶段使用四个TP1副本，分阶段独占四卡。禁止另起进程争抢训练设备。`live_status.json`列出本阶段全部GPU，`active_gpu_sessions/`与`cost_ledger.jsonl`记录累计占用。只有存在明确研究者授权记录才解除上限；进程中断不删除失败账本。完整吞吐证据见PARALLEL_REPORT.md。

从单卡checkpoint1切换时复用保存的update2采样张量；全局mini64、每卡micro4、16题×8轨迹保持。每个rank分别验证恢复指纹，保存前再次核对复制状态一致。`check_parallel_restore`对已保存审计候选在两个独立CPU引擎核验观察、停手评分和实际购买；修正URL路由误读会话ID后43个非终止候选全部通过。

原生采样完整批次保存在`native_batch_cache/`。重启使用原张量或已有精确输入/输出token重建；`native_cache_reuse.jsonl`记录来源和缺失诊断。恢复后仍通过原生actor重算old logprob。未完成的部分批次不静默重新采集。

## TensorBoard 实时监控

运行中的训练无需重启。`tensorboard_monitor.py`只读已有JSON/JSONL，将已完成更新、固定验证点和资源用量转换为事件文件，不改变采样、损失或冻结配置。已核验TensorBoard 2.21.0。

日志目录：`/root/ForkAudit/.runtime/runs/shop_grpo_forced_20261117_01/tensorboard`。当前服务监听服务器`127.0.0.1:6006`；将服务器6006端口转发到本机后访问`http://127.0.0.1:6006`。

复现命令（在本工作区执行，两个进程分别保持运行）：

```bash
CUDA_VISIBLE_DEVICES='' .runtime/venvs/verl-agent-20bd331b/bin/python -m exploratory_shop_audit.grpo_forced_audit.tensorboard_monitor
```

```bash
CUDA_VISIBLE_DEVICES='' .runtime/venvs/verl-agent-20bd331b/bin/python -m tensorboard.main --logdir /root/ForkAudit/.runtime/runs/shop_grpo_forced_20261117_01/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5 --load_fast=false
```

- `training`：横轴是GRPO更新编号。`train/actor/*`为完成更新后的loss/KL/梯度等实际指标；`train/rollout/*`为对应原生批次回报等指标。
- `eval/sample/hard/*`为主验证口径；`eval/greedy/*`为贪心辅助口径；`population_weighted`为还原易难比例的全体指标。
- `signal/*`来自采样和优势检查，可能先于对应优化器更新完成；不能把signal已有step 1解释成第一次更新已完成。
- `runtime_seconds`横轴是监控启动后的秒数，包含GPU使用率/显存、主机内存、累计GPU小时和已完成更新数。

每15秒同步一次，事件每5秒刷新。缺失/NA不写零，未完成更新不伪造loss。`export_state.json`记录已输出点，正常重启不重复回填。监控进程和服务PID分别保存在运行目录的`tensorboard_exporter.pid`和`tensorboard_server.pid`，日志为同名`.log`。TensorBoard与导出器都禁用CUDA，不分配训练GPU。
