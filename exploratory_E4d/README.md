# E4d: 死局救回（旧策略单步 vs 训练后续执行）

独立实验，**不是 E4 训练臂 D，也不是 E4b 的重训练**。研究方案与结果阅读规则见 [`PREREG_E4d.md`](PREREG_E4d.md)；`credit-grok` 的**仅次要探针/日志修订**见 [`CREDIT_GROK_AMENDMENT.md`](CREDIT_GROK_AMENDMENT.md)。没有做任何训练；所有模型均复用 `exploratory_E4b/data/adapters/{Azero,Dzero,Agap,Dgap}_s{0,1,2}`。

## 前置条件

- 服务器已安装 E4 `m0` 环境（ALFWorld、vLLM、torch、LoRA）；完整原游戏目录在 `alfenv.ALFWORLD_DATA`。
- **必须**存在 E4 原始 `exploratory_E4/data/rollout/valid_unseen/base/s{seed}_r0.json`（134 局）。这些文件 `full=False`，没有原决策 prompt，E4d 从动作与 `obs_after` 确定性重建。缺文件或观察不一致立即停止，**不重新生成 baseline**。
- **必须**存在 E4b 的 4 组 × 3 个训练 seed 的 `adapter_model.safetensors`；无则从原 E4b 服务器存档恢复，不允许改模型或重训练。
- `M0_MODEL` 设置为 E4/E4b 同一 Qwen2.5-7B 基座路径。GPU 命令应在完整仓库 checkout 的根目录执行；不使用 E5-L 的训练数据。

## 先执行 CPU 测试

```bash
cd ForkAudit
python -m unittest discover -s exploratory_E4d -p 'test_e4d.py' -v
python -m py_compile exploratory_E4d/e4d.py exploratory_E4d/e4d_common.py
```

## 服务器分阶段运行

以下命令在 `ForkAudit/exploratory_E4d/` 下运行。目录 `data/` 是运行数据，Git 已忽略；不要删旧 E4/E4b 的原始数据。

```bash
cd ForkAudit/exploratory_E4d

# 1. 旧评估轨迹重建、严格核对、搜索阶段+10步余量+四个替代，先资格后每局三等间隔取点。
python e4d.py candidates --workers 20
# 查看 data/candidates.json（资格统计）和 data/selected.json（仅预选点）

# 2. 旧策略筛选：可以先只跑20点试点估算耗时/比例，只准观察 pi0。
python e4d.py screen --limit 20 --workers 20 --max_concurrent 400
# 通过旧策略成本/比例门后，续跑剩余点（已完成20个的 JSON 将跳过）。
python e4d.py screen --workers 20 --max_concurrent 400

# 3. 冻结全部 pi0 screening、Z3 点集与 E4b 模型权重 SHA。先审核；一经锁定不重选。
python e4d.py lock --approve --probe-n 72  # 默认 72；可在任何 pi1 结果前改为 96，不能事后改

# 4. 必须预先生成全部4组×3训练种子的 pi1 当前一步，再开始任何正式续跑。
for arm in Azero Dzero Agap Dgap; do
  for seed in 0 1 2; do
    python e4d.py first --arm "$arm" --seed "$seed"
  done
done

# 5. 新种子独立 Base，24条/状态；共享用于所有 arm 的 00 条件。
# 首次正式 run 会先生成并锁住 data/first_manifest.json（全部12份首步文件的 SHA），之后禁止更改。
python e4d.py run --arm base --cond 00 --workers 20 --max_concurrent 400

# 6. 主比较（zero训练模型，四格中的01/10/11；00在上一步已跑）。
for arm in Azero Dzero; do
  for cond in 01 10 11; do
    python e4d.py run --arm "$arm" --cond "$cond" --workers 20 --max_concurrent 400
  done
done

# 7. 预先登记的次要对照（gap训练模型），不是主 reading。
for arm in Agap Dgap; do
  for cond in 01 10 11; do
    python e4d.py run --arm "$arm" --cond "$cond" --workers 20 --max_concurrent 400
  done
done

# 8. 完整性检查、按任务做bootstrap 10,000次，输出 e4d.json、E4d_REPORT.txt。
python e4d.py analyze
```

### credit-grok：72/96 点 × 5 臂 × 8 次的独立次要探针

`lock.json` 的 `probe_ids` 在**观察任何 π₁ 结果前**冻结。默认 72 个 Z3 点；如果成本允许，在 `lock --approve --probe-n 96` 时预先选 96 个（成本约 +33%，之后禁止更改）。每臂对 `base`、`Azero`、`Dzero` 分别续跑 **8 次**，两半 4+4 双向交叉估计；Azero/Dzero 的 8 次是**跨三个已有模型种子**轮流分配（3/3/2），**不是每个种子再跑 8 次**，统计时种子等权。72 点共 8640 次续跑（含 Base）。

```bash
# 已完成上述 lock、全部首步采样与 first_manifest 冻结后
for policy in base Azero Dzero; do
  python e4d.py armprobe --policy-id "$policy" --workers 20 --max_concurrent 400
done
python e4d.py probe-analyze  # -> e4d_armprobe.json，次要判读/前向 O(h) 预测
```

`candidates` 阶段先在 CPU 上对每个锁定候选的**事实 + 四个替代动作各试一步**，存下五臂的完整环境观察、目标出现标志、`O_h`（一部分臂看见目标而另一部分没看见时为 1）。任何 π₁ 续跑开始之前锁住 `O_h`，不能后算后选。`screen`、所有四格 `run` 和 `armprobe` 保存每条续跑中首次看见任务目标实例的**全局回合号**；主报告额外包含找到目标和找到后成功的**描述性**分解，原四格主判定规则不变。注意这组 O(h) 需要**执行候选一步**，不是免费状态特征。

#### 未来 RL/MCR checkpoint 的相同样本重测

可以在 **已有 lock/first_manifest** 之后，单独注册不可变的 LoRA adapter 或完整模型 checkpoint（权重哈希+修订编号）：

```bash
# 下面只示例一个 LoRA checkpoint。可给出 3 个路径实现三种 seed 等权轮换。
python e4d.py register-probe --policy-id mcr-v1 --revision R1 \
  --checkpoint-kind lora --checkpoint-paths /path/to/mcr_adapter --approve
python e4d.py armprobe --policy-id mcr-v1 --workers 20 --max_concurrent 400
python e4d.py probe-analyze --policy-id mcr-v1

# 完整 GRPO/MCR 模型也可注册，单个 checkpoint 对应一个 policy ID：
python e4d.py register-probe --policy-id grpo-v1 --revision R1 \
  --checkpoint-kind full --checkpoint-paths /path/to/full_checkpoint --approve
python e4d.py armprobe --policy-id grpo-v1
python e4d.py probe-analyze --policy-id grpo-v1
```

所有未来 checkpoint 使用同一批 `probe_ids`、五臂和八个后续随机种子。该接口**只影响次要 armprobe**，不会加入/修改 E4d 的原主检验或原 reading。如果已经存在旧版 E4d 筛选/lock 输出，须先将它们单独存档，再在全新数据目录重新执行本分支的前向 O(h) 冻结，不允许根据已观察的 π₁ 结果追算或重选。

**待后续独立决定：** 约 40 个 V3 点的同样 2×2 对照目前没有实现；不能假装该结果已经存在，也不要在看过 Z3 的 π₁ 效果后才追补成确认性对照。

## 判定与边界

主检验 = `success(π0→π1) − success(π1→π0)`，要求在 Azero/Dzero 合并后、按局 bootstrap 的 95% CI **下界 > 0**，并要求 `success(π0→π1) − success(π0→π0)` 的 CI 下界 >0；最少 80 Z3 状态 / 40 局，否则标记功效不足。不把筛选阶段的 0/15 当成 Base 真值。第一步由 π1 产生的动作，无论与 a0 重合或超出原四个候选，都**不能过滤**；报告其比例。

**防误跑提示：** `scripts/11_replay.VllmStream` 不自动传 LoRARequest；本实验的续跑采用 `stream_for(pol)`，显式给 `.add_request(..., lora_request=pol.lora)`，否则假装“训练策略接管”其实会悄悄用 Base。已加入 CPU mock 检查，真实 GPU 上仍需核对 vLLM 版本和 checkpoint 哈希。阶段输出可续跑：完整 JSON 已存在则校验锁哈希并跳过，缺失才继续。

**没有运行过 GPU E4d**。CPU 通过只证明实现中的纯函数、关键桥接与统计在假数据上按预定规则运行；真实服务器上还要完成环境/模型前置门。

## 服务器独立双卡队列（2026-11-07）

执行修订见根目录 `DECISIONS.md` D-065。`run_server.py` 等待现有 hindsight 的 GPU 工作完成，
然后运行 Base 双卡 smoke、候选构建、20 点 pi0 试点、全部筛选、`lock --approve --probe-n 72`、
全部首步采样、主四格与三种策略的次要五臂探针，最后分析并独立核验。
screen/run/armprobe 支持 `--shard-count 2 --shard-index 0|1`；按状态 ID 的固定哈希分卡，
同一状态的条件和重复不拆到不同卡。样本和统计门槛不变。

```bash
python exploratory_E4d/run_server.py
python exploratory_E4d/finalize_server.py --thread "$CODEX_THREAD_ID"
```

前置 `run/plan.json` 保存模型、输入、源代码和 GPU 配置哈希。进度为 `run/status.json`，
CPU 检查为 `run/cpu_verification.json`，20 点成本为 `run/pilot_budget.json`，
最终核验为 `run/verification.json`。`finalization/status.json` 监控两个独立分支的发布与最终关机。
数据只写本目录，输入链接仅指向旧 E4/E4b 的只读文件；credit-grok 结果只推 `credit-grok`。
目标发现日志在每条完整记录发布时同步原子保存，避免关机留下只有结果却没有目标日志的断点。
