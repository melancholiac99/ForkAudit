# E4d: 死局救回（旧策略单步 vs 训练后续执行）

独立实验，**不是 E4 训练臂 D，也不是 E4b 的重训练**。研究方案与结果阅读规则见 [`PREREG_E4d.md`](PREREG_E4d.md)。没有做任何训练；所有模型均复用 `exploratory_E4b/data/adapters/{Azero,Dzero,Agap,Dgap}_s{0,1,2}`。

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
python e4d.py lock --approve

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

### 可选且已预登记：五臂价值是否重新显现

正式样本 Z3 在 `lock.json` 里已冻结，并在里面确定前 24 个用于此项。此项不影响主 reading，必须在锁定后独立运行：

```bash
for arm in base Azero Dzero; do
  python e4d.py armprobe --arm "$arm" --workers 20 --max_concurrent 400
done
python e4d.py probe-analyze  # e4d_armprobe.json
```

## 判定与边界

主检验 = `success(π0→π1) − success(π1→π0)`，要求在 Azero/Dzero 合并后、按局 bootstrap 的 95% CI **下界 > 0**，并要求 `success(π0→π1) − success(π0→π0)` 的 CI 下界 >0；最少 80 Z3 状态 / 40 局，否则标记功效不足。不把筛选阶段的 0/15 当成 Base 真值。第一步由 π1 产生的动作，无论与 a0 重合或超出原四个候选，都**不能过滤**；报告其比例。

**防误跑提示：** `scripts/11_replay.VllmStream` 不自动传 LoRARequest；本实验的续跑采用 `stream_for(pol)`，显式给 `.add_request(..., lora_request=pol.lora)`，否则假装“训练策略接管”其实会悄悄用 Base。已加入 CPU mock 检查，真实 GPU 上仍需核对 vLLM 版本和 checkpoint 哈希。阶段输出可续跑：完整 JSON 已存在则校验锁哈希并跳过，缺失才继续。

**没有运行过 GPU E4d**。CPU 通过只证明实现中的纯函数、关键桥接与统计在假数据上按预定规则运行；真实服务器上还要完成环境/模型前置门。
