# credit-grok：E4d 独立次要探针与可复用 checkpoint 接口

**状态：方案修订 / pre-GPU，2026-11-06。** 本文件在 `credit-grok` 分支新增。它**不替换、重定义或追认** `PREREG_E4d.md` §2–§5 的主 Z3 筛选、四条件 2×2、Azero/Dzero 主组、8×3 续跑、游戏级 bootstrap、80 状态/40 局功效门或 `E4d-SUPPORT / COUNTER / RESCUE-ONLY / INCONCLUSIVE` 判定顺序。`E4d_REPORT` 的主要数值含义不变。

以下次要探针规则在本分支合并并**在任何 E4d π₁ 结果产生前**记录；原 `main` 未修改。若某台服务器已产生旧版 E4d `data/selected.json` / `data/lock.json` / 续跑结果，**不得在该数据目录上事后升级筛选规则或追算 O(h)**，必须采用新目录做独立冻结，旧输出另存档。

## A. 单步环境不对称 O(h)（选择之前冻结）

原 E4d `candidates` 严格要求：E4b 搜索阶段（决策时的 prompt 没有目标实例）、四个不同合法替代、事实动作及前缀确定性重放、`50 - turn >= 10`、每失败局等间隔最多三个资格点。以上不变。**完成全部资格和等间隔选择后**，对选中状态的五臂（事实 + 4 替代）各执行环境一步；只使用 CPU 的 ALFWorld 环境，不读取五臂后续结局、模型成功率或任何训练后策略。五臂一步观察全文、目标实例匹配位以及环境观察哈希随 `selected.json` 保存，事实臂必须与 E4 base 原记录逐字相同。

目标实例沿用 E4 的 `target_word` + 正整数识别口径（例如 `bowl 2`）。当五臂观察中**至少一臂含目标实例、至少一臂不含目标实例**时，`O_h=1`，否则 `O_h=0`。这不要求该实例在更久之前的已截断历史里从未出现；其语义是“在当前可见 prompt 缺乏目标实例时，五臂的一步返回是否不对称”。它是**额外环境交互才能得到的前向实验信号**，不是 Agent 部署时免费获得的状态特征。

在完成旧策略的 15 次五臂筛选后，`lock --approve --probe-n 72`（或 **仅在 lock 前**明确选 `--probe-n 96`）同时写死 Z3、O(h)、探针点 ID 与适配器 SHA。筛选数据是唯一允许在主实验锁定前观察的实验结局（全部属于 π₀），不得通过 π₁ 结果改变探针配额。

## B. Armprobe：点更多、每臂重复更少，使用同一资源预算

旧方案 24 点 × 5 臂 × 24 次 × 3 种策略 = **8,640** 次后续续跑。现在默认 **72 个锁定 Z3 状态 × 5 臂 × 8 次 × {base, Azero, Dzero} = 8,640 次**。预先选择 96 点会使探针成本增加约 33%；若锁定 Z3 不足预设样本数，则取全部并报告实际数量，不填补不替换。主实验每训练 seed×条件 8 次的预算**不动**。

在每种策略/每点/每臂的八次中，复用共同的后续 seed 编号 `u=0...7`，偏移量原定 `9×10^12`。base 由单一模型生成全部八次；Azero/Dzero 的训练种子 0/1/2 由 `SHA256("E4d-probe-seed:"+point_id)` 固定的轮换起点循环分配八次，总配额 3/3/2。任一 `4` 次半样本都有每个训练种子至少一次，统计**先对每个训练种子内的重复取均值，再对种子等权聚合**。如此不把 3/3/2 变成偏向两个种子的估计。代价是每种子没有 8 个回放，无法估计高精度的 seed 独立效果；这是在锁定成本下扩大跨点样本的权衡。

五臂的两个半样本（0..3、4..7）双向 4+4 交叉估计：在 A 半中选回放均值最高/最低的臂，在独立 B 半评分；交换 A/B 重做，取两次均值 `heldout_gap`。除交叉估计外同时报告 `range_naive`（有选择偏差，只描述）、五臂均值、按点/游戏的正 `heldout_gap` 比例。**某一点的正 heldout gap 并不等于对该点完成显著性检测**；判据以聚类均值的置信区间为主，不能把 4+4 的检出功效推演当作 E4d 的已观测数据。

## C. 前向 O(h) 的可证伪预测（仅次要，独立于原 E4d reading）

每个探针点：

- `gap_gain = (heldout_gap_Azero + heldout_gap_Dzero)/2 − heldout_gap_base`。
- `value_gain = (mean_Azero + mean_Dzero)/2 − mean_base`。
- `O_contrast = E_game[gap_gain | O=1] − E_game[gap_gain | O=0]`，同一游戏跨两个 O 层的贡献按共享游戏聚类重抽样。

预先预测：`O=1` 的 `gap_gain` 为正，并比 `O=0` 更大，尤以剩余步数 `10–19` 时为明显；`O=0` 更可能主要表现为 `value_gain`，而不是臂间分化。报告 `O=0/1` 各自样本/游戏数和 10,000 次游戏级 bootstrap 95% CI，并按 `10–19 / 20–29 / 30+` 分层报告；小样本层只作描述，不在事后调整门槛。

**三种可行动读法，加一个不足以判定的退出项（全部为次要研究路线判断，不构成 MCR 算法效果证明）：**

1. `MCR-PREDICTIVE-CANDIDATE`：整体 `gap_gain` CI 下界 > 0，`O=1` 组 `gap_gain` CI 下界 > 0，且 `O_contrast` CI 下界 > 0；O=0 和 O=1 各有至少 10 点及 5 个游戏。此时值得**独立预注册**用 O(h) 预测式重启。
2. `EMERGENCE-NOT-PREDICTED`：整体 `gap_gain` CI 下界 > 0，但没有通过 O(h) 预测门。此时优先测试均匀重启消融；**不能**把 O 预测未达门等同于证明无预测信息。
3. `UPLIFT-WITHOUT-RESOLVED-FORK`：`value_gain` CI 下界 > 0，且 `gap_gain` 置信区间完全处于 ±10pp 的预定描述性等价带内。说明当前 DPO 主要呈现可被整体救回、但不足以支持新分叉；不能说真实各臂严格相等。此时不因 DPO 结果加入该模块，等待 RL checkpoint。
4. `UNRESOLVED`：其他情形，包括群组不足或同时无明确价值提升，完整报告，不强行归类。

这三类行动方向的机制证据未来要在未参与上述规则选择的 GRPO/MCR checkpoint 上重复验证。它们不是训练算法性能的确认性推断，也不能修改 E4d 的主终点。

## D. 每条续跑记录第一次发现目标；救回收益的分解

`screen`、`run`、`armprobe` 均使用相同原始 `11_replay.run_async` 及 ALFWorld API；仅在环境返回值处挂载被动日志层，逐条保存：完整轨迹中的 `target_first_seen_turn`（初始观察计为 0）、`first_seen_after_decision`、`seen_before_decision`、`seen_on_forced_step`、`ever_seen` 和 `seen_after_decision`。匹配口径与 E4c 行为分析一致；目标实例可能在旧历史中看过、却已因两步窗口遗忘，单独报告这种情况。观察前缀及第一步的重放必须继续逐字节一致。

`analyze` 输出 `target_discovery`，额外比较 01 与独立重估的 00 的目标发现率、首次发现回合、`success | seen`、`success | not seen`，并精确拆成 `ΔP(success ∩ seen) + ΔP(success ∩ not seen)`；若两组的 seen/not-seen 条件概率都可识别，额外给出对称 Shapley 的**代数**分解。**看到目标是受策略干预影响的后处理量，按看到/没看到条件化不能被解释为严格因果中介效果。** 原 2×2 的 success、primary 与 reading 不使用这些辅助字段。

## E. 未来 checkpoint 的注册与重测

新增 `register-probe`，`--policy-id`、`--revision`、`--checkpoint-kind {lora,full}`、`--checkpoint-paths` 以及显式 `--approve`。注册文件存入 `data/probe_registrations/<id>.json`，不可重复覆盖，固定当时的 `lock.json`、点/臂、八个随机种子及适配器/完整模型 config、权重 SHA。每次续跑和分析重新核对哈希；权重不相符就停止。LoRA 可以单个或三个独立 seed；全权重模型一次注册**一个** checkpoint，避免悄悄切换不到实际新模型。每个 policy 的 8 次全部使用同一个冻结的点集、五臂及 `u=0..7`，不同模型不会因微调程度差而获得新取样机会。

新模型只进入次要 `armprobe` / `probe-analyze`，**从不**添加到原 E4d 00/01/10/11 主检验，不重排训练臂，不更改 `lock.json`。相同数据上多次试验需按注册 revision 报告尝试数量，以防后验挑模型。

## F. 目前不实施的可选 V3 对照

约 40 个 V3 状态的相同四格实验可以对抗“Z3 按全失败选择导致 `success(10)` 天然偏低”的批评。这是**不同训练人群上的新增主文对照**；为遵守研究者的“仅改次要与日志、原主判据不变”约束，当前 branch 只在筛选清单中继续报告 V3/E3 数量，不启动额外 V3 π₁ 续跑。若决定实施，应另行预注册样本、预算和统计读法，不能根据 Z3 的 π₁ 效果决定要不要加 V3。

## G. 前置条件与冻结纪律

原 E4d 主预注册保持只读；原 E4b/E4c、E5-L 等数据及代码不得修改。CPU 单测只核实现纯函数和模拟桥接；服务器真实 vLLM/ALFWorld 的完全一致性还需通过 E4d 的 `candidates` / `screen --limit 20` 门。日志中的二次分析不能改变成功率，主评判读法只取原始四格结果。
