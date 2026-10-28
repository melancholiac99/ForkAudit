# 三域形成机制：五臂信息补齐与信息分配

2026-10-28，冻结后 exploratory。原 Phase 3 标签、数据、主结果完全保留。

## 科学问题

三域首先是当前策略与候选集下的操作性分类。本实验研究：改变分支获得的同一项真实任务信息，
能否提高整体成功机会，同时减小当前动作造成的长期差异。它检验 Fork 的一种形成机制，
不把所有 Dead 都理解成不可恢复，也不声称所有 Fork 都来自信息。

令五臂均匀参考下 `mu = mean_a(q_a)`，`C = Var_a(q_a)`。原始条件为 `q0`，补齐后为 `q1`，
`d = q1-q0`。恒等式为：

```
C1 - C0 = 2 Cov_a(q0, d) + Var_a(d)
```

这个恒等式不是新理论。新的经验问题是：信息增益是否集中在原来较弱的动作，
使负协方差项足够大，从而在 mu 提高时 C 下降。单纯提高平均成功率并不保证 C 下降。

## 样本与固定预算

- 24 个不同任务，原 Fork / Indifferent / 全失败 Dead 各 8 个。
- 16 点来自已经检查过的两臂机制样本。选点仅按旧域、任务类型与固定哈希，未按信息干预增益筛选。
- 8 个新增 Dead 点从旧标签、全历史未见目标的 peek 和真实位置事实筛选；最多第 40 步，留至少 10 步。
- 每点保留原 factual + 4 alternatives，全部在 native / completed 条件各续跑 16 次。
- 总计 3840 条；复用原两臂实验 1024 条，新增 **2816 条**，另 10 条 GPU 管道 smoke。
- 依据上一轮实际吞吐，预计约 1–1.5 小时 GPU；不训练，不启动 WebShop，不自动追加样本。
- 任务/候选/线索与主次分析已在 `design.json` 固定并哈希；具体点见 `design.md`。

## 干预与对照

线索来自实际 peek 文本，例如“手机在床上”。在全部五条分支中验证它仍为真。
completed 仅在执行原动作后追加一次相同线索，随后按原两轮记忆自然移出。
位置、持物、容器、合法动作、总步数、Qwen2.5-7B、T=0.7 均遵循原协议。

实际执行的条件只有两个：native / completed。原两臂结果复制到新目录时明确记录源文件、源哈希、
原协议、原种子与对应关系；不伪装成新独立实验。其他臂使用新种子命名空间 91e12。

**主分析：** 在 8 个原 Fork 上，五臂 C 是否下降，同时报告 mu、每个动作的增益与负结果。
另外两个旧域及全部样本完整报告。二次矩用不同 repeat 的 U-statistic，避免噪声制造方差
或“低基线高增益”的假相关；不把负的无偏估计截成零。区间按不同任务重采样。

**辅助分析：信息如何分配。** 用上述已实测的各臂两条件结果，计算：

1. native：不给额外线索。
2. broadcast：所有动作都得到线索。
3. selective：只在预先按元数据固定的一个未显示目标实例的动作后给线索。
4. lottery20：与动作无关，以 0.2 概率给线索。

selective 与 lottery20 在五臂均匀参考下都有 1/5 的期望注入频率。后者的动作值为
`0.8*q0 + 0.2*q1`；前者替换固定一个动作的 q。它们是从实测单元识别出的反馈协议，
不是又跑出的两个独立实验。信息可用时，选择性分配会造成动作差异的一部分属于数学上的预期；
不要把这个代数事实当成新发现。重点是实际信息能否兑现，以及原 Fork 在共同补齐后还剩多少差异。

## 三域的使用边界

旧域用于固定分层，永不覆盖。另用本轮每臂 16 次、同预算的 G 置换检验展示结局结构：
全失败 / 全成功 / 非常量且未检出 Fork / 检出 Fork。它与原双池标签协议不同，清楚标为描述性。
不能把不显著当成零效应，也不能仅凭标签变化宣称跨域机制成立。

原 `target_revealed` OR bit 在所有样本 native 与 completed 中均为 1；它不直接表示各臂是否共享信息。
若完整动作结构随补齐而变化，这说明这个有效预测特征不是机制的完整描述。
既报告分支均衡，也报告仍然失败的分支与方差上升案例，不将方差降幅叫作已识别的中介比例。

## 执行

```bash
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/regime_mechanism_2026_10_28/test_metrics.py
bash exploratory_deep_discovery/regime_mechanism_2026_10_28/run.sh smoke
REGIME_WORKERS=8 REGIME_CONCURRENCY=256 bash exploratory_deep_discovery/regime_mechanism_2026_10_28/run.sh full
```

CPU preflight 见 `design.preflight.json`；输入与设计哈希在 `design.json`。断点续跑保留已完成记录，
更改运行代码或配置会拒绝混写。完整结果生成 `runs/main_24x16/summary.*`、逐臂表、图和原始续跑。
部分结果仅写 `partial_summary.*`，不用于选择扩量或挑选成功点。

`analyze_existing_pairs.py` 是先行低成本分析：复用旧 2048 条结果，输出两臂 C 的变化与方差分解，
不能把它当成五臂三域转换证据。

## 分析与交付文件

正式运行结束后，`run.sh` 自动执行 `analyze.py`。其余分析命令：

```bash
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/regime_mechanism_2026_10_28/behavior.py exploratory_deep_discovery/regime_mechanism_2026_10_28/runs/main_24x16
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/regime_mechanism_2026_10_28/verify_run.py exploratory_deep_discovery/regime_mechanism_2026_10_28/runs/main_24x16
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/regime_mechanism_2026_10_28/mechanism_summary.py exploratory_deep_discovery/regime_mechanism_2026_10_28/runs/main_24x16
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/regime_mechanism_2026_10_28/figure_mechanism.py exploratory_deep_discovery/regime_mechanism_2026_10_28/runs/main_24x16
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/regime_mechanism_2026_10_28/export_analysis.py exploratory_deep_discovery/regime_mechanism_2026_10_28/runs/main_24x16
```

- `behavior.py` 区分走向目标、尝试拾取、实际拾取与成功，不把动作意图当成环境结果。
- `mechanism_summary.py` 是看过前五个完整案例后新增的事后分析，比较线索原本可见/不可见的分支，
  并完整展示持物约束共享/局部/不存在的元数据分组。它不是新的随机干预。
- `inspect_affordance.py` 与 `scripted_recoverability.py` 只用 CPU 环境，后者是两个任务的合法完成路径，
  不评估修复后的 LLM 成功率。
- `results/` 是完整运行结束后导出的轻量报告、表、图和七条事后选取的原始案例；
  `bundle_manifest.json` 记录来源与哈希。全部 3840 条原始续跑保存在 `runs/main_24x16/results/`。
- `next_decisive_experiment.md` 是下一轮小型因子实验草案，未启动。
- 主报告：`reports/regime_mechanism_2026-10-28.md`。
