# 2026-11-06 补充修订：Token-slot 不劣于、动作探针伪影、数学续写保真

**类型：前瞻性统计/解读修订；不重打分，不修改任何正在运行的评分脚本。**
基于当前分支实现 `exp5_token_slot.py`、`token_slot_backend.py`、`token_slot_analysis.py`、`exp6_math_slot.py` 和 `math_slot_core.py` 逐项核对。两个实验继续按照冻结的原执行协议运行；新增的补充分析在 **full 记录与原核验完成后** 从原始文件只读重算。修订必须先冻结，**不得在看到全量结果后反向挑选条件/阈值**。

**请服务器 Agent：** 不要在现有运行期间修改 `PROTOCOL_NEXT_TOKEN_SLOT_AND_MATH_20261106.md`、`TOKEN_SLOT_EXECUTION_20261106.txt`、`MATH_SLOT_EXECUTION_20261106.txt`、`exp5_token_slot.py`、`exp6_math_slot.py`、`token_slot_analysis.py`、`test_token_slot.py` 等参与 source fingerprint 的文件。独立补充分析脚本 `supplement_slot_analysis_20261106.py` 不在原指纹里。不得重写原 `analysis.json` / `verification.json`；原先的严格 gate 原样保留并明确标记为 **legacy gate**。

## 1. 修订原生 Token 的保真度不劣于判停

当前 `token_slot_analysis.calculate`：
- 用 `all_think`（包含 `literal_action_mention` 等决策点）计算 T2−T1 的 KL，**不是协议写的普通位置**；
- 要求两种 KL 的 paired CI **上界 ≤ 0**，相当于要求有显著“更小 KL”的证据，不是“基本不劣于”。

**新增 supplement gate（只读取既存记录）：**
- **普通位置定义**：`kind == "think"` 且 `reasons` 至少有一个 `ordinary_pre / ordinary_post / ordinary_no_k`，并排除有 `literal_action_mention`、`action_first`、`action_discriminating` 归属的同一位置。其他位置仍完整保留在原结果/辅助报告中。
- 两种 KL 分别计算 `D(T||p_AF)` 与 `D(p_AF||T)`；以同一点内普通位置均值为单位，再按原轨迹 cluster bootstrap 4,000 次，seed 20261106。
- 事先规定的不劣于界值：**允许 T2 的 KL 比 T1 最多高 T1 KL 的 10%**。用配对量 `mean(KL_T2 - 1.10 * KL_T1)` 检验，其轨迹聚类 95% CI **上界 ≤ 0** 才通过。两种方向都要通过。另报告原始 `KL_T2-KL_T1` 的估计和 CI、T1 自身 KL 的值，不能仅报告通过与否。这是工程容忍阈值，不代表训练已经有效。
- Student 原文下一个 token 的 `log T2(y_t)-log T1(y_t)` 仍按普通位置评估，配对 95% CI **下界 ≥ −0.1 nat/token**。
- 训练候选原主门槛不变：**k 前**动作探针 T2−T1 ≥3 pp、CI 下界 >0，至少 30 个点和 10 个轨迹 cluster。必须同时满足上述 KL、logp guard。若不存在足够的普通位置，则报告 **inconclusive / insufficient support**，不得默认通过。
- **同时报告 legacy gate 和 supplement gate**；不要回填覆盖 legacy 的预先冻结输出。每个 guard 都报告有效点/位置/轨迹数，no-k 不偷换成 k 前。

## 2. 动作探针的外部槽位逐 Token 重新开轮伪影

当前 `exp5_token_slot.py` 在每个候选动作 token 处调用 `engine.quartet(h+ids[:j])`；其中 **E 槽位**每次把 `h+已写候选 token` 放进 User 引用，新 Assistant 又在开头预测下一个 token。因而完整动作的 T3/外部 F−M 斜率中可能包含共享前缀、结束标签、EOS 等非自然续写贡献。这不是数值 bug，而是目标定义与连续生成不同。

**新增预先登记的动作探针敏感性分析（只读每臂 `ids`、`tokens[*].target_logp`、`raw_logp`）：**
- 在同一探针的 **五个候选 token 序列**上计算全局最长公共前缀 `LCP` 和不与其重叠的最长公共后缀 `LCS`；`[LCP, len(ids)-LCS)` 是可区分的最小中间 token span。
- 分别用原完整动作序列（主结果不变）和该 **discriminating span-only** 的 token logprob 求五臂 softmax，按冻结 q3456 得到离线价值；同时报告 T2−T1、T3、外部 F−M 值差 / Q 对齐斜率的变化。按点内平均、轨迹 bootstrap CI，分 all / pre-k / at-or-after-k / no-k。
- 记录每点消除多少 LCP/LCS token、span 为空的比例，和结果在过滤前后是否改变符号/量级。
- **严格限定解释**：这只是敏感性诊断，**不等于真正的连续动作概率**，不能消除剩余 token 的 E 槽位逐 token 开新轮次问题。主 T2 的外部证据差也可能受影响，不能事先声称它“基本不受影响”；靠敏感性结果判断。
- 本次外部化设置是 **中性引用 + 从断点续写**，不是上一轮的 QS/QO；**禁止直接把最后一个边界的 T3 探针值与历史 QO 的 61.0 pp 当同一个实验条件比较**。

## 3. 数学实验必须补“是否真的接着写”

当前 `exp6_math_slot.py` 原样保存各条件的 `generation.text`、生成 token 数以及每题 `cut_annotations.json`；原分析只报告 correctness 和 retention，缺少对 QUOTE-CONTINUE 的续写行为核查。

从同一个已固定的原始回复中，**只读计算以下预注册的二级指标**：
- `restart_opener`：回复开头匹配一组固定的重启词/模式（如 `Let's`、`First`、`We need`、`To solve`、`Given`、`Step 1` 等；不得把它当可靠分类器）；
- `prefix_repetition`：QUOTE-CONTINUE 开头是否复写了原 Student 前缀的长片段（建议标准化空白后取起始约 80 字符作保守的字面匹配）；
- `length_ratio`：QUOTE-CONTINUE 的新增 token 数 / 同题 ACTIVE 的新增 token 数；分母为零时单独标注；
- `carryover_hint`：前缀尾部出现过的非平凡数值/中间式，是否在新回复开头出现；该启发式 **不是“已正确续写”的证明**；
- 预先固定题目 ID 的盲审样本：展示前缀结尾 + ACTIVE/QUOTE-CONTINUE 开头，**不展示正确答案和正确率**，让审阅者判断 `genuine_continuation / restart_or_rewrite / ambiguous`，并记录审阅覆盖率。可以增加所有疑似重启项目作描述性审查，但不得用效果选择样本。

将上述续写标记与各条件正确率**分开**报告；“QUOTE-CONTINUE 比 ACTIVE 准确”但大多重开推理，只能支持 **外部化有助于重解**，不能支持 **在原学生前缀上的自然恢复**。若需要分层报告正确率，标明是按实验后行为筛选的 **descriptive association**，不是有效的随机化因果估计。

## 4. 操作要求与修订边界

1. 这份文件与新增 `supplement_slot_analysis_20261106.py` 独立提交到 `research/hindsight-judges`；**原实验不暂停、不改代码、不更改旧 protocol/test 指纹、不重打分**。
2. 仅在 `token_slot_20261106/{7b,32b}/full` 的 `run.json` 已 complete、`verification.json` 已通过，及 `math_slot_20261106/full/verification.json` 已通过后生成正式补充报告。允许先用 pilot 测试脚本，但 pilot 不能用于科学判停。
3. 补充输出新建独立目录，例如 `hindsight_judges/results/next_slot_supplement_20261106/`。写出 `token_{model}_{cohort}.json`、`math_{stage}.json` 和无结果标签的审阅样本，不写回原始目录；包括原 `manifest.json` 和 `analysis.json` SHA256 及新增脚本 SHA256。
4. 不增加 GPU、训练或新采样；若某个补充指标不能从既存数据精确定义，标记 unavailable 并注明原因，**不得悄悄补造一个等价指标**。
5. 保留原样本数、记录所有过滤位置和不可用样本。任何后续“是否进入训练”的决定必须 **并列展示原 legacy 判停与本次 supplement 判停**，避免事后重新命名成功。

**研究定位不变：** 当前结果仍是离线监督诊断；只有在学生测试时拿不到 F 的情况下做训练并提高 held-out 表现，才能证明新的 OPD 方法收益。
