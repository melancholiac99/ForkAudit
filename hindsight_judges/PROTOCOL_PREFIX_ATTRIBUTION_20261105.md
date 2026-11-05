# Prefix Attribution × Message-Slot Diagnostic — Revised Protocol (2026-11-05)

**状态：正式实验方案修订版；尚未运行 GPU。** 本文件替代 2026-11-05 的旧版六条件协议。需要先同步更新现有 `exp4_prefix_attribution.py` / `prefix_attribution_analysis.py` / 测试（实现提交 `efea3bd` 仍是六条件版本），**不要直接拿未修改的旧实现运行本协议**。

**GitHub 分支：** `research/hindsight-judges`。**不修改 main，不覆盖 E1/E1c/P4/E4 或现有动作/逐 token 结果；不启动训练或新 rollout。**

## 1. 研究问题

我们已观察到：相同 Teacher、相同五臂一步反馈 F，P4 上 32B 的五候选独立回放加权价值从 **Keep 42.2%** 升至 **Drop 67.8%**。本实验要进一步识别，这个落差主要和哪一种因素有关：

- **推理内容**：文本明确承诺某个动作，导致自然后续偏向这个动作。
- **消息槽位 / 轮次**：Student 的文字位于 Teacher 当前未结束的 Assistant 回复、已结束的 Assistant 回复，还是外部 User 引用。
- **来源归属描述**：相同引用被标注为“你写的”还是“另一 Agent 写的”。

**核心假说是“Student 的错误在 Assistant 槽位被当作已有自我推理，从而妨碍纠错”；这是待检验假说，不预设会成立。** 只改归属标签不等同于改变实际消息槽位。研究对象首先是**动作评分与 F 的证据响应**，不声称测到了 OPD 逐 token 梯度或训练后的 Agent 成功率。

## 2. 数据、模型和不变量

- 直接复用冻结的 **386 个 P4 状态 / 60 条原始轨迹**；每点 Student 真正写出的原始前动作推理 R，五个固定合法动作（事实臂 f + 四个替代臂），独立 `q3456` 回放值；**不抽新数据、不重新估计 Q**。
- 两个 Teacher：**Qwen2.5-7B-Instruct** 和 **Qwen2.5-32B-Instruct**。7B 使用已验证 FP32 恢复；32B 使用已验证 FP16 transformer + FP32 head；遵循旧实验的 SDPA MATH / chunked prefill 与数值校验。
- 两种证据：**M** = 原决策时输入；**F** = 现有五动作对称的一步反馈。证据和动作候选在同一点、同模型、不同条件间严格保持不变。
- R 包括 Student 在原动作开标签前的原样文本，Think 原封不动；任何改动仅限于**放置位置及固定来源说明**。不提示“错误”“检查”“纠正”“忽略旧推理”，不透露最终正确动作；不得把 `executed_action` 或原始动作字符串随引用复制进去。
- 使用真正的 Qwen chat template 和 role token，不用文本 `[Assistant]` 假装结构化消息；保留原始动作开标签、空格与结束标签，继续使用 `exp3_action_diagnostic.py` 的候选路径评分语义。

## 3. 八种条件（六条件 + AP + QN）

| 条件 | R 放置位置 | 来源说明 | 当前动作输出位置 | 主要用途 |
|---|---|---|---|---|
| **KEEP** | **正在生成的 Assistant 前缀** | 无 | 接在原 R 后的动作标签 | 原始 OPD 式锚点 |
| **DROP** | 不提供当前 R | 无 | 原始无 Think 动作前缀 | 原始无推理锚点 |
| **AS** | 正在生成的 Assistant 前缀 | “你写的” | 接在原 R 后 | 对照 AO |
| **AO** | 正在生成的 Assistant 前缀 | “另一 Agent 写的” | 接在原 R 后 | Assistant 槽位内归属提示 |
| **QS** | 第一条 User 消息内逐字引用 | “你写的” | 新的 Assistant 从动作标签开始 | 与 AP 比消息槽位；与 QO 比来源 |
| **QO** | 同样的 User 引用 | “另一 Agent 写的” | 新的 Assistant 从动作标签开始 | User 内归属提示 |
| **QN** | 同样的 User 引用 | **中性、无作者说明** | 新的 Assistant 从动作标签开始 | 控制归属提示本身 |
| **AP** | **已结束的上一条 Assistant 消息** | 与 QS 相同：“你写的” | 新 User 请求后再开 Assistant、从动作标签开始 | 与 QS 比已结束 Assistant vs User 槽位 |

**AP 是本次修订最重要的新条件。** 它不是把 R 保留在当前正在生成的 Assistant 前缀，而是先让 Assistant 发完 R、正常以 `<|im_end|>` 结束该轮，再让 User 提出同一句动作请求，最后由新的 Assistant 输出动作。**这样 AP 与 QS 都经历新的一轮 Assistant 生成，避免旧版 `QS−AS` 把“轮次重启”混进槽位对比。** 但 AP vs QS 仍不可避免地改变了消息结构、引用格式和对话历史，所以解释为“匹配重启条件下的消息槽位效应”，不宣称绝对纯粹的角色因果识别。

### 明确的聊天结构（示意；实际以 tokenizer 的 chat template 渲染）

下述 `BASE` 是完全相同的 M/F 信息，`R` 是原样 Student 前动作推理，`TAG` 是原始动作开标签，`REQUEST` 固定为 `Output the next legal action.`。

```text
AS: [user: BASE + SELF_CUE] [assistant (OPEN): R + TAG]
AO: [user: BASE + OTHER_CUE] [assistant (OPEN): R + TAG]
KEEP: [user: BASE] [assistant (OPEN): R + TAG]
DROP: [user: BASE] [assistant (OPEN): original no-Think prefix]

QS: [user: BASE + SELF_CUE + QUOTE(R)] [user: REQUEST] [assistant (OPEN): TAG]
QO: [user: BASE + OTHER_CUE + QUOTE(R)] [user: REQUEST] [assistant (OPEN): TAG]
QN: [user: BASE + NEUTRAL_CUE + QUOTE(R)] [user: REQUEST] [assistant (OPEN): TAG]

AP: [user: BASE + SELF_CUE] [assistant (COMPLETED): R]
    [user: REQUEST] [assistant (OPEN): TAG]
```

**提示语统一修订：** 不再使用 `in an earlier attempt`。例如：

- `SELF_CUE = "[DRAFT SOURCE] The draft reasoning below for the current step was written by you. [/DRAFT SOURCE]"`
- `OTHER_CUE = "[DRAFT SOURCE] The draft reasoning below for the current step was written by another agent. [/DRAFT SOURCE]"`
- `NEUTRAL_CUE = "[DRAFT] Below is draft reasoning for the current step. [/DRAFT]"`

QS 与 AP 的 **SELF_CUE 字符串完全相同**；二者最后的 User `REQUEST` 也完全相同。QS/QO/QN 的引用边界及 R 完全相同。AS/AO 只改变来源描述。无任何纠错暗示。

**代码特别注意：** 旧实现只有一个平铺的 `prompt` 加 `reply_prefix`，无法生成真实 AP 的历史 Assistant 消息；新实现需接收结构化 `messages=[{role,content},...]` 后用 chat template 渲染。不要将 `<|im_start|>assistant` 直接塞进 User 文本伪造角色。AP 的完整 R 必须确实位于历史 Assistant 轮次，且 `<|im_end|>` 后才出现新 User 请求；不能带上事实动作。

**一个已知边界：** `s071_t01` 的 R 在 Think 外还包含旧 `[action]` 文本。原样保留、记录其存在，不为了让结果好看而清理；汇总时注明这一点即可。

## 4. 主终点与主要对照（先固定，后看结果）

对每个点 `i`、条件 `c`、证据 `e`，用**同一个 Teacher**给五条固定动作续写算原始序列对数概率 `ell[e,c,i,j]`，归一化得到五候选分布：

```text
p[e,c,i,:] = softmax(ell[e,c,i,:])
V[e,c,i]   = sum_j p[e,c,i,j] * q3456[i,j]
```

`V` 是“固定五动作概率加权的独立回放价值”，不是实测策略成功率。

**两个并列主终点：**

1. **消息槽位对照（最重要）：** `V[F,QS] − V[F,AP]`，同时报告 M 下对应差异与 `(F−M)[QS] − (F−M)[AP]`。R 原样相同；两条件都已重启 Assistant 且最终请求相同。解释上保留消息结构差异限制。
2. **引用内来源效应：** `V[F,QO] − V[F,QS]`，同时报告 M 下差异与 `(F−M)[QO] − (F−M)[QS]`。它检验的是自我/他人**来源措辞**，不是消息槽位。

**配套次级对照（都报告，不临时挑选）：**

- `AP − AS`：在 self 归属下，从当前 Assistant 延续变成结束前一轮再开始新一轮；包括重启与用户请求效应。
- `QN` 与 QS/QO：来源提示是否使相同引用产生不同响应。
- `AO − AS`：在当前 Assistant 前缀里，仅来源文字是否有效。
- `QS − AS`：旧方案的混合位置/重启对照，仅作辅助。
- `DROP − KEEP`：复现原始强效应。
- 每条件单独的 M/F 离线价值、选事实动作的比例、事实臂概率、坏事实动作纠错率，以及所有配对对照的置信区间。

**证据利用能力不能只看 V。** 用相同动作的 `M→F` 相对 Log-Odds 变化评估，而不是只看 F 下的绝对评分。对于事实臂 `f=0` 和替代臂 `j=1..4`：

```text
E[c,i,j] = [ell[F,c,i,j] - ell[F,c,i,0]]
         - [ell[M,c,i,j] - ell[M,c,i,0]]
```

`E` 与 `q3456[i,j]-q3456[i,0]` 比较，报告价值方向一致率/回归斜率。

**与先前机制 A/B 分析统一尺度：**

- 除以事实臂为参照的 E 外，再计算每个点五臂的 **去均值证据增量**，以 `Q_j - mean(Q)` 对 `E_j - mean(E)` 做合并的价值对齐斜率；在同一点配对比较各条件。
- 以相同公式计算 `slope(c)/slope(DROP)`，用**同一批轨迹的 bootstrap**得到区间，同时报告绝对斜率，避免分母接近零时误判。和先前 KEEP/DROP 的 32B 0.55、7B 0.36 参照值只在定义完全一致时直接对照。

以上保证能区分“引用只是降低原动作的强先验”与“引用恢复了 F 对更好动作的证据响应”。

## 5. 统计与结果解释

- 全量 **386 点、60 条轨迹**，原轨迹为 bootstrap cluster；**4,000 次**，seed `20261105`；所有差异按**同一点配对**后再聚类重采样。报告 95% CI、单位和绝对值；未做多重比较校正，不能把置信区间跨 0 当作“证明无影响”。
- `QS≫AP`：支持匹配重启条件下的消息槽位干预有效；不直接断言 Teacher 真的“把文本当自己写的”。
- `QO≫QS`：支持来源归属措辞有效；若在 F−M 的证据增量上也明显改善，更有理由讨论证据利用。
- `AP≈QS≫AS`：更可能是**当前未结束 Assistant 续写**造成额外约束，而非一切 Assistant 历史都会造成相同干扰。
- `AS≈AO≈AP≈QS≈QO≈QN≈KEEP` 但 `DROP` 好：更像推理文本内容/动作承诺效应；也可能是这些呈现方式没有有效改变模型行为，不能强称单一因果机制。
- 所有发现都只限**动作续写打分**；对数学推理、完整 OPD 监督分布、训练成功率的结论需要下一阶段单独验证。

## 6. 历史锚点必须重现（本次增加）

旧 `KEEP/DROP` 使用与已发布 `exp3_action_diagnostic.py` 完全相同的真实 prompt、reply prefix、五条动作续写、tokenizer、模型精度。

新评分结束后，**只读**对照历史 `results/action_diagnostic/{7b,32b}/full/scores.jsonl`：

1. 按点和 M/F × KEEP/DROP 匹配，比较五条原始 `sequence_logp` 和候选概率；
2. 报告每点概率总变差距离 TV 的均值、最大值及不一致点；
3. 重算并报告历史头条价值：32B 的 `M keep 37.6 / M drop 41.2 / F keep 42.2 / F drop 67.8`（百分点），及 7B 对应值；
4. 如出现非数值误差所能解释的差异，**暂停将新旧结果混表解读**，先查清 prompt、序列化、精度或数据版本差异；不得覆盖历史结果。

## 7. 实现、预算与交付

- 使用现有 P4 数据加载、独立 Q、动作标记解析、候选评分和缓存/直接评分的数值校验；所有新文件和评分写入独立目录 `hindsight_judges/results/prefix_attribution_v2_20261105/`。
- 修订现有实现（`exp4_prefix_attribution.py`、`prefix_attribution_analysis.py`、`test_prefix_attribution.py` 及 README），增加真实多轮模板、AP/QN、主终点、旧结果只读复现、证据斜率/DROP 比值。**该代码更新不包含在本协议提交中，旧代码在同步之前不能据此直接运行。**
- 运行前先完成 CPU 测试、真实 tokenizer 的结构检查以及候选 score/direct 数值门槛；**首先在 32–64 个点试跑测速**，正式 386 点和试跑严格分目录。测试集里需要明确覆盖 AP 的 assistant 轮次结束 token，以及 QS/QO/QN 的逐字引用一致性。
- **规模：** `386×2(M/F)×8 条件 = 6,176` 个上下文/模型，7B 与 32B 合计 **12,352** 个上下文；每上下文固定五候选动作。比此前六条件方案每模型增加 1,544 个上下文（**+33.3%**）。两张大显存卡可各负责一个模型；总耗时以 pilot 实测为准，不承诺“1–2 GPU 小时”。
- 报告逐条件价值/事实动作偏好、两条并列主终点、F−M 交互、去均值证据斜率、DROP 斜率比、轨迹聚类 CI、老锚点复现、所有源/权重/代码哈希、数值校验和排除点。
- 服务器 Agent 完成实现后再申请或按单独任务指示执行 GPU。**本次仅保存方案，不自动运行、不训练 Student、不修改 main。**

**决策原则：** 先测清楚是否真的有可重复的“消息槽位效应”或“来源提示效应”。只有当它还改善 F 的价值对齐证据响应时，才进一步讨论角色外部化的 Token-level OPD 训练方法；绝不因动作打分改善就直接宣布 OPD 去偏成功。
