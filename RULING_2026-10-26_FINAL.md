# RULING 2026-10-26 FINAL — 目标、特征、预测、判定、执行序（此后不再改目标）

> 研究者发出，服务器端 Claude 执行。本文件**取代** RULING 10-23 §1–§2、RULING 10-25 §2–§5 中与之冲突的内容，合并三轮讨论（D-031、D-032、两位合作者意见、10-25/26 对话）中采纳的修改。改入 `CLAUDE.md` §1/§4/§6/§8，单独提交 `CLAUDE.md: FINAL ruling 2026-10-26`。
>
> **冻结纪律：** 本文件提交 + `FEATURES_FROZEN.md` 提交之后，目标定义、特征清单、判定规则不再变动。唯二例外：(a) 本文件已预先写明的机械切换（§2.3 gold 触发）；(b) `exploratory_` 前缀的追加，单独报告。研究者本人也受此约束。
>
> 依据：所有数字来自标签（D-035、D-039、D-040）与 CWGT 校准（D-023）。未计算任何特征–结局关联；`FEATURES_FROZEN.md` 未提交。

---

## 0. 已过的门（不再重复验证）

| 门 | 要求 | 实测 | 来源 |
|---|---|---|---|
| 深度不变性（自有数据） | d=3→6 符号一致率变化 < 0.05 | 0.593 → 0.600（+0.007）；Pearson 0.186 → 0.314 | D-035 |
| B1 标签信度 | 两库一致 ≥ 0.85 | 0.902 | D-031 |
| Pivotal 集大小 | ≥ 150 | 229（Newcombe 90%，库 3–6） | D-035 |
| Indifferent 非策略采样产物 | a′_u 不显著高于策略替代 | 0.110 vs 0.098（null 0.020） | D-039 |
| D-003 口径 | 保留−排除 conf 偏移为负 | −0.494 [−0.574, −0.414]（CWGT −0.70） | D-040 |
| HF–vLLM 秩序 | 中位 Spearman > 0.95 | 0.997 | D-040 |
| 打分后端 | — | **HF fp32 shared-prefix 为主**，vLLM bf16 为敏感性 | D-041 |

未过/待做的门：ρ_t 的 T3 ∈ [0.647, 0.793]（Phase 3）；gold 上 pivotal 规则精确率 ≥ 0.7（§2.3）。

---

## 1. 研究问题（替换 CLAUDE.md §1）

当 LLM Agent 的前向决策被压缩为日志中的已选动作时，模型侧的决策结构（候选分布、隐状态）**可由日志与 θ_old 重算但未被事后估计器利用**，世界侧对未选动作的回应**不可恢复**。本研究以执行回放为参考，先回答信用本身的结构：决策点分为 Dead（结局已定，51%）、Indifferent（结局随机但与本步动作无关，41%）、Pivotal（本步真改变结局期望，7.5%）；Dead|Live 边界 3 次/臂即可测，Indifferent|Pivotal 边界需 ≥12 次/臂，Indifferent 内的信用符号在任何深度都不可识别。然后回答：**模型侧结构、兄弟轨迹、世界侧一步 peek 各能以多少 rollout 等价成本定位每一条边界**，并据此给出 fork 预算分配规则。

本阶段不训练策略，不提新算法。产出是带对照、带 CI、按成本计价的测量，与预注册判定。

---

## 2. 目标定义（替换 §6.0 / §6.3）

### 2.1 集合与标签

- **D**：cf_defined（库 1），n = 3,038。
- **Dead / Live(𝓛)**：库 1+2 合并 30 次续跑池常量 / 非常量。𝓛 n = 1,478。
- **Pivotal_agg（主）**：𝓛 内，库 3–6 的 factual(12) vs 替代池(48) Newcombe 90% 区间不含 0。n = 229。敏感性规则：|A_3456| ≥ 0.25 且 A_34、A_56 同号非零（n = 193，重叠 187）。
- **Pivotal_het（共同报告，新增）**：𝓛 内，五臂（factual 12、替代 j 各 12）似然比 G 检验，p 值由 60 个结局在五臂间的 10,000 次置换给出（条件于边际），α = 0.10。报告：n、与 Pivotal_agg 的 2×2 重叠、**被掩盖的分叉** = het ∧ ¬agg 的比例及其 cluster CI。
- **Indifferent**：𝓛 \ Pivotal_agg。
- **Pair-level C_j（新增）**：每 (点, 替代 j)，factual 12 vs 替代 j 12 的 Newcombe 90%；sig_j = 区间不含 0。
- **参考依赖 T-ref（新增，报告项）**：𝓛 内 ≥2 个 sig_j 的点中，sig_j 符号不一致的比例，cluster CI；另报 Pivotal_agg 在 a′_u 替代下的翻转率（已有 D-039 的 100 点，若 gold 覆盖更多则更新）。
- **零质量（报告项，措辞受限）**：混合拟合 p ∈ [0.20, 0.25]（q ∈ [0.90, 1.00]）。论文只许写"在 12 次/臂下无可检出效应的 live 点比例 ≥ 75%；真实零效应比例不可点识别，由深度曲线给出区间"。**禁止**写"41% 的点是零效应"。

### 2.2 三条边界 + 一条报告行

| 边界 | 集合 | 标签 | 回放天花板（已测） |
|---|---|---|---|
| **B1** Dead\|Live | D | Live | 0.93（库 2 池方差，3/臂，D-035） |
| **B2** Indifferent\|Pivotal | 𝓛 | Pivotal_agg（主）；Pivotal_het（共同报告） | 0.66（库 1+2 \|A\|，6/臂） |
| **B3** 符号 | Pivotal_agg | sign(A_3456) | 0.84（库 1+2 A，6/臂） |
| L2-sign | 𝓛 \ {A=0} | sign(A_3456) | 0.64 — **只报告** |

每条边界、每个臂报告：AUROC + 按轨迹 cluster bootstrap 95% CI；分层 shuffle 对照（B2/B3 按轨迹内进度三分位分层，B1 不分层；1,000 次）；对 V、对臂 A 的 paired 差值 CI；**归一化效应** = (AUROC − 0.5)/(天花板 − 0.5)，天花板为同边界同集合的库 1+2 回放（m=6；B1 用 m=3）；**REV** = 使 B_m（库 1+2 子抽样 200 次均值）AUROC ≥ 该臂的最小 m ∈ {1,2,3,6}。

**过线** := 下界 > shuffle 97.5% 分位 ∧ paired-vs-V CI 不含 0 ∧ paired-vs-A CI 不含 0。

### 2.3 Gold 集与机械切换

300 点，每臂补至 32 次（库 7–13 种子格式，bank·10¹¹）。分两层：
- **G-unbiased（200 点）**：𝓛 内按 `n_draws` 三分位 × 进度三分位 × 任务类型分层随机抽（`n_draws` 为草稿采样器输出，forward-safe）。用途：三域比例的无偏重估；B3 高信度子集；pivotal 规则召回率。
- **G-candidate（100 点）**：从 |A_3456| ≥ 0.2 的 𝓛 点中随机抽。用途：pivotal 规则精确率。**不用于任何比例估计。**

**机械切换（预注册）**：若 G-candidate 上 Pivotal_agg 规则的精确率（32/臂下 90% 区间仍不含 0 的比例）< 0.70，则全文 Pivotal_agg 改用敏感性规则，写 DECISIONS，不另行裁决。≈ 7 h GPU。

---

## 3. 臂与特征（替换 §4 中的增补部分；原 19 项不动）

### 3.1 信息阶梯

| 级 | 臂 | 访问模型 | 状态 |
|---|---|---|---|
| R1 日志 | A（−conf）、H1（ρ_t） | 仅日志 | 原始分数已算（fp32） |
| R2 +重算 | B（19 特征五家族）、C（两位置 × 3 层）、V | 日志 + θ_old | 原始分数与隐状态已算 |
| R3 +兄弟 | G | GRPO 组内免费 | **待跑**（≈20 min） |
| R4 +世界一步 | W1-pure、W1-model | 执行时访问环境 | peek 文本已有；peek 态隐状态与 W1-J 待跑 |
| R5 特权 | W0 事实集 | 特权 | 只作诊断天花板 |
| R6 预算回放 | B_m | 回放 | 已算 |

### 3.2 新增特征（编号接 §4 的 19）

**控制与视界**
20. `budget_left` = 50 − turn（控制列；交互基底）

**W1-pure（纯世界文本，forward）**
21. `w1_nothing_frac` = 替代 peek 中观察为 "Nothing happens." 的比例
22. `w1_obs_len_sd` = 五个 peek 观察长度的标准差
23. `w1_adm_delta_sd` = 五个 peek 后合法集大小变化的标准差
24. `w1_task_sim_max` / 25. `w1_task_sim_mean` = peek 观察与任务描述的 bge-small 余弦相似度的最大/均值
26. **`target_revealed`** = 1[任一 peek 观察含任务描述中的目标物体词（从 "put a X in Y" / "find X" 等模板按正则抽取 X；抽不出则 NaN，计数报告）]
27. `w1_div_emb` = 五个 peek 观察嵌入的平均两两余弦距离
28. `w1_div_n_distinct` = 五个 peek 观察中不同字符串的个数

**W1-model（模型处理过的世界信息，forward）**
29. `w1_V_sd` = V probe 在五个 peek 态隐状态上的预测的标准差；30. `w1_V_chosen_minus_mean`
31. `w1_P_entropy_mean` = 策略在五个 peek 态的 entropy_q 均值
32. `w1_J_progress_frac` / 33. `w1_J_regress_frac` = 冻结 7B 对每个 peek 观察判 {进展, 中性, 倒退} 的比例（固定 prompt，温度 0，写入冻结文件）

**交互（forward）**
34. `w1_div_emb × budget_left`；35. `w1_task_sim_max × budget_left`；36. `target_revealed × budget_left`

**G（兄弟，forward 对 GRPO 而言）**
37. `g_cover` = 在同任务 7 条兄弟中观察文本精确匹配当前状态的次数
38. `g_succ_state` = 匹配状态上兄弟的成功率（无匹配 NaN）
39. `g_succ_alt_sd` = 匹配状态上各 a′ 兄弟成功率的标准差（≥2 个不同动作才定义）
40. `g_singleton` = 1[g_cover == 0]

**家族**：F6 World-pure {21–28}；F7 World-model {29–33}；F8 Interaction {34–36}；F9 Siblings {37–40}。指认规则与 §4 相同。

### 3.3 预注册预测（冻结文件中写明，Phase 3 逐条对账）

- **P1**：`entropy_q` 在 B1 上过线，在 B2 上归一化效应 CI 含 0。（"熵找得到活状态找不到分叉"。）
- **P2**：`target_revealed` 单特征在 B2 上的归一化效应 ≥ 全部 W1-pure 多特征的 50%。（ALFWorld 分叉主要是信息揭示。）
- **P3**：W1-pure 多特征在 B2 上的归一化效应 ≥ W1-model 多特征的 50%。（世界的回应不需要模型解读就有一半的用。）
- **P4**：`budget_left` 交互项 leave-out 使 W1 家族在 B2 上的 AUROC 下降，差值 CI 不含 0。（视界约束是真的。）
- **P5**：被掩盖的分叉比例（het ∧ ¬agg）/ |Pivotal_het| ≥ 0.15。

预测错了照写，不改特征。

### 3.4 成本列

每臂的成本向量 (环境步, 模型前向 token, LLM 调用数, 回放续跑数)，写入 `audit_metadata`，E1 用。

---

## 4. 判定（替换 §8）

**门**：复现门（臂 A 在库 1 A≠0 标签上 13.1% 路由召回 CI 含 13.1%；ρ_t T3 相容）；标签门（§0 已过 + gold 精确率）。不过则不判定。

| 读法 | 条件 | 主张 |
|---|---|---|
| **R-LADDER** | 模型侧（B 或 C）过 B1；W1-pure 过 B2；B3 无廉价臂过或仅 W1 过 | 模型定位活状态，世界定位分叉，符号要回放 |
| **R-MODEL** | 模型侧过 B1 且过 B2 | 决策结构能定位分叉；世界侧非必需 |
| **R-WORLD** | 只有 W1（pure 或 model）过 B2 | 世界侧是分叉的唯一廉价来源；若只有 W1-model 过，标 "(model)" |
| **R-REPLAY-ONLY** | 无廉价臂过 B2 | 分叉必须执行才能找到；E1 只按 B1 路由 |
| **R-NULL** | 无臂在 B1 上过线 | 步级信用 = 状态价值 + 噪声 |

**后缀**：若某读法仅在 Pivotal_het 上成立而 Pivotal_agg 不成立，读法加 "(het)"，论文必须说明两个估计量的差别。

每个读法附：三域表、深度曲线（自有 + CWGT）、T-ref、REV 表、E1 曲线、P1–P5 对账。全部可成文。

---

## 5. E1：fork 预算分配（零 GPU，§6.6）

每轨迹选 k ∈ {1,2,4,8} 个点。选择器（只用 forward 量）：uniform、random、entropy-top-k、B-top-k（P(Live) out-of-fold）、B→W1-pure（B 选 Live 再 W1-pure 排 Pivotal 候选）、oracle（gold/库 3–6 标签）。评估：选中集 Pivotal_agg 与 Pivotal_het 覆盖率；选中集上用库 1+2 m=3 估 A 与库 3–6 A 的 Pearson；每轨迹捕获的 |A| 总量 / k。轨迹 paired bootstrap。成本横轴用 §3.4 的向量。与 CRR 的 ≤4 fork 直接对话。

---

## 6. 执行序

| 步 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 6.1 | 改 CLAUDE.md §1/§4/§6/§8，提交 | — | 预注册修订 |
| 6.2 | **CPU 今天**：Pivotal_het 计数、2×2 重叠、被掩盖分叉比例；pair-level C_j；T-ref；三域按成败 × 进度交叉表 | CPU | `eval/het_fork.json`、`eval/pairs.json`，D-042 |
| 6.3 | GPU 队列：gold（≈7 h）→ 兄弟 G（≈20 min）→ peek 态隐状态 for W1-V/P（≈1 h）→ W1-J 调用（≈3 h） | GPU ≈ 11 h | `data/replay_gold/`、`data/siblings/`、`hidden_peek/`、`data/w1_judge.parquet` |
| 6.4 | gold 完成 → 机械切换判断 → DECISIONS | CPU | — |
| 6.5 | 起草 `FEATURES_FROZEN.md`（§3 全部 + 原 19 + 先验符号 + P1–P5 + E1 定义 + 成本向量定义 + W1-J prompt 原文），**贴给研究者过目后提交** | — | 冻结 |
| 6.6 | Phase 2：从 fp32 分数算 B 特征与 A/H1；C/V probe；W1 特征；G 特征 | CPU + 少量 GPU | `features_*.parquet` |
| 6.7 | Phase 3：B1/B2(agg,het)/B3/L2-sign × 全部臂；归一化效应；REV；E1；P1–P5 对账；判定 | CPU | `RESULTS.md` |
| 6.8 | **并行开始**：论文 §3（三域、深度曲线、T-ref、零质量界、a′_u）——数据已终版 | — | 草稿 |
| 6.9 | WebShop：50–100 条、库 1–2 全 D、库 3–6 仅 𝓛、peek；复制目标 = **P2 的机制是否成立**，不只是 AUROC 表 | GPU 数天 | 第二环境 |

---

## 7. 相关工作定位句（写作时照用，每句一篇）

- **CWGT**：协议与 chance 结论；其 pivotal 标签 = 我们的 Live，chance 来自其中约 80% 的 Indifferent。
- **CARL**：熵在分布层面区分 critical 状态 = B1；我们到步级并进到 B2（预测 P1 直接对账）。
- **Critical-State RL**（2609.24985）：嵌套采样分离动作方差与续跑噪声 = 我们的 Pivotal_het 估计量，必须明写采用；差别：他们的候选与局部奖励由任务给定（BFCL），我们的替代由策略采样、只有终局结局；他们**执行**找 B2，我们问**不执行**能否找到。
- **CRR**（NeurIPS 2026）：fork ≤4 点作 advantage；E1 回答该 fork 哪几个。
- **TreeHCA**：执行全部候选用于探索；W1 是一步版本用于信用并对照回放。
- **Doomed from the Start**：隐状态 probe 早期预测失败；我们的 Dead 池 1,516/1,560 全失败，B1 ≈ "doomed detection"，臂 C 在 B1 上的结果与之直接可比。
- 综述 2604.09459 命题 2：符号不可从轨迹识别——Indifferent 域是其经验形态。
- BiPACE / GiGPO / GraphHCA：R3。

**不能说**：首次执行候选（TreeHCA）；首次熵选 fork（CARL/CRR）；首次分离动作方差与续跑噪声（Critical-State RL）；记录即增益（可重算）；"41% 零效应"。

---

## 8. 信心

| 判断 | 信心 |
|---|---|
| 三域结构与深度不变性成立（已在两套数据上复现） | 90% |
| 模型侧过 B1 | 65% |
| 某廉价臂过 B2 | 45%（`target_revealed` 最可能） |
| P1 成立 | 60% |
| P2 成立 | 50% |
| R-LADDER/R-MODEL + WebShop + E1 → 顶会主会 | 35–40% |
| R-REPLAY-ONLY 版本 → 顶会主会 | 15–20%；TMLR/D&B 更稳 |

---

## 9. 给服务器 Claude 的直接指令

1. 读本文件；改 CLAUDE.md，提交。
2. 跑 6.2，写 D-042。
3. 启动 6.3 GPU 队列，gold 的两层抽样种子记录；W1-J 的 prompt 先写进 DECISIONS 草稿等研究者确认。
4. 起草 `FEATURES_FROZEN.md`，**不提交**，贴给研究者。
5. 此后任何"要不要改目标"的想法，写进 `proposals/` 作为下一篇论文的材料，不改本文。
