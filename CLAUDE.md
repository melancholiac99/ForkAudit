# CLAUDE.md — 项目 M0：决策结构 vs 事后信用（三臂审计）

> 把本文件放在项目根目录并命名为 `CLAUDE.md`。你（服务器上的 Claude）在本项目中的全部工作都以本文件为准。本文件自包含，不依赖任何先前对话。遇到本文件没有规定的决策，先在 `DECISIONS.md` 里写下你的选择和理由再继续，不要静默决定。

---

## 1. 研究问题（FINAL RULING 2026-10-26；不要改写）

当 LLM Agent 的前向决策被压缩为日志中的已选动作时，模型侧的决策结构（候选分布、隐状态）**可由日志与 θ_old 重算但未被事后估计器利用**，世界侧对未选动作的回应**不可恢复**。本研究以执行回放为参考，先回答信用本身的结构：决策点分为 Dead（结局已定，51%）、Indifferent（结局随机但与本步动作无关，41%）、Pivotal（本步真改变结局期望，7.5%）；Dead|Live 边界 3 次/臂即可测，Indifferent|Pivotal 边界需 ≥12 次/臂，Indifferent 内的信用符号在任何深度都不可识别。然后回答：**模型侧结构、兄弟轨迹、世界侧一步 peek 各能以多少 rollout 等价成本定位每一条边界**，并据此给出 fork 预算分配规则。

本阶段不训练策略，不提新算法。产出是带对照、带 CI、按成本计价的测量，与预注册判定。

> **修订与冻结（RULING_2026-10-26_FINAL.md）：** 本节取代原三臂审计表述（RULING 10-23 §1–§2、RULING 10-25 §2–§5 中冲突的内容）。本修订提交且 `FEATURES_FROZEN.md` 提交之后，目标定义、特征清单、判定规则不再变动；唯二例外：(a) §6.3a 预先写明的 gold 机械切换；(b) `exploratory_` 前缀的追加，单独报告。研究者本人也受此约束。此后任何"要不要改目标"的想法写进 `proposals/`，作为下一篇论文的材料。相关工作定位句与"不能说的话"见 RULING_2026-10-26_FINAL.md §7。
>
> **冻结补充（RULING_2026-10-26_FREEZE_ADDENDUM.md，冻结前最后一次标签侧修正）：** B2 主标签改为 Pivotal_het；Pivotal_agg 改名 Credit-pivotal；三域改为 **Dead 51.3% / Indifferent（Live ∧ ¬het）33.0% / Fork（het）15.6%**，Fork 内分 Credit-pivotal（170）与 Masked（305）。上段中的 Indifferent 41% / Pivotal 7.5% 为旧定义，以 §6.0 为准。论文中必须写"15.6% 被检出，其中零假设下期望 3.4%"，禁止写"16% 的步是分叉"。此后即使再有同等强度的证据，也只能作 `exploratory_`。

## 2. 必须先知道的已有结果（不要重复发现）

| 工作 | arXiv | 已确立的事实 | 对本项目的作用 |
|---|---|---|---|
| **Credit Without Ground Truth (CWGT)**, Haiyue Zhang | 2608.19760 v2 | ALFWorld, Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct, T=0.7, HCAPO 模板。每个决策点：factual 动作重放 ≥3 次，从策略采样 K=4 个不同合法替代动作各续跑 ≥3 次；A_replay(t)=mean(outcome\|factual)−mean(outcome\|alt)。2,034 → 1,768 个完整 Qwen 决策点。**69.5% 的点 A_replay=0**；13.1%(Qwen)/26.8%(Llama) 的点采不出 4 个替代 → 反事实未定义，排除不插补。标量置信度（chosen action 的 token 平均 logprob）：13.1% 路由率下 pivotal 召回 11.9% [9.4,14.9]，chance level；事后 logprob 信用与流畅度秩相关 +0.752；条件掉流畅度后与因果增量偏相关 −0.004。LLM judge 与 shuffle 对照区间重叠。训练 7 臂无一胜基线。**声明公开 replay 存档（29,402 文件）、per-turn 摘要、logprob dump、生成器**。 | **协议、指标、对照、基线数字全部沿用。它是本项目的起点，不是竞争者。** 它没测任何来自未选动作的信号，没测隐状态。 |
| **BiPACE** | 2606.25556 | 用 actor 隐状态几何的 cosine 距离聚类步骤（bisimulation 代理），簇内动作条件基线估计 Q−V；ALFWorld/Qwen2.5-7B 97.1%。代码公开。 | 隐状态 → 步级信用已有训练级结果。臂 C 是对它的 replay 审计，不是复现。 |
| **Doomed from the Start** | 2607.06503 | 隐状态线性 probe 从第一轮预测 episode 失败；行为特征在其上无增益。 | 臂 C 是臂 B 的信息上界。 |
| **CARL** (Shen et al.) | 2512.04949 v3 | 搜索 QA + 数学，Qwen3-4B。关键性定义 = 把随机性隔离到该状态的动作采样时结果奖励的**标准差**（无符号）。发现 >50% 状态重采样后对结果几乎无影响；高关键性状态的**采样估计**动作熵更高（分布层面；CWGT 重分析为 Cliff's δ=0.42, n=294）。训练方法：熵作关键性代理，只更新高熵状态。 | "熵与关键性有分布层面关联"不是新结果。但 CARL 的熵是自由文本采样估计，不是合法集上的精确熵；关键性是无符号 SD，不是有符号对比；没有逐步识别指标、没有 shuffle 对照、没有流畅度检验、没有熵以外的结构、没有隐状态、不在 ALFWorld/7B。**CWGT 已明确指出 CARL 的分布层面分离与 CWGT 的逐步路由 chance level "不冲突"——这个张力正是本项目的切入点：什么结构能把分布层面的分离变成逐步可用的识别？** |
| **EMPG** | ICML 2026, PMLR 306 | 步级熵调制策略梯度。 | 熵不是新信号。 |
| **Prospective Hindsight** | 2610.02740 | 事前预测 vs 事后评价的 gap 用于训练。 | 事前预测 probe 不是本阶段主信号。 |
| **CWGT ICLR 2027 修订版** | OpenReview Me3WCeeMWW | 两库 × 16 续跑/臂 × 172 turn（43 轨迹 × 4 位置）。A16–B16 Pearson 0.736 / pooled Spearman 0.330；old3–B16 Spearman 0.063；五臂区间排除零仅 3/172。recovery 位置选择：confidence 选择器 vs 同位置 factual +8.54 pp，vs 随机位置 +3.33 pp [−2.1, 13.3] 未定。hindsight 行对决策无影响。置信度口径 = 裸 prompt teacher-force 动作命令 token（不含 think，不含标签 token）。 | 相关，不触发停止。3 次/臂不足以定义 turn 级动作贡献 → §6 的深度设计；recovery 目标 → L3；confidence ≈ implicit → 臂 H1 预期与臂 A 重合。补充材料在 `downloads/openreview_supp/`（无 replay 存档）。 |
| **RMPO** | OpenReview ac6BotAkVF | replay 测得的信用算子（删除 / 前缀进度 / 同状态替换）加进 GRPO advantage；Qwen3-4B/8B；ALFWorld 删除信用 vs 规划器进度 ρ=0.842。 | 相关，不触发停止。用 replay 做训练信号，不是前向决策结构预测器。 |
| **ACRE（审计）** | OpenReview HbNTIXwoeU / plrHvqYzhK | 审计各 pipeline 的信用观测量 vs walkthrough 进度；fidelity −0.037…0.789 与留出增益排序一致。 | 相关，不触发停止。不涉及决策结构。 |

**阅读任务（Phase 0）：** 读 CWGT 全文，特别是 Sec. 2 协议、Sec. 4 结果、Appendix E（可测性）、Appendix O（router）、**Appendix Q（并行工作 CARL/CAR/CSO/C3 —— 这部分尚未被核对，若其中有与本项目更近的工作，立即写入 `DECISIONS.md` 并停下等指示）**。

## 3. 环境锁定（不可更改，更改需写 DECISIONS.md）

- 策略：`Qwen/Qwen2.5-7B-Instruct`，fp16/bf16，采样温度 0.7，与 CWGT 一致。**禁止用 3B。** 第二家族 `meta-llama/Llama-3.1-8B-Instruct` 为可选复制。
- 环境：`alfworld==0.4.2`, `textworld==1.7.0`（CWGT 的 pin）。先 `alfworld-download` 拉游戏文件。
- Agent 模板：HCAPO (arXiv 2603.08754) 附录公开的 ALFWorld 模板。**Phase 1 必须确认模板是否把合法动作列进 prompt**，记录在 `DECISIONS.md`。
- 推理：vLLM（bf16）用于 rollout/replay。**候选动作与 chosen 变体的 teacher-forced scoring 主口径 = HF transformers fp32 + 共享前缀 KV cache**（`20_score_candidates.py --backend hfprefix`，D-036/D-041；产出 `data/scores_fp32.parquet`、`data/scores_chosen_fp32.parquet`）；vLLM bf16 打分（`data/scores.parquet`、`data/scores_chosen.parquet`）只作数值稳健性敏感性行。HF bf16 用于隐状态导出（需要 `output_hidden_states`）。两套打分秩序一致性已核对（D-033、D-040）。
- 硬件：单卡 RTX Pro 6000（按 96GB 计；若实为 48GB 版本亦足够 7B）。
- 随机性：所有采样固定 seed 并记录；每个产出文件记录生成它的脚本路径 + git commit hash。
- Python 3.10+，`numpy scipy scikit-learn pandas pyarrow torch transformers vllm`。

## 4. 三臂设计

所有臂在**同一批决策点**上评估，对**同一组**参考标签（FREEZE ADDENDUM：§6.0 的 Live / Fork(Pivotal_het) / Credit-pivotal ∩ het 上的 sign(A_3456)）。

**臂 A — 标量置信度（CWGT 复现，对照臂）**
`conf_t = mean token logπ(a_t | s_t)`。**口径（D-003，RULING 7.2）：** 原 prompt（chat template，`add_generation_prompt=True`）之后、assistant 轮起始处直接 teacher-force `<action>{a}</action>`，**只取 `{a}` 的命令 token**，标签 token 不计，不含实现的 `<think>`。敏感性行：`<think></think><action>{a}</action>`（空推理）；以实现 think 为条件的版本只报告一行（预期退化，D-020），不进判定。**主分数 s = −conf**（CWGT router 方向，D-004）；+conf 为预声明敏感性行。目标：与 CWGT 的 chance-level 路由结论相容（复现门槛见 §8）。臂 A 的 M4 报告但标注"近乎机械"，流畅度复现目标由臂 H1 承担（D-005）。`chosen_freq`（特征 18）不属于臂 A，但在 L2 的 paired 比较中作为独立基线列出。

**臂 B — 人工决策结构特征（主假设臂）**
**候选集** 𝒜_t := 合法动作集 ∪ {chosen} \ {`help`}（与 prompt 展示给策略的集合一致；D-021）。对候选集中每个动作 a 按臂 A 的同一口径做 teacher-forced scoring，得 `logp_sum(a)`, `logp_mean(a)`, `n_tokens(a)`；另在**去掉任务目标描述**的 prompt 下再 score 一次得 `logp_sum_notask(a)`。令 q_t = softmax over 𝒜_t of logp_sum。特征列表**在 `FEATURES_FROZEN.md` 中冻结并 commit 之后才允许计算任何与 A_replay 的关联**：

1. `margin_raw` = logp_sum(a₁) − logp_sum(a₂)，a₁=argmax，a₂=次大（注意：a₁ 可能 ≠ chosen；同时记录 `chosen_rank`）
2. `margin_mean` = 同上用 logp_mean
3. `margin_detask` = [logp_sum(a₁)−logp_sum_notask(a₁)] − [logp_sum(a₂)−logp_sum_notask(a₂)]
4. `entropy_q` = H(q_t)
5. `eff_size` = exp(H(q_t))
6. `chosen_prob` = q_t(chosen)
7. `runner_same_type` = 1[type(a₂)==type(chosen)]，type ∈ {goto, take, put, open, close, toggle, use, examine/look, inventory, other}，用正则从动作串解析
8. `top3_n_types` = top-3 候选的不同类型数
9. `loop_chosen` = 1[chosen(t) ∈ {chosen(t−5..t−1)}]
10. `loop_runner` = 1[a₂ ∈ {chosen(t−5..t−1)}]
11. `alt_revisit` = 1[a₂ ∈ {chosen(t+1..t+5)}]（替代重访：runner-up 后来被真的执行了）
12. `js_prev` = JS(q_t, q_{t−1})（候选集不同时按并集补零）
13. `sem_dispersion` = 用冻结句子编码器（`BAAI/bge-small-en-v1.5`）嵌入每个候选，q 加权的平均成对余弦距离
14. `sem_dist_chosen_runner` = cos_dist(emb(chosen), emb(a₂))
15. 控制列：`turn_index`, `n_admissible`, `chosen_n_tokens`, `type(chosen)` one-hot
16. `entropy_sampled` = CARL 式采样估计熵：用 replay 阶段从策略 T=0.7 拒绝采样得到的那些草稿（Branch B 最多 300 次；Branch A 若存档保留了草稿）里**合法且不重复**的动作频率算熵。零额外成本。与 `entropy_q` 并列，用于区分"精确合法集熵"与"采样估计熵"。若 Branch A 存档无草稿，则对每点额外采样 32 次，标注成本。
17. `chosen_conf` = 臂 A 的标量置信度，作为臂 B 内部的对照列，确保 B 的增益是在 A 之上。
18. `chosen_freq` = log[(c + 0.5) / (32 + 0.5·|𝒜_t|)]，c = 32 份首批草稿（D-008）中解析为 chosen 的数量。估计策略对 chosen 的**真实边际概率**（含推理），与 `chosen_conf`（无推理偏好）构成对照；`entropy_sampled` vs `entropy_q` 是同一对照的分布版本。
19. 控制列 `chosen_admissible` = 1[chosen ∈ 合法动作集]。

**家族划分（RULING §1，冻结于 `FEATURES_FROZEN.md`）：**

| 家族 | 特征 |
|---|---|
| F1 竞争/边际 | `margin_raw`, `margin_mean`, `chosen_prob`, `chosen_rank`, `chosen_freq` |
| F2 分布离散度（CARL 对照） | `entropy_q`, `eff_size`, `entropy_sampled` |
| F3 候选构成 | `runner_same_type`, `top3_n_types`, `sem_dispersion`, `sem_dist_chosen_runner` |
| F4 时间结构 | `loop_chosen`, `loop_runner`, `js_prev`；`alt_revisit`（hindsight，单列） |
| F5 任务条件化 | `margin_detask` |
| 控制列 | `turn_index`, `n_admissible`, `chosen_n_tokens`, `type(chosen)` one-hot, `chosen_conf`, `chosen_admissible` |

**指认规则：** 某家族在某条边界（B1/B2/B3，§6.3；FINAL RULING 前为 L2）上 (1) family-only 模型击败 F2 与状态价值基线 V（paired 差值 CI 不含 0），且 (2) 从全模型中 leave-one-family-out 后性能显著下降（paired 差值 CI 不含 0），才称为"那种结构"。
**先验符号（D-004）：** `FEATURES_FROZEN.md` 为每个单特征在 B1、B2、B3 上各声明一个先验符号或"无先验"（FINAL RULING；原为 L1、L2）；有先验的按先验方向报告 AUROC，无先验的报告原始 AUROC（可 < 0.5），不事后翻转。

**可得性标注（必须写入 audit_metadata 的 `availability` 列）：** 特征 1–10、12–14、16–19 只用当前及过去信息，标 `forward`；特征 11 `alt_revisit` 用到未来 5 步的实际执行，标 `hindsight`。论文中"rollout 时可记录"的主张只能基于 `forward` 特征；主表要分别报告"仅 forward"与"forward + hindsight"两个版本。

臂 B 的预测器：GroupKFold(by trajectory, 5 折) 的 L2 logistic 回归（分类）/ 岭回归（连续），只调一个正则系数，内层 CV。同时报告**单特征**版本（每个特征单独作为 score）。

**臂 C — 隐状态线性 probe（信息上界臂）**
导出两个位置的残差流，层 = {14, 21, 26}（Qwen2.5-7B 共 28 层），float16：(i) **主**：原 prompt（含 generation prompt）的最后一个 token —— 臂 B 的信息上界（q_t 与 `chosen_freq` 都是该位置状态的函数）；(ii) **次**：实现的 `<think>…</think>` 之后、`<action>` 首 token 之前（"承诺时刻状态"，同样 forward 可得）。**一次 HF 前向**：对 [prompt + 实现的 response 直到 `<action>` 前] 做前向，因果注意力下位置 (i) 不受后续 token 影响，两个位置同时取出。每层每位置一个 ridge logistic probe，GroupKFold by trajectory，内层 CV 选 λ。报告每层结果与最佳层。

**臂 H1 — HCAPO ρ_t（被审计的事后信用，RULING 7.3）**
同一 recorded action、同一命令 token 掩码，两次 teacher-forced 打分：policy pass（原 prompt）与 hindsight pass（原 prompt 末尾追加一行 `Final outcome: {SUCCESS|FAILURE}. Final observation: {obs_after_final[:500]}`）。q_t = exp(Σ logp_cmd / (5.0 · n_tokens))（hindsight pass）；ρ_t = clip(q_t / mean_{同轨迹全部已评分 turn，含 cf 未定义者}(q), 0.8, 1.2)；无平滑、无 epsilon。预期 ρ_t ≈ conf（CWGT 修订版 confidence ≈ implicit）；仍做，因为 M4 的 +0.752 目标与"事后估计器忽略"的检验需要它。LLM judge（H2）本阶段不做。

**状态价值基线 V（L2 必须击败的对手）**
输入只限前向可得量：`turn_index`、`n_admissible`、臂 C 位置 (i) 的隐状态。目标 = 该点续跑池的平均成功率（库 1 与库 2 各算一次，交叉拟合）。GroupKFold by trajectory。V 的 out-of-fold 预测在 L2 中既作独立对手，也作偏回归协变量。

**联合与分解：** B+C 联合；把 C 最佳层 probe 的逐点输出对 B 特征做线性回归，报告 R²（"C 里有多少能被 B 解释"）。所有臂的主分析在 §6.3 的三条边界上进行（FINAL RULING；原为 L2）。

### 4.1 增补臂与特征（FINAL RULING 2026-10-26 §3；原 1–19 项不动）

所有臂在**同一批决策点**上、对 §6 定义的**同一组标签**（B1/B2/B3/L2-sign）评估。

**信息阶梯：**

| 级 | 臂 | 访问模型 |
|---|---|---|
| R1 日志 | A（−conf）、H1（ρ_t） | 仅日志 |
| R2 +重算 | B（19 特征五家族）、C（两位置 × 3 层）、V | 日志 + θ_old |
| R3 +兄弟 | G | GRPO 组内免费 |
| R4 +世界一步 | W1-pure、W1-model | 执行时访问环境 |
| R5 特权 | W0 事实集 | 特权；**只作诊断天花板，永不作为方法** |
| R6 预算回放 | B_m（m ∈ {1,2,3,6}，库 1+2 子抽样） | 回放；计价轴 |

**新增特征（编号接 1–19）：**

- 控制与视界：20. `budget_left` = 50 − turn（控制列；交互基底）。
- **W1-pure（纯世界文本，forward）**：
  21. `w1_nothing_frac` = 替代 peek 中观察为 "Nothing happens." 的比例；
  22. `w1_obs_len_sd` = 五个 peek 观察长度的标准差；
  23. `w1_adm_delta_sd` = 五个 peek 后合法集大小变化的标准差；
  24. `w1_task_sim_max` / 25. `w1_task_sim_mean` = peek 观察与任务描述的 bge-small 余弦相似度的最大/均值；
  26. **`target_revealed`** = 1[任一 peek 观察含任务描述中的目标物体词（从 "put a X in Y" / "find X" 等模板按正则抽取 X；抽不出则 NaN，计数报告）]；
  27. `w1_div_emb` = 五个 peek 观察嵌入的平均两两余弦距离；
  28. `w1_div_n_distinct` = 五个 peek 观察中不同字符串的个数。
- **W1-model（模型处理过的世界信息，forward）**：
  29. `w1_V_sd` = V probe 在五个 peek 态隐状态上的预测的标准差；30. `w1_V_chosen_minus_mean`；
  31. `w1_P_entropy_mean` = 策略在五个 peek 态的 entropy_q 均值；
  32. `w1_J_progress_frac` / 33. `w1_J_regress_frac` = 冻结 7B 对每个 peek 观察判 {进展, 中性, 倒退} 的比例（固定 prompt，温度 0，prompt 原文写入冻结文件）。
- **交互（forward）**：34. `w1_div_emb × budget_left`；35. `w1_task_sim_max × budget_left`；36. `target_revealed × budget_left`。
- **G（兄弟，forward 对 GRPO 而言）**：
  37. `g_cover` = 在同任务 7 条兄弟中观察文本精确匹配当前状态的次数；
  38. `g_succ_state` = 匹配状态上兄弟的成功率（无匹配 NaN）；
  39. `g_succ_alt_sd` = 匹配状态上各 a′ 兄弟成功率的标准差（≥2 个不同动作才定义）；
  40. `g_singleton` = 1[g_cover == 0]。

**家族：** F6 World-pure {21–28}；F7 World-model {29–33}；F8 Interaction {34–36}；F9 Siblings {37–40}。指认规则与上文相同。`audit_metadata` 的访问模型列：21–36 = `env_access`，37–40 = `siblings`，W0 = `privileged`。

### 4.2 预注册预测（冻结文件中写明，Phase 3 逐条对账；预测错了照写，不改特征；FREEZE ADDENDUM：P1–P4 中的 "B2" 一律指 Pivotal_het 标签）

- **P1**：`entropy_q` 在 B1 上过线，在 B2 上归一化效应 CI 含 0。（"熵找得到活状态找不到分叉"。）
- **P2**：`target_revealed` 单特征在 B2 上的归一化效应 ≥ 全部 W1-pure 多特征的 50%。（ALFWorld 分叉主要是信息揭示。）
- **P3**：W1-pure 多特征在 B2 上的归一化效应 ≥ W1-model 多特征的 50%。（世界的回应不需要模型解读就有一半的用。）
- **P4**：`budget_left` 交互项 leave-out 使 W1 家族在 B2 上的 AUROC 下降，差值 CI 不含 0。（视界约束是真的。）
- ~~P5~~ → **已知事实 F1**（FREEZE ADDENDUM §1 Flag 3；纯标签量，冻结前已由 D-042 算出）：**64.2% [59.2, 69.0] 的分叉（Pivotal_het）在 chosen-vs-mean 信用下不可见**（Masked = het ∧ ¬Credit-pivotal，305/475）。论文 §3 的第二个头条。

### 4.3 成本列

每臂的成本向量 (环境步, 模型前向 token, LLM 调用数, 回放续跑数)，写入 `audit_metadata`，E1 用。

## 5. 数据 schema（所有脚本必须遵守，放在 `data/`）

`decision_points.parquet` — 一行一个决策点
```
traj_id:str, turn:int, task_id:str, task_type:str, seed:int,
prompt_path:str          # 该步真实输入文本的文件路径（不要存进表里）
chosen_action:str, admissible_actions:list[str],
A_replay:float|null, n_factual:int, n_alt:int,
factual_outcomes:list[int], alt_outcomes:list[list[int]],
sigma_floor:float, cf_defined:bool, exclude_reason:str|null,
source:str               # "cwgt_archive" | "own_replay"
```

`scores.parquet` — 一行一个 (决策点, 候选动作)
```
traj_id, turn, action:str, is_chosen:bool, is_admissible:bool,
logp_sum:float, logp_mean:float, n_tokens:int,
logp_sum_notask:float, logp_mean_notask:float,
scorer:str ("hfprefix" 主口径 | "vllm" 敏感性 | "hf" 核对), model_hash:str
```

`hidden/{position}_{layer}.npy`（position ∈ {prompt_end, pre_action}）+ `hidden/index.parquet` (traj_id, turn, row)

`data/replay_b{bank}/{traj_id}/t{turn:02d}.json` — 第 bank+1 个 replay 库（bank=1…）；schema 同库 1（`data/replay/`）；替代动作列表从库 1 复制，不重采；种子 = 库 1 公式 + bank·10¹¹（D-024）。

`data/labels_L.parquet` — 一行一个决策点：各库计数、`live_b12`（库 1+2 池非常量）、L2 标签（库 3+ 合并 A 与符号）、信度划分（库 3+4 vs 5+6）。

`features_B.parquet` — traj_id, turn, 第 4 节全部特征列

FINAL RULING 新增：`data/replay_gold/`（gold 集续跑，§6.3a）；`data/siblings/`（兄弟轨迹，R3）；`data/peek_alts/`（W1/W0 peek，D-029）；`hidden_peek/`（peek 态隐状态，W1-V/P）；`data/w1_judge.parquet`（W1-J）；`eval/het_fork.json`、`eval/pairs.json`（§6.0 新增标签）；`features_W1.parquet`、`features_G.parquet`（§4.1 特征 20–40）。

`eval/*.json` — 每个检验一个文件，含：指标值、CI、n、脚本路径、commit、生成时间

`audit_metadata.parquet` — 每个特征：`availability` (forward|hindsight)、计算时点、是否需要策略快照、是否需要额外前向、每点平均耗时

## 6. 指标精确定义（FINAL RULING 2026-10-26 修订；与 CWGT 可比的部分保留）

### 6.0 集合、库与标签（FINAL RULING §2.1）

- **库**：库 1 = Phase 1 的 replay（`data/replay/`，每点 factual×3 + 4 替代×3）。库 b ≥ 2 = 同一组 4 个替代动作、换种子、同样每臂 3 次（`data/replay_b{b-1}/`）。替代与草稿不重采。σ_floor(t) = 库 1 factual 重放结果的标准差（CWGT 定义）。
- **D**：cf_defined（库 1），n = 3,038。
- **Dead / Live(𝓛)**：库 1+2 合并 30 次续跑池常量 / 非常量。𝓛 n = 1,478。
- **Credit-pivotal（原 Pivotal_agg；FREEZE ADDENDUM 改名，继续报告）**：𝓛 内，库 3–6 的 factual(12) vs 替代池(48) Newcombe 90% 区间不含 0；含义"chosen 显著异于替代均值"。n = 229（零假设下期望 118）。敏感性规则：|A_3456| ≥ 0.25 且 A_34、A_56 同号非零（n = 193，期望 86），只报告。
- **Pivotal_het = Fork（B2 主标签，FREEZE ADDENDUM）**：𝓛 内，五臂（factual 12、替代 j 各 12）似然比 G 检验，p 值由 60 个结局在五臂间的 10,000 次置换给出（条件于边际），α = 0.10。n = 475（零假设下期望 104；保守精确率下界 0.78）。报告：n、与 Credit-pivotal 的 2×2 重叠、**Masked** = het ∧ ¬Credit-pivotal（305）的比例及其 cluster CI。
- **三域（FREEZE ADDENDUM）**：Dead（1,560，51.3%）/ **Indifferent := Live ∧ ¬het**（1,003，33.0%）/ **Fork := het**（475，15.6%）。Fork 内分 Credit-pivotal ∩ het（170，5.6%）与 Masked（305，10.0%）。零假设期望计数（het 104 / Credit-pivotal 118 / ∩ 35）与观测并列报告。a′_u 敏感性（D-039）是在旧 Indifferent（𝓛 \ Credit-pivotal）上做的，报告时注明，不重跑。
- **B3 集合**：Credit-pivotal ∩ Pivotal_het，n = 170（零假设下期望 35，保守精确率下界 0.79）；Masked 点上的 sign(A) 是噪声，不进 B3。
- **Pair-level C_j**：每 (点, 替代 j)，factual 12 vs 替代 j 12 的 Newcombe 90%；sig_j = 区间不含 0。
- **参考依赖 T-ref（报告项）**：𝓛 内 ≥2 个 sig_j 的点中，sig_j 符号不一致的比例，cluster CI（D-042：14.1%）；另报 Credit-pivotal 在 a′_u 替代下的翻转率（D-039 的 100 点，若 gold 覆盖更多则更新）。写进"信用定义依赖参考集"一段。
- **零质量（报告项，措辞受限）**：混合拟合 p ∈ [0.20, 0.25]（q ∈ [0.90, 1.00]）。论文只许写"在 12 次/臂下无可检出效应的 live 点比例 ≥ 75%；真实零效应比例不可点识别，由深度曲线给出区间"。**禁止**写"41% 的点是零效应"。
- **CWGT 对比标签**：pivotal_cwgt = 1[库 1 A_replay ≠ 0]，只用于复现门与 §6.2 的 M1（与 Fork、Credit-pivotal 都不是同一个量，写作时区分）。
- **第二目标（CARL 复现，零成本）**：`crit_sd`（库 1 全部续跑结果的 SD）；高熵组 vs 低熵组 Cliff's δ，与 CARL 的 0.42 并排。

### 6.1 统计方法

主 CI = **按轨迹 cluster bootstrap**（10,000 次）。所有配对比较（B vs A、B vs `chosen_freq`、B vs V、B vs ρ_t、家族之间、B vs C）用轨迹层面 paired bootstrap 差值 CI。Wilson CI 只保留在臂 A 对 CWGT 的直接对比行。学习型分数（B 多特征、C、V、B+C）一律用 GroupKFold(by trajectory, 5 折) 的 out-of-fold 预测。

### 6.2 L1 指标（描述性 + CWGT 对比；B1 的 CWGT 可比版本）

**M1 Recall@route。** 对分数 s，按 s 降序取 D 的前 r% 为"路由集"。r ∈ {13.1（CWGT 固定数字）, 本数据集 no-counterfactual 率}。recall = |路由集 ∩ pivotal| / |pivotal|；额外报告 AUROC（对 y_state）与 precision@route。

**M5 可测性结构（sanity）。** |D|/总数、pivotal 比例、σ_floor=0 比例、被排除点与保留点的 chosen conf 均值差（CWGT 86.9% / 30.5% / 86.0% / −0.70），整除性三类计数；另报"若按 strict-xml 解析，有多少步会变成 Nothing happens"（存储的 `fmt` 字段）。报告，不调整。

### 6.3 三条边界 + 一条报告行（主判据所在层，FINAL RULING §2.2）

| 边界 | 集合 | 标签 | 回放天花板（已测） |
|---|---|---|---|
| **B1** Dead\|Live | D | Live | 0.93（库 2 池方差，3/臂，D-035） |
| **B2** Indifferent\|Fork | 𝓛 | **Pivotal_het（主）**；Credit-pivotal（报告） | het：库 1+2 子抽样 m/臂的五臂 G 统计量（D-044）；Credit-pivotal：0.66（库 1+2 \|A\|，6/臂） |
| **B3** 符号 | Credit-pivotal ∩ het（170） | sign(A_3456) | 库 1+2 A，在 170 点上重算（D-044）；旧 229 点为 0.84 |
| L2-sign | 𝓛 \ {A=0} | sign(A_3456) | 0.64 — **只报告** |

每条边界、每个臂报告：AUROC + 按轨迹 cluster bootstrap 95% CI；分层 shuffle 对照（B2/B3 按轨迹内进度三分位分层，B1 不分层；1,000 次）；对 V、对臂 A 的 paired 差值 CI；**归一化效应** = (AUROC − 0.5)/(天花板 − 0.5)，天花板为同边界同集合的库 1+2 回放（m=6；B1 用 m=3）；**REV** = 使 B_m（库 1+2 子抽样 200 次均值）AUROC ≥ 该臂的最小 m ∈ {1,2,3,6}。

**过线** := 下界 > shuffle 97.5% 分位 ∧ paired-vs-V CI 不含 0 ∧ paired-vs-A CI 不含 0。

**保留报告项（不进判定，与 CWGT 可比）：**

**M2 轨迹内 Spearman。** 在 𝓛 内，对每条有 ≥5 个 𝓛 点（同时报告 ≥4 的版本，D-006）且 A_3456 非常数的轨迹算 Spearman(s, A_3456)，取跨轨迹中位数，10,000 次轨迹 bootstrap。附信度校正值（除以 √(库 3+4 vs 5+6 的 Spearman)）。

**M3 分层 shuffle 对照（必做）。** 每条轨迹内，按轨迹内进度三分位分层，在集合点内置换 s，重算 AUROC 与 M2；1,000 次得对照分布。报告置换实际改变的位置比例（CWGT 90.4%）。L1 上的 M1 也做（不分层版）。上面"过线"所用的 shuffle 即此。

**M4 流畅度污染。** 每条轨迹内 Δ_t = z(s_t) − z(A_t)；Spearman(Δ_t, conf_t)，跨轨迹中位数 + cluster bootstrap CI。**复现目标 = 臂 H1（ρ_t）的 T3 ≈ +0.752 [0.647, 0.793]**（D-005）；臂 A 的 M4 标注"近乎机械"。对 `margin_detask`、`margin_raw`、臂 C 分别报告，并报告 OLS 中条件掉流畅度后 s 与 A 的偏相关。M4 在 D（库 1 A_replay，与 CWGT 可比）与 𝓛（A_3456）上各报告一次。

**状态价值偏回归。** 在各边界集合内，把 V 的 out-of-fold 预测作为协变量，报告各臂分数对该边界标签的剩余判别力（logistic 偏系数 + cluster bootstrap CI）。

### 6.3a Gold 集与机械切换（FINAL RULING §2.3）

300 点，每臂补至 32 次（种子按 D-024 的 bank·10¹¹ 格式；具体 bank 编号见 DECISIONS）。分两层：
- **G-unbiased（200 点）**：𝓛 内按 `n_draws` 三分位 × 进度三分位 × 任务类型分层随机抽（`n_draws` 为草稿采样器输出，forward-safe）。用途：三域比例的无偏重估；B3 高信度子集；pivotal 规则召回率。
- **G-candidate（100 点）**：从 |A_3456| ≥ 0.2 的 𝓛 点中随机抽。用途：标签复现率。**不用于任何比例估计。**

**复现率口径（FREEZE ADDENDUM §1 Flag 2）**：只用 gold 新增的 20 次/臂（factual 20 vs 替代池 80 的 Newcombe 90%；五臂 G 用 20/臂），对 G-candidate 中被 12 次规则选中的点计算"独立复现率"（Credit-pivotal、het、Credit-pivotal ∩ het 各一个）；同时报告独立 20 次口径的零假设复现率（置换）。合并 32 次的口径同时报告，标明有选择偏差。

**机械切换（预注册，FREEZE ADDENDUM 改写）**：若 Credit-pivotal ∩ het 在独立 20 次下的复现率 < 0.50，则 **B3 降为报告项**（不进 §8 判定表），写 DECISIONS，不另行裁决。B2（het）不设切换。原"切到敏感性规则"的条款删除。≈ 7 h GPU。

### 6.4 L3（可选，预注册，不进 §8 判定）：recovery 位置选择（RULING 7.6）

失败轨迹上：选择器在每条轨迹选一个位置（臂 B 多特征 out-of-fold 预测 P(A<0) 最大；对照 = −conf、identifier-hash 随机）；替代 = 库 1+2 合并计数最大者（平局取最小索引；factual 不参与）；评估 = 库 3+ 该替代的分支成功率。两个差值：vs 同位置 factual 重跑；vs 随机位置同规则。轨迹层面 paired bootstrap，Bonferroni。与 CWGT 修订版 confidence 行并排（+8.54 pp / +3.33 pp）。

### 6.5 M6 机制分析（探索性，`exploratory_`，不进判定）

(i) 𝓛 内，替代与 chosen 的类型/语义距离 vs 两者结果差——这是 replay 标签自身的性质，写作时不得作为"特征为什么预测信用"的机制证据；(ii) 失败轨迹中被指认家族的高值步是否先于不可挽回点（此后 L1 序列恒为全失败）出现。

### 6.6 E1：fork 预算分配（零 GPU，FINAL RULING §5）

每轨迹选 k ∈ {1,2,4,8} 个点。选择器（只用 forward 量）：uniform（轨迹内进度等间隔）、random、entropy-top-k、B-top-k（P(Live) out-of-fold）、B→W1-pure（B 先取 P(Live) 最高的 2k 个点，再按 W1-pure 在 B2 上的 out-of-fold 分数取前 k）、oracle（gold/库 3–6 标签）。评估：选中集 Fork(het) 与 Credit-pivotal 覆盖率；选中集上用库 1+2 m=3 估 A 与库 3–6 A 的 Pearson；每轨迹捕获的 |A| 总量 / k。轨迹 paired bootstrap。成本横轴用 §4.3 的向量（部署口径与实验口径并报）。与 CRR 的 ≤4 fork 直接对话。**`exploratory_adaptive_replay`**（FREEZE ADDENDUM §4）：离线从库 1–4 按序抽续跑的自适应回放选择器，标签用库 5–6；只作探索行，不进主表、不进判定。

## 7. 分阶段任务与停/走检查

### Phase 0（第 1–2 天）：找存档、锁环境、读论文
0.1 **找 CWGT 存档。** 顺序：(a) `https://arxiv.org/src/2608.19760` 下载 TeX 源，grep `http`、`anonymous`、`4open`、`osf`、`zenodo`、`github`、`huggingface`；(b) arXiv 页面 Code/Data 标签、alphaXiv、Papers with Code、Semantic Scholar；(c) OpenReview 按标题搜（可能是 NeurIPS 2026 / ICLR 2027 投稿）；(d) GitHub 按标题与 "replay archive" 搜；(e) 都没有则通过 arXiv 作者邮箱礼貌索取，同时**不等回复，直接进 Branch B**。结果写 `DECISIONS.md`。
0.2 建 conda 环境，pin 版本，`alfworld-download`，跑通一局随机策略。
0.3 读 CWGT（含 Appendix Q）与 HCAPO 附录模板。确认模板是否列合法动作。
0.4 建目录：`data/ scripts/ eval/ hidden/ prompts/ logs/ DECISIONS.md FEATURES_FROZEN.md`。
**停/走：** 环境跑通；模板问题有结论；存档状态明确。

### Phase 1（第 1 周）：取得 A_replay 与对齐

**Branch A（存档拿到）：**
1A.1 解析存档，填 `decision_points.parquet`（source="cwgt_archive"）。必须能重建每个决策点的真实 prompt；若存档只有摘要无 prompt，按 HCAPO 模板 + 存档里的动作/观察序列重建，并抽 20 点把重建 prompt 下 chosen action 的 logprob 与存档的 logprob dump 对比，秩序一致（Spearman > 0.95）才继续。
1A.2 用 HF 对 20 点做 chosen scoring，与 vLLM 对比，秩序一致。
**停/走：** 对齐通过；|D| ≈ 1,768（Qwen）。

**Branch B（存档没拿到）：**
1B.1 用 vLLM + HCAPO 模板跑 50 条 Qwen 轨迹（seeds 0–49，固定任务列表，T=0.7），存每步 prompt、chosen、观察、合法动作。
1B.2 对每个决策点按 CWGT 协议 replay：factual ×3；从策略在 T=0.7 下拒绝采样最多 300 次取 4 个不同合法替代，各续跑 ×3；续跑到终局或步数上限（与原轨迹上限一致）。采不出 4 个替代 → `cf_defined=False`，记 `exclude_reason`，**不插补**。
1B.3 计算 A_replay、σ_floor，填表（source="own_replay"）。
1B.4 M5 sanity：与 CWGT 的可测性结构对比。
**预算：** 50 × ~35 步 × 15 续跑 × ~20 步 ≈ 50 万步生成；单卡 96GB vLLM 7B 约 2–4 天。先用 5 条轨迹试跑估算真实耗时再全量。
**停/走：** M5 与 CWGT 同量级；|D| ≥ 1,200。

### Phase 1C（RULING 2026-10-23 §7.5）：扩轨迹、多库、校准

1C.0 **校准（CPU）**：用 CWGT 修订版 `evidence.json` 的两库 16 次计数做超几何子抽样（2,000 次），d ∈ {3,6,9,12,16}：活点比例、符号一致率（双方都活且非零）、Pearson、pooled Spearman；d=16 必须复现论文 0.330 / 0.736。脚本 `scripts/review/r03_cwgt_depth_calibration.py`，结果 `eval/phase1c0_depth_calibration.json`。**扩展规则（阈值定死）：d=12 符号一致率 < 0.65 → 库 3–6 扩为库 3–10（每臂 24 次）。** 校准不是保证；实测的库 3+4 vs 5+6 一致率出来后以实测为准，但实测后不再改设计，只报告。
1C.1 **扩到 100 条轨迹**：`05_freeze_tasks.py` 同一 seed、同一分层规则从剩余 train 游戏再冻结 50 个（s050–s099），提交 `task_list.json` 后 rollout + 库 1（同 1B.1–1B.3）。
1C.2 **库 2**：全部 D（100 条轨迹），每臂 3 次。
1C.3 **库 3–6（或 3–10）**：仅 𝓛（库 1+2 定义），每库每臂 3 次。
1C.4 写 `data/labels_L.parquet`；报告 L2 信度天花板与 L1 两库一致性。
**预算（0.84 s/续跑实测）：** 新 50 条 rollout + 库 1 ≈ 7 h；库 2 ≈ 11.4 h；库 3–6 ≈ 22 h（扩展再 +22 h）。
**停/走：** 种子互异且逐请求记录；重放分歧 = 0；库 1 重算 0 不匹配。

### Phase 1D（FINAL RULING 2026-10-26 §6）：执行序

| 步 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 6.1 | 改 CLAUDE.md §1/§4/§6/§8，提交 | — | 预注册修订 |
| 6.2 | Pivotal_het 计数、2×2 重叠、被掩盖分叉比例；pair-level C_j；T-ref；三域按成败 × 进度交叉表 | CPU | `eval/het_fork.json`、`eval/pairs.json`，D-042 |
| 6.3 | GPU 队列：gold（≈7 h）→ 兄弟 G（≈20 min）→ peek 态隐状态 for W1-V/P（≈1 h）→ W1-J 调用（≈3 h） | GPU ≈ 11 h | `data/replay_gold/`、`data/siblings/`、`hidden_peek/`、`data/w1_judge.parquet` |
| 6.4 | gold 完成 → 机械切换判断 → DECISIONS | CPU | — |
| 6.5 | 起草 `FEATURES_FROZEN.md`（§4 + §4.1 全部 + 先验符号 + P1–P5 + E1 定义 + 成本向量定义 + W1-J prompt 原文），**贴给研究者过目后提交** | — | 冻结 |
| 6.6 | Phase 2：从 fp32 分数算 B 特征与 A/H1；C/V probe；W1 特征；G 特征 | CPU + 少量 GPU | `features_*.parquet` |
| 6.7 | Phase 3：B1/B2(het 主, Credit-pivotal 报告)/B3(∩)/L2-sign × 全部臂；归一化效应；REV；E1；P1–P4 对账 + F1；判定 | CPU | `RESULTS.md` |
| 6.8 | **并行开始**：论文 §3（三域、深度曲线、T-ref、零质量界、a′_u）——数据已终版 | — | 草稿 |
| 6.9 | WebShop：50–100 条、库 1–2 全 D、库 3–6 仅 𝓛、peek；复制目标 = **P2 的机制是否成立**，不只是 AUROC 表 | GPU 数天 | 第二环境 |

W1-J 的 prompt 先写进 DECISIONS 草稿，研究者确认后才调用。

**FREEZE ADDENDUM 执行序（取代上表 6.1–6.5）**：(1) 改 CLAUDE.md §6.0/§6.3/§6.3a/§8，提交；(2) CPU：B2-het 天花板、B3(∩) 天花板 → `eval/ceilings.json`，三域按新命名 → `eval/regimes.json` 的 `fork_het` 字段，D-044；(3) 按补充裁决 §3 修改冻结草稿 → 根目录 `FEATURES_FROZEN.md` → 提交，hash 写 DECISIONS，此后冻结；(4) 写 `logs/W1J_APPROVED`；(5) GPU 队列 `run_phase1d_gpu.sh`；(6) gold 完成 → 复现率 → D-045 → B3 是否降级；(7) Phase 2、3。

### Phase 2（第 2 周）：特征与隐状态
> FINAL RULING：2.0 与 2.2（原始分数与隐状态，fp32 主口径）已完成（D-033、D-034、D-040、D-041）。2.1 的冻结 = Phase 1D 6.5（内容按 §4、§4.1–4.3），2.3/2.4 = 6.6。
2.0 **D-003 pipeline 核对（HF，§9.2 安全）**：≥200 个随机点（含 ≥60 个被排除点）上按 §4 臂 A 口径算 chosen conf，比较"保留 − 排除"均值差（预期为负、量级接近 CWGT −0.70；若不是，写 DECISIONS 并继续）。
2.1 **先写并 commit `FEATURES_FROZEN.md`**（第 4 节 B 列表 + 家族 + 每特征 L1/L2 先验符号 + 臂 A/H1 构造 + V 基线输入 + L3 设计），commit hash 记入 `DECISIONS.md`。此后不得增删特征；若实在要加，标为 `exploratory_` 前缀并单独报告。
2.2 对 D 中每点：vLLM `prompt_logprobs` teacher-forced scoring 候选集（原 prompt 与去任务 prompt 各一次，先在 20 点上与 HF 核对秩序）；HF 一次前向导出两个位置的 hidden states（3 层）；臂 H1 的 hindsight pass。填 `scores.parquet`、`hidden/`。
2.3 算 `features_B.parquet`、臂 A 的 `conf_t`。
2.4 填 `audit_metadata.parquet`。
**预算：** ~1,800 点 × ~15 候选 × 2 ≈ 5.4 万次 scoring，prompt 1–2k token，单卡数小时。
**停/走：** 无 NaN；`chosen_rank==1` 的比例报告（应为多数但非全部，T=0.7）；臂 A 的 conf 与 Phase 1 的 logprob 一致。

### Phase 3（第 3 周）：评估与判定
> FINAL RULING：按 §6.3 三条边界与 §8 读法执行（Phase 1D 6.7）；下面 3.1 中的 L2/M7 由 §6.3 的 B2/B3/L2-sign 取代。
3.1 L1：M1、M5、CARL δ。L2：M7、M2、M3、M4、V 偏回归；臂 A / `chosen_freq` / V / H1 / B（多特征、每个单特征、每个家族 family-only 与 leave-one-out）/ C（每层每位置）/ B+C / C→B 分解。L3（可选）。M6（探索性）。
3.2 全部结果写 `eval/`，并生成一张汇总表 `RESULTS.md`，每个数字带 CI、n、commit。
3.3 按第 8 节规则写判定，写入 `RESULTS.md` 末尾。
**停/走：** 满足 §8 的复现门槛；否则回查 pipeline，不得进入判定。

### Phase 4（条件触发）：按判定执行
> FINAL RULING：任一 §8 读法成立后均做 WebShop 复制（Phase 1D 6.9），复制目标 = P2 的机制是否成立。下面三条按旧读法保留作记录。
- "继续" → WebShop 复制（AgentOPSD 或 verl-agent 的 WebShop 分支，同协议）。
- "转向 C" → 分解 C：probe 方向投影到 B、残差分析、逐层对比。
- "证伪" → WebShop 复制一次；仍证伪则写负结果报告，停。

## 8. 预注册判定规则（FINAL RULING 2026-10-26 §4；冻结后不得再修改）

**门**（不过则不判定）：
- **复现门**：臂 A 在 pivotal_cwgt（库 1 A≠0）标签上 13.1% 路由召回的 cluster bootstrap CI 含 13.1%（若显著高于 13.1%，写入 RESULTS 解释并继续，不算 pipeline 失败）；ρ_t 的 T3 与 [0.647, 0.793] 相容；HF–vLLM 候选秩序一致（已过，D-040）；M5 三量与 CWGT 同量级。
- **标签门**：B1 两库一致 ≥ 0.85（0.902，已过）；|B3 集合| ≥ 150（170，已过）；B2 = het 不设切换（保守精确率下界 0.78）；B3 按 §6.3a 的独立 20 次复现率决定是否降为报告项。

| 读法 | 条件 | 主张 |
|---|---|---|
| **R-LADDER** | 模型侧（B 或 C）过 B1；W1-pure 过 B2；B3 无廉价臂过或仅 W1 过 | 模型定位活状态，世界定位分叉，符号要回放 |
| **R-MODEL** | 模型侧过 B1 且过 B2 | 决策结构能定位分叉；世界侧非必需 |
| **R-WORLD** | 只有 W1（pure 或 model）过 B2 | 世界侧是分叉的唯一廉价来源；若只有 W1-model 过，标 "(model)" |
| **R-REPLAY-ONLY** | 无廉价臂过 B2 | 分叉必须执行才能找到；E1 只按 B1 路由 |
| **R-NULL** | 无臂在 B1 上过线 | 步级信用 = 状态价值 + 噪声 |

**优先级（FREEZE ADDENDUM §3 第 6 项）**：R-MODEL 与 R-LADDER 可同时满足。两者都过 B2 时，算模型侧多特征与 W1-pure 多特征在 B2 上的 paired 差值 CI：模型显著更高 → **R-MODEL**；W1 显著更高 → **R-LADDER**；CI 含 0 → **R-EITHER**（两种廉价来源各自足以定位分叉，论文以 REV 更低者为主叙事）。其余读法互斥，按表序取第一个满足者。

**后缀**（FREEZE ADDENDUM 后 het 已是 B2 主标签，原 "(het)" 后缀不再适用）：若某读法在 het 上成立而在 Credit-pivotal 上不成立，在论文中说明两个估计量的差别；不另加后缀（D-044 记录此解释）。

每个读法附：三域表（含零假设期望计数）、深度曲线（自有 + CWGT）、T-ref、F1（Masked 64.2%）、REV 表、E1 曲线、P1–P4 对账。全部可成文。

## 9. 禁止事项

1. 不用 3B 模型。不改温度。不改模板（改了就和 CWGT 不可比）。
2. 任何与 A_replay 的关联计算之前，`FEATURES_FROZEN.md` 必须已 commit。
3. 所有交叉验证按轨迹分组（GroupKFold），绝不按决策点随机分。
4. 反事实未定义的点：排除并计数，**不插补，不用 0 代替**。
5. 不跳过 M3 shuffle 对照。没有对照的数字不写进 `RESULTS.md`。
6. 本阶段不做任何策略训练、不改 advantage、不接 GRPO。
7. 不把"更多日志字段"或"judge 解释更合理"当作结果。结果只有 §6 定义的指标。
8. 看到结果之后新想的特征一律标 `exploratory_`，不进主表。
9. 每个数字可追溯到脚本 + commit + 输入文件 hash。
10. 遇到本文件没规定的选择，写 `DECISIONS.md` 再做，不静默。

## 10. 目录

```
project/
  CLAUDE.md                 # 本文件
  DECISIONS.md              # 所有未规定选择的记录
  FEATURES_FROZEN.md        # Phase 2.1 commit
  RESULTS.md                # Phase 3 产出
  env/                      # conda yml, pip freeze, alfworld config
  prompts/                  # 每个决策点的真实 prompt 文本
  data/                     # 第 5 节的 parquet
  hidden/                   # 残差流 npy + index
  scripts/
    00_find_archive.md      # Phase 0.1 的检索记录
    10_rollout.py           # Branch B
    11_replay.py            # Branch B
    12_parse_archive.py     # Branch A
    13_align_check.py
    20_score_candidates.py  # HF teacher-forced + hidden export
    21_features_B.py
    30_eval_M1_M5.py
    31_probe_C.py
    32_decompose_C_on_B.py
  eval/                     # json
  logs/
```

## 11. 汇报模板（`RESULTS.md` 必含）

1. 环境与版本表（模型 hash、库版本、模板来源、seed）。
2. 数据表：总决策点、|D|、pivotal 数与比例、排除数与原因分布、σ_floor=0 比例、排除点 logprob 偏移。
3. 复现表：臂 A 的 M1（L1）、M4，臂 H1 的 T3，M5，与 CWGT 数字并排；两种 conf 构造行；strict-xml 解析敏感性行。
4. 主表分两块。**L1 块**：各臂 M1（召回、AUROC、CI）与 shuffle 对照；`crit_sd` 的 Cliff's δ（高熵 vs 低熵）与 CARL 0.42 并排。**L2 块**：A / `chosen_freq` / V / H1 / B(多特征) / B(每个单特征，`entropy_q` 与 `entropy_sampled` 各占一行并标注"CARL 复现") / C(每层每位置) / B+C 的 M7（AUROC、CI）、M3 分层对照区间、M2（中位数、CI、信度校正值）、M4、V 偏回归；信度天花板行；`chosen_admissible=1` 子集敏感性行；𝓛 仅库 1 定义的敏感性行。另附家族 family-only / leave-one-out 两表。L3（若做）单列。
4a. **FINAL RULING 主表（FREEZE ADDENDUM 命名）**：B1 / B2（Fork = het 主，Credit-pivotal 报告）/ B3（Credit-pivotal ∩ het）/ L2-sign × 全部臂（R1–R6），AUROC、CI、shuffle 区间、paired vs V/A、归一化效应、REV；三域表（按成败 × 进度，含零假设期望计数）、深度曲线（自有 + CWGT）、T-ref、F1、零质量界、E1 曲线（含 `exploratory_adaptive_replay` 探索行）、P1–P4 对账表。上面第 4 项的 L1/L2 块按此改读，M7 由 B2/B3/L2-sign 取代。
5. C→B 分解 R²。
6. 成本表（来自 audit_metadata）。
7. 判定：引用第 8 节哪一条，附触发它的具体数字。
8. 偏离记录：`DECISIONS.md` 摘要。
9. 探索性结果（若有，单列）。

---

**最终提醒：** 本项目的价值在于测量的可信度，不在于结果的方向。一个带对照、带 CI、可复现的"证伪"与一个"成立"同等有价值。不要为了得到正结果调整协议。
