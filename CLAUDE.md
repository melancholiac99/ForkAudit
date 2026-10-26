# CLAUDE.md — 项目 M0：决策结构 vs 事后信用（三臂审计）

> 把本文件放在项目根目录并命名为 `CLAUDE.md`。你（服务器上的 Claude）在本项目中的全部工作都以本文件为准。本文件自包含，不依赖任何先前对话。遇到本文件没有规定的决策，先在 `DECISIONS.md` 里写下你的选择和理由再继续，不要静默决定。

---

## 1. 研究问题（一段话，不要改写）

当 LLM Agent 的前向决策被压缩为日志轨迹中的已选动作时，决策结构（候选动作集合、各候选的策略分数、它们的相对关系与跨时间关系）被丢弃。事后逐步信用估计器（LLM judge、logprob 比、标量置信度）只消费这条压缩后的轨迹。**本实验检验：在相同的执行 replay 参考信用下，决策结构特征（臂 B）和策略隐状态线性 probe（臂 C）是否比标量置信度（臂 A）更能识别 pivotal 步骤、更能预测 step contribution、更不受动作流畅度污染。** 已知两端：CARL 显示采样估计的动作熵在分布层面区分关键状态；CWGT 显示 chosen-action 置信度在逐步路由上是 chance level，并指出二者不冲突。本实验要回答的是：**熵之外、置信度之外的决策结构，能否把分布层面的分离变成逐步可用的识别。**

本阶段**不做策略训练**。本阶段不提出新算法。本阶段的产出是一组可复现、带对照、带置信区间的测量，以及一个按预注册规则得出的"继续/转向/停止"判定。

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

所有臂在**同一批决策点**上评估，对**同一个** A_replay。

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

**指认规则：** 某家族在 L2 上 (1) family-only 模型击败 F2 与状态价值基线 V（paired 差值 CI 不含 0），且 (2) 从全模型中 leave-one-family-out 后性能显著下降（paired 差值 CI 不含 0），才称为"那种结构"。
**先验符号（D-004）：** `FEATURES_FROZEN.md` 为每个单特征在 L1、L2 上各声明一个先验符号或"无先验"；有先验的按先验方向报告 AUROC，无先验的报告原始 AUROC（可 < 0.5），不事后翻转。

**可得性标注（必须写入 audit_metadata 的 `availability` 列）：** 特征 1–10、12–14、16–19 只用当前及过去信息，标 `forward`；特征 11 `alt_revisit` 用到未来 5 步的实际执行，标 `hindsight`。论文中"rollout 时可记录"的主张只能基于 `forward` 特征；主表要分别报告"仅 forward"与"forward + hindsight"两个版本。

臂 B 的预测器：GroupKFold(by trajectory, 5 折) 的 L2 logistic 回归（分类）/ 岭回归（连续），只调一个正则系数，内层 CV。同时报告**单特征**版本（每个特征单独作为 score）。

**臂 C — 隐状态线性 probe（信息上界臂）**
导出两个位置的残差流，层 = {14, 21, 26}（Qwen2.5-7B 共 28 层），float16：(i) **主**：原 prompt（含 generation prompt）的最后一个 token —— 臂 B 的信息上界（q_t 与 `chosen_freq` 都是该位置状态的函数）；(ii) **次**：实现的 `<think>…</think>` 之后、`<action>` 首 token 之前（"承诺时刻状态"，同样 forward 可得）。**一次 HF 前向**：对 [prompt + 实现的 response 直到 `<action>` 前] 做前向，因果注意力下位置 (i) 不受后续 token 影响，两个位置同时取出。每层每位置一个 ridge logistic probe，GroupKFold by trajectory，内层 CV 选 λ。报告每层结果与最佳层。

**臂 H1 — HCAPO ρ_t（被审计的事后信用，RULING 7.3）**
同一 recorded action、同一命令 token 掩码，两次 teacher-forced 打分：policy pass（原 prompt）与 hindsight pass（原 prompt 末尾追加一行 `Final outcome: {SUCCESS|FAILURE}. Final observation: {obs_after_final[:500]}`）。q_t = exp(Σ logp_cmd / (5.0 · n_tokens))（hindsight pass）；ρ_t = clip(q_t / mean_{同轨迹全部已评分 turn，含 cf 未定义者}(q), 0.8, 1.2)；无平滑、无 epsilon。预期 ρ_t ≈ conf（CWGT 修订版 confidence ≈ implicit）；仍做，因为 M4 的 +0.752 目标与"事后估计器忽略"的检验需要它。LLM judge（H2）本阶段不做。

**状态价值基线 V（L2 必须击败的对手）**
输入只限前向可得量：`turn_index`、`n_admissible`、臂 C 位置 (i) 的隐状态。目标 = 该点续跑池的平均成功率（库 1 与库 2 各算一次，交叉拟合）。GroupKFold by trajectory。V 的 out-of-fold 预测在 L2 中既作独立对手，也作偏回归协变量。

**联合与分解：** B+C 联合；把 C 最佳层 probe 的逐点输出对 B 特征做线性回归，报告 R²（"C 里有多少能被 B 解释"）。所有臂的主分析在 L2（§6）上进行。

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

`eval/*.json` — 每个检验一个文件，含：指标值、CI、n、脚本路径、commit、生成时间

`audit_metadata.parquet` — 每个特征：`availability` (forward|hindsight)、计算时点、是否需要策略快照、是否需要额外前向、每点平均耗时

## 6. 指标精确定义（RULING 2026-10-23 修订；与 CWGT 可比的部分保留）

### 6.0 集合、库与标签

- **库**：库 1 = Phase 1 的 replay（`data/replay/`，每点 factual×3 + 4 替代×3）。库 b ≥ 2 = 同一组 4 个替代动作、换种子、同样每臂 3 次（`data/replay_b{b-1}/`）。替代与草稿不重采。
- D = {cf_defined}（库 1 定义）。σ_floor(t) = 库 1 factual 重放结果的标准差（CWGT 定义）。
- **L1 标签（状态关键性）**：y_state = 1[库 1 的 15 次续跑池非常量]；连续版 `crit_sd`。CWGT 对比用的 pivotal = 1[A_replay ≠ 0]（库 1）同时报告（与 y_state 仅差整除性巧合点）。
- **活跃集** 𝓛 = {t ∈ D : 库 1+2 合并 30 次续跑池非常量}（主）；敏感性 = 仅库 1 定义（15 次）。
- **L2 标签（动作贡献）**：库 3–6 合并（每臂 12 次；若 §7 Phase 1C 扩展规则触发则库 3–10，每臂 24 次）的 A 及其符号。A = 0 的点不进符号 AUROC（不做三分类），计数报告。敏感性：全部库合并标签。
- **L2 信度天花板**：在 𝓛 上，库 3+4 vs 库 5+6（扩展时按奇偶库分半）的 (i) 符号一致率（双方非零的点）、(ii) Spearman、(iii) Pearson；与主结果并列报告。
- **第二目标（CARL 复现，零成本）**：`crit_sd`（库 1 全部续跑结果的 SD）；高熵组 vs 低熵组 Cliff's δ，与 CARL 的 0.42 并排。

### 6.1 统计方法

主 CI = **按轨迹 cluster bootstrap**（10,000 次）。所有配对比较（B vs A、B vs `chosen_freq`、B vs V、B vs ρ_t、家族之间、B vs C）用轨迹层面 paired bootstrap 差值 CI。Wilson CI 只保留在臂 A 对 CWGT 的直接对比行。学习型分数（B 多特征、C、V、B+C）一律用 GroupKFold(by trajectory, 5 折) 的 out-of-fold 预测。

### 6.2 L1 指标（描述性 + CWGT 对比）

**M1 Recall@route。** 对分数 s，按 s 降序取 D 的前 r% 为"路由集"。r ∈ {13.1（CWGT 固定数字）, 本数据集 no-counterfactual 率}。recall = |路由集 ∩ pivotal| / |pivotal|；额外报告 AUROC（对 y_state）与 precision@route。

**M5 可测性结构（sanity）。** |D|/总数、pivotal 比例、σ_floor=0 比例、被排除点与保留点的 chosen conf 均值差（CWGT 86.9% / 30.5% / 86.0% / −0.70），整除性三类计数；另报"若按 strict-xml 解析，有多少步会变成 Nothing happens"（存储的 `fmt` 字段）。报告，不调整。

### 6.3 L2 指标（主判据所在层）

**M7 符号 AUROC（L2 主指标）。** 在 𝓛 内（剔除 A=0）以 sign(A) 为标签，分数 s 的 AUROC；cluster bootstrap CI。

**M2 轨迹内 Spearman。** 在 𝓛 内，对每条有 ≥5 个 𝓛 点（同时报告 ≥4 的版本，D-006）且 A 非常数的轨迹算 Spearman(s, A)，取跨轨迹中位数，10,000 次轨迹 bootstrap。附信度校正值（除以 √(库 3+4 vs 5+6 的 Spearman)）。**报告项，不是判定条件。**

**M3 分层 shuffle 对照（必做）。** 每条轨迹内，按轨迹内进度三分位分层，在 𝓛 点内置换 s，重算 M7 与 M2；1,000 次得对照分布。只有真实值落在对照 95% 区间之外才算有信号。报告置换实际改变的位置比例（CWGT 90.4%）。L1 上的 M1 也做（不分层版）。

**M4 流畅度污染。** 每条轨迹内 Δ_t = z(s_t) − z(A_t)；Spearman(Δ_t, conf_t)，跨轨迹中位数 + cluster bootstrap CI。**复现目标 = 臂 H1（ρ_t）的 T3 ≈ +0.752 [0.647, 0.793]**（D-005）；臂 A 的 M4 标注"近乎机械"。对 `margin_detask`、`margin_raw`、臂 C 分别报告，并报告 OLS 中条件掉流畅度后 s 与 A 的偏相关。M4 在 D（库 1 A_replay，与 CWGT 可比）与 𝓛（L2 标签）上各报告一次。

**状态价值偏回归。** 在 𝓛 内，把 V 的 out-of-fold 预测作为协变量，报告各臂分数对 sign(A) 的剩余判别力（logistic 偏系数 + cluster bootstrap CI）。

### 6.4 L3（可选，预注册，不进 R0–R5 判定）：recovery 位置选择（RULING 7.6）

失败轨迹上：选择器在每条轨迹选一个位置（臂 B 多特征 out-of-fold 预测 P(A<0) 最大；对照 = −conf、identifier-hash 随机）；替代 = 库 1+2 合并计数最大者（平局取最小索引；factual 不参与）；评估 = 库 3+ 该替代的分支成功率。两个差值：vs 同位置 factual 重跑；vs 随机位置同规则。轨迹层面 paired bootstrap，Bonferroni。与 CWGT 修订版 confidence 行并排（+8.54 pp / +3.33 pp）。

### 6.5 M6 机制分析（探索性，`exploratory_`，不进判定）

(i) 𝓛 内，替代与 chosen 的类型/语义距离 vs 两者结果差——这是 replay 标签自身的性质，写作时不得作为"特征为什么预测信用"的机制证据；(ii) 失败轨迹中被指认家族的高值步是否先于不可挽回点（此后 L1 序列恒为全失败）出现。

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

### Phase 2（第 2 周）：特征与隐状态
2.0 **D-003 pipeline 核对（HF，§9.2 安全）**：≥200 个随机点（含 ≥60 个被排除点）上按 §4 臂 A 口径算 chosen conf，比较"保留 − 排除"均值差（预期为负、量级接近 CWGT −0.70；若不是，写 DECISIONS 并继续）。
2.1 **先写并 commit `FEATURES_FROZEN.md`**（第 4 节 B 列表 + 家族 + 每特征 L1/L2 先验符号 + 臂 A/H1 构造 + V 基线输入 + L3 设计），commit hash 记入 `DECISIONS.md`。此后不得增删特征；若实在要加，标为 `exploratory_` 前缀并单独报告。
2.2 对 D 中每点：vLLM `prompt_logprobs` teacher-forced scoring 候选集（原 prompt 与去任务 prompt 各一次，先在 20 点上与 HF 核对秩序）；HF 一次前向导出两个位置的 hidden states（3 层）；臂 H1 的 hindsight pass。填 `scores.parquet`、`hidden/`。
2.3 算 `features_B.parquet`、臂 A 的 `conf_t`。
2.4 填 `audit_metadata.parquet`。
**预算：** ~1,800 点 × ~15 候选 × 2 ≈ 5.4 万次 scoring，prompt 1–2k token，单卡数小时。
**停/走：** 无 NaN；`chosen_rank==1` 的比例报告（应为多数但非全部，T=0.7）；臂 A 的 conf 与 Phase 1 的 logprob 一致。

### Phase 3（第 3 周）：评估与判定
3.1 L1：M1、M5、CARL δ。L2：M7、M2、M3、M4、V 偏回归；臂 A / `chosen_freq` / V / H1 / B（多特征、每个单特征、每个家族 family-only 与 leave-one-out）/ C（每层每位置）/ B+C / C→B 分解。L3（可选）。M6（探索性）。
3.2 全部结果写 `eval/`，并生成一张汇总表 `RESULTS.md`，每个数字带 CI、n、commit。
3.3 按第 8 节规则写判定，写入 `RESULTS.md` 末尾。
**停/走：** 满足 §8 的复现门槛；否则回查 pipeline，不得进入判定。

### Phase 4（条件触发）：按判定执行
- "继续" → WebShop 复制（AgentOPSD 或 verl-agent 的 WebShop 分支，同协议）。
- "转向 C" → 分解 C：probe 方向投影到 B、残差分析、逐层对比。
- "证伪" → WebShop 复制一次；仍证伪则写负结果报告，停。

## 8. 预注册判定规则（RULING 2026-10-23 修订；Phase 3 之前不得再修改）

先看复现门槛（否则无判定）。L2 = §6.3 的层；"①②④"指 R1 的条件编号。

| 读法 | 条件 | 判定 |
|---|---|---|
| **复现门槛** | 臂 A（−conf）在 L1 上 13.1% 路由率下的召回，其 cluster bootstrap 95% CI 含 13.1%（与随机路由一致，与 CWGT chance-level 结论相容；若显著高于 13.1%，写入 RESULTS 解释并继续，不算 pipeline 失败）；ρ_t 的 T3 与 [0.647, 0.793] 相容；HF–vLLM 候选秩序一致；M5 三量与 CWGT 同量级 | 不过则回查，不判定 |
| **R0 只识别状态** | L1 上臂 B 超过 shuffle，L2 上不满足 R1 ①② | 主张收窄为 CARL 类增强版 |
| **R1 主线成立（L2）** | ① 臂 B 多特征 M7 符号 AUROC 的 cluster bootstrap 95% 下界 > 分层 shuffle 分布 97.5% 分位，且下界 ≥ 0.55；② 对臂 A、`chosen_freq`、V、ρ_t 的 paired 差值 CI 均不含 0；④ `margin_detask` 的 M4 流畅度相关显著低于臂 A | 继续；进 Phase 4 WebShop；必分 R1a/R1b |
| **R1a 熵即可** | R1 成立，且 F2 family-only vs 全家族 paired 差值 CI 含 0 | 主张收窄为"合法集精确熵可逐步路由，置信度不能"——CARL（分布层面）与 CWGT（逐步 chance）张力的解决 |
| **R1b 结构超越熵** | R1 成立，且 ≥1 个非 F2 家族满足 §4 指认规则两条 | 完整主张成立；该家族即"那种结构" |
| **R2 信息在隐状态** | 臂 C 满足 R1 ①②，臂 B 不满足；C→B R² < 0.3 | 转向：主线改为"信用信息在策略内部状态而非输出分布"，Phase 4 做分解 |
| **R3 结构 ≈ 上界** | R1 成立且 B、C 的 M7 CI 重叠，C→B R² ≥ 0.5 | 最强结果；继续 + 强调 |
| **R4 弱信号** | R1 ① 显著但下界 < 0.55，或 R1 ①② 部分不成立但 M7 超出 shuffle | 部分成立；WebShop 复制后再定 |
| **R5 证伪** | L1、L2 上 B、C 均不超过 shuffle 对照 95% 区间 | 决策时刻状态不含可用信用信息（ALFWorld）；Phase 4 WebShop 一次复制；仍 R5 则停 |

L2 信度天花板（§6.0）与 R0–R5 并列报告。若库 3+4 vs 5+6 的符号一致率 < 0.60，L2 判定须注明"标签信度不足"，诚实结论可为"每臂 12–24 次续跑的 replay 对比在 ALFWorld 上不足以定义 turn 级动作贡献"。

## 9. 禁止事项

1. 不用 3B 模型。不改温度。不改模板（改了就和 CWGT 不可比）。
2. 任何与 A_replay 的关联计算之前，`FEATURES_FROZEN.md` 必须已 commit。
3. 所有交叉验证按轨迹分组（GroupKFold），绝不按决策点随机分。
4. 反事实未定义的点：排除并计数，**不插补，不用 0 代替**。
5. 不跳过 M3 shuffle 对照。没有对照的数字不写进 `RESULTS.md`。
6. 本阶段不做任何策略训练、不改 advantage、不接 GRPO。
7. 不把"更多日志字段"或"judge 解释更合理"当作结果。结果只有 M1–M5。
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
5. C→B 分解 R²。
6. 成本表（来自 audit_metadata）。
7. 判定：引用第 8 节哪一条，附触发它的具体数字。
8. 偏离记录：`DECISIONS.md` 摘要。
9. 探索性结果（若有，单列）。

---

**最终提醒：** 本项目的价值在于测量的可信度，不在于结果的方向。一个带对照、带 CI、可复现的"证伪"与一个"成立"同等有价值。不要为了得到正结果调整协议。
