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

**阅读任务（Phase 0）：** 读 CWGT 全文，特别是 Sec. 2 协议、Sec. 4 结果、Appendix E（可测性）、Appendix O（router）、**Appendix Q（并行工作 CARL/CAR/CSO/C3 —— 这部分尚未被核对，若其中有与本项目更近的工作，立即写入 `DECISIONS.md` 并停下等指示）**。

## 3. 环境锁定（不可更改，更改需写 DECISIONS.md）

- 策略：`Qwen/Qwen2.5-7B-Instruct`，fp16/bf16，采样温度 0.7，与 CWGT 一致。**禁止用 3B。** 第二家族 `meta-llama/Llama-3.1-8B-Instruct` 为可选复制。
- 环境：`alfworld==0.4.2`, `textworld==1.7.0`（CWGT 的 pin）。先 `alfworld-download` 拉游戏文件。
- Agent 模板：HCAPO (arXiv 2603.08754) 附录公开的 ALFWorld 模板。**Phase 1 必须确认模板是否把合法动作列进 prompt**，记录在 `DECISIONS.md`。
- 推理：vLLM 用于 rollout/replay；HF transformers 用于 teacher-forced scoring 与隐状态导出（需要 `output_hidden_states`）。两者 logprob 允许数值差异但秩序必须一致（Phase 1 检查）。
- 硬件：单卡 RTX Pro 6000（按 96GB 计；若实为 48GB 版本亦足够 7B）。
- 随机性：所有采样固定 seed 并记录；每个产出文件记录生成它的脚本路径 + git commit hash。
- Python 3.10+，`numpy scipy scikit-learn pandas pyarrow torch transformers vllm`。

## 4. 三臂设计

所有臂在**同一批决策点**上评估，对**同一个** A_replay。

**臂 A — 标量置信度（CWGT 复现，对照臂）**
`conf_t = mean token logπ(a_t | s_t)`，chosen action 在原 prompt 下 teacher-forced。目标：复现 CWGT 的 11.9% 召回与 +0.75 流畅度相关。复现不出来说明 pipeline 有问题，停。

**臂 B — 人工决策结构特征（主假设臂）**
对决策点 t 的合法动作集 𝒜_t 中每个动作 a 做 teacher-forced scoring，得 `logp_sum(a)`, `logp_mean(a)`, `n_tokens(a)`；另在**去掉任务目标描述**的 prompt 下再 score 一次得 `logp_sum_notask(a)`。令 q_t = softmax over 𝒜_t of logp_sum。特征列表**在 `FEATURES_FROZEN.md` 中冻结并 commit 之后才允许计算任何与 A_replay 的关联**：

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

**可得性标注（必须写入 audit_metadata 的 `availability` 列）：** 特征 1–10、12–14、16、17 只用当前及过去信息，标 `forward`；特征 11 `alt_revisit` 用到未来 5 步的实际执行，标 `hindsight`。论文中"rollout 时可记录"的主张只能基于 `forward` 特征；主表要分别报告"仅 forward"与"forward + hindsight"两个版本。

臂 B 的预测器：GroupKFold(by trajectory, 5 折) 的 L2 logistic 回归（分类）/ 岭回归（连续），只调一个正则系数，内层 CV。同时报告**单特征**版本（每个特征单独作为 score）。

**臂 C — 隐状态线性 probe（信息上界臂）**
在原 prompt 的最后一个 token（即 chosen action 第一个 token 生成之前）处导出残差流，层 = {14, 21, 26}（Qwen2.5-7B 共 28 层），float16。每层一个 ridge logistic probe，同样 GroupKFold by trajectory，内层 CV 选 λ。报告每层结果与最佳层。**与臂 B 在同一次前向中导出**（scoring chosen action 时顺带 `output_hidden_states=True`），避免数值不一致。

**联合与分解：** B+C 联合；把 C 最佳层 probe 的逐点输出对 B 特征做线性回归，报告 R²（"C 里有多少能被 B 解释"）。

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
scorer:str ("hf"|"vllm"), model_hash:str
```

`hidden/{layer}.npy` + `hidden/index.parquet` (traj_id, turn, row)

`features_B.parquet` — traj_id, turn, 第 4 节全部特征列

`eval/*.json` — 每个检验一个文件，含：指标值、CI、n、脚本路径、commit、生成时间

`audit_metadata.parquet` — 每个特征：`availability` (forward|hindsight)、计算时点、是否需要策略快照、是否需要额外前向、每点平均耗时

## 6. 指标精确定义（与 CWGT 可比）

记 D = {cf_defined == True} 的决策点集合。pivotal: y_t = 1[A_replay(t) ≠ 0]。σ_floor(t) = factual 重放结果的标准差（CWGT 定义）。

**第二目标（CARL 复现用，零成本）：** `crit_sd(t)` = 该点 factual 重放 ∪ 全部替代续跑的结果的标准差（CARL 的无符号关键性定义，在 ALFWorld 上的实现）。M1/M2/M3 对 pivotal 与 A_replay 为主；对 `crit_sd` 另报一列，用于与 CARL 的 δ=0.42 对比（算 Cliff's δ：高熵组 vs 低熵组的 crit_sd）。

**M1 Recall@route（主指标）。** 对分数 s，按 s 降序取 D 的前 r% 为"路由集"。r 取两个值都报告：r=13.1（CWGT 的固定数字，用于直接对比）和 r=本数据集自己的 no-counterfactual 率（CWGT 的原始定义）。recall = |路由集 ∩ pivotal| / |pivotal|。Wilson 95% CI。额外报告 AUROC 与 precision@route。学习型臂（B 多特征、C）用 GroupKFold 的 out-of-fold 分数。

**M2 轨迹内 Spearman。** 对每条有 ≥5 个 D 点且 A_replay 非常数的轨迹算 Spearman(s, A_replay)，取跨轨迹**中位数**；10,000 次对轨迹的 bootstrap 得 95% CI。CWGT 参考：implicit 0.019 [−0.109, 0.081]，judge 0.114 [0.027, 0.168]。

**M3 轨迹内 shuffle 对照（必做）。** 对每条轨迹，将 s 在其 D 点内随机置换，重算 M1/M2；1,000 次置换得对照分布。**只有真实值落在对照 95% 区间之外才算有信号。** 同时报告置换实际改变了多少位置（CWGT 报告 90.4%）。

**M4 流畅度污染（CWGT T3 复现）。** 每条轨迹内 Δ_t = z(s_t) − z(A_replay(t))；Spearman(Δ_t, mean token logπ(a_t|s_t))，跨轨迹中位数 + bootstrap CI。臂 A 应复现 ≈ +0.75。对臂 B 的 `margin_detask`、`margin_raw`、臂 C 分别报告，并报告 OLS 中条件掉流畅度后 s 与 A_replay 的偏相关。

**M5 可测性结构（sanity）。** 报告 |D|/总数、pivotal 比例、σ_floor=0 比例、被排除点与保留点的 chosen logprob 均值差。Branch B 自跑时这些应与 CWGT 的 86.9% / 30.5% / 86.0% / −0.70 同量级，否则 pipeline 可疑。

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

### Phase 2（第 2 周）：特征与隐状态
2.1 **先写并 commit `FEATURES_FROZEN.md`**（照第 4 节 B 列表），commit hash 记入 `DECISIONS.md`。此后不得增删特征；若实在要加，标为 `exploratory_` 前缀并单独报告。
2.2 对 D 中每点：HF 批量 teacher-forced scoring 全部合法动作（原 prompt 与去任务 prompt 各一次），同一前向导出 chosen 第一 token 前的 hidden states（3 层）。填 `scores.parquet`、`hidden/`。
2.3 算 `features_B.parquet`、臂 A 的 `conf_t`。
2.4 填 `audit_metadata.parquet`。
**预算：** ~1,800 点 × ~15 候选 × 2 ≈ 5.4 万次 scoring，prompt 1–2k token，单卡数小时。
**停/走：** 无 NaN；`chosen_rank==1` 的比例报告（应为多数但非全部，T=0.7）；臂 A 的 conf 与 Phase 1 的 logprob 一致。

### Phase 3（第 3 周）：评估与判定
3.1 跑 M1–M5，臂 A / B（多特征 + 每个单特征）/ C（每层）/ B+C / C→B 分解。
3.2 全部结果写 `eval/`，并生成一张汇总表 `RESULTS.md`，每个数字带 CI、n、commit。
3.3 按第 8 节规则写判定，写入 `RESULTS.md` 末尾。
**停/走：** 臂 A 复现 CWGT（召回在 [9, 15]%，M4 ≈ +0.7）；否则回查 pipeline，不得进入判定。

### Phase 4（条件触发）：按判定执行
- "继续" → WebShop 复制（AgentOPSD 或 verl-agent 的 WebShop 分支，同协议）。
- "转向 C" → 分解 C：probe 方向投影到 B、残差分析、逐层对比。
- "证伪" → WebShop 复制一次；仍证伪则写负结果报告，停。

## 8. 预注册判定规则（Phase 3 之前不得修改）

先看臂 A 是否复现（否则无判定）。然后：

| 读法 | 条件 | 判定 |
|---|---|---|
| **R1 主线成立** | 臂 B 多特征 M1 召回 ≥ 25%，Wilson 下界 > 14.9%；M3 shuffle 对照落回 chance；M4 中 `margin_detask` 的流畅度相关显著低于臂 A | 继续；进 Phase 4 WebShop。**必须再分 R1a/R1b** |
| **R1a 熵即可** | R1 成立，且 `entropy_q` 单特征的 M1 召回与多特征的 CI 重叠（对轨迹 paired bootstrap 的差值 CI 含 0） | 主张收窄为："合法集上精确计算的熵 ≠ chosen-action 置信度；前者逐步可路由，后者不能"——这是对 CARL（分布层面）与 CWGT（逐步 chance）张力的解决，仍可发表，但不再是"结构超越熵" |
| **R1b 结构超越熵** | R1 成立，且多特征 M1 召回显著高于 `entropy_q` 与 `entropy_sampled` 各自单特征（paired bootstrap 差值 CI 不含 0） | 完整主张成立 |
| **R2 信息在隐状态不在结构** | 臂 C 最佳层 M1 召回 ≥ 25% 且下界 > 14.9%，但臂 B 不满足 R1；C→B 分解 R² < 0.3 | 转向：论文主线改为"信用信息在策略内部状态而非输出分布"，Phase 4 做分解 |
| **R3 结构 ≈ 上界** | R1 成立，且臂 B 与臂 C 的 M1 召回 CI 重叠，C→B R² ≥ 0.5 | 最强结果：可解释特征等价于黑盒上界；继续 + 强调这一点 |
| **R4 弱信号** | 臂 B 或 C 召回在 (14.9%, 25%)，shuffle 对照通过 | 部分成立；WebShop 复制后再定 |
| **R5 证伪** | 臂 B 与 C 均不超过 shuffle 对照 95% 区间 | 决策时刻状态不含可用信用信息（ALFWorld）；Phase 4 WebShop 一次复制；仍 R5 则停 |

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
3. 臂 A 复现表：M1、M2、M4 与 CWGT 数字并排。
4. 主表：A / B(多特征) / B(每个单特征，`entropy_q` 与 `entropy_sampled` 必须各占一行并标注"CARL 复现") / C(每层) / B+C 的 M1（召回、AUROC、CI）、M2（中位数、CI）、M3（对照区间、是否超出）、M4（流畅度相关、偏相关）。另一列：对 `crit_sd` 目标的 Cliff's δ（高熵 vs 低熵），与 CARL 的 0.42 并排。
5. C→B 分解 R²。
6. 成本表（来自 audit_metadata）。
7. 判定：引用第 8 节哪一条，附触发它的具体数字。
8. 偏离记录：`DECISIONS.md` 摘要。
9. 探索性结果（若有，单列）。

---

**最终提醒：** 本项目的价值在于测量的可信度，不在于结果的方向。一个带对照、带 CI、可复现的"证伪"与一个"成立"同等有价值。不要为了得到正结果调整协议。
