# FEATURES_FROZEN.md — 特征、标签、预测与判定的冻结文件

> - **状态：FROZEN（2026-10-26）。** 本文件的提交 commit 即冻结点，hash 记入 DECISIONS D-044。此后目标定义、特征清单、判定规则不再变动。唯一的预先写明的切换是 §1 的 B3 降级规则；任何追加只能用 `exploratory_` 前缀，单独报告。研究者本人也受此约束（RULING_2026-10-26_FINAL 与 FREEZE ADDENDUM）。
> - **依据：** `CLAUDE.md` @ 702f89b（FINAL RULING + FREEZE ADDENDUM 已写入）；`RULING_2026-10-26_FINAL.md`；`RULING_2026-10-26_FREEZE_ADDENDUM.md` §3（草稿十项的确认）。草稿版本在 `proposals/FEATURES_FROZEN_DRAFT.md`。
> - **冻结前已知的只有标签自身的统计：** D-031、D-035、D-039、D-042、D-044。其中"按进度分层"一项本身就是 `turn_index` / `budget_left` 与标签的关联，所以这两列只作控制列，不作检验对象。冻结前没有计算过任何特征与任何结局的关联。

---

## 1. 决策点与标签（CLAUDE.md §6.0、§6.3a）

**集合：**
- **D** = cf_defined（库 1），n = 3,038。
- **Live(𝓛)** = 库 1+2 合并 30 次续跑池非常量，n = 1,478。Dead = D \ 𝓛。

**标签：**
- **Fork = Pivotal_het（B2 主标签）：** 𝓛 内，库 3–6 的五臂（factual 12、替代 j 各 12）似然比 G 检验。p 值由 60 个结局在五臂间的 10,000 次置换给出（条件于边际），α = 0.10。n = 475，零假设下期望 104。
- **Credit-pivotal（原 Pivotal_agg，B2 报告标签）：** factual(12) vs 替代池(48) 的 Newcombe（method 10 hybrid score）90% 区间不含 0。n = 229，期望 118。
  - 敏感性规则：|A_3456| ≥ 0.25 且 A_34、A_56 同号非零。n = 193，期望 86；只报告。
- **B3 集合：** Credit-pivotal ∩ Fork，n = 170，期望 35。B3 标签 = 1[A_3456 > 0]，其中 72 个为正。
- **Masked：** Fork ∧ ¬Credit-pivotal，n = 305。
- **三域：** Dead 1,560 / Indifferent（Live ∧ ¬Fork）1,003 / Fork 475。
- **L2-sign：** 𝓛 \ {A_3456 = 0} 上的 1[A_3456 > 0]，只报告。
- **pivotal_cwgt：** 1[库 1 A_replay ≠ 0]，只用于复现门与 M1。

**Gold 复现率（FREEZE ADDENDUM §1 Flag 2）：**
- 只用 gold 新增的 20 次/臂计算。Credit-pivotal 用 factual 20 vs 替代池 80 的 Newcombe 90%；Fork 用 20/臂的五臂 G（置换 10,000 次，α = 0.10）。
- 对象：G-candidate 中被 12 次规则选中的点。分别报告 Credit-pivotal、Fork、Credit-pivotal ∩ Fork 三种复现率，各附按轨迹的 cluster CI。
- 同时报告独立 20 次口径的零假设复现率（在每点的 100 个新结局上置换）。
- 合并 32 次的口径也报告，标明有选择偏差。

**预先写明的唯一切换：** 若 Credit-pivotal ∩ Fork 在独立 20 次下的复现率 < 0.50，则 B3 降为报告项（不进 §5 判定表），写 DECISIONS，不另行裁决。B2 不设切换。

## 2. 打分与构造口径（不变；后端 D-041）

- **主口径：** HF fp32 + 共享前缀 KV cache（`data/scores_fp32.parquet`、`data/scores_chosen_fp32.parquet`）。vLLM bf16 打分只作敏感性行。
- **候选集：** 𝒜_t = 合法动作 ∪ {chosen} \ {help}（`scoring.candidate_set`，D-021）。q_t = 𝒜_t 上 logp_sum 的 softmax。
- **臂 A：** conf_t = chosen 动作命令 token 的平均 logπ。context = chat(prompt) 加 generation prompt；teacher-force `<action>{a}</action>`，只计 `{a}` 的 token（D-003）。
  - 主分数 s = −conf。
  - 敏感性行：+conf；emptythink 口径；realized 口径只报告一行（D-020）。
- **去任务 prompt：** `scoring.detask`，删除任务模板行，以及观察中所有任务陈述的副本。
- **臂 H1：** 在 hindsight pass 上计算 q_t = exp(Σ logp_cmd / (5.0 · n_tokens))。hindsight pass = prompt 末尾加一行 `Final outcome: {SUCCESS|FAILURE}. Final observation: {final_obs[:500]}`。ρ_t = clip(q_t / 同轨迹全部已评分 turn 的 q 均值, 0.8, 1.2)。
- **`chosen_freq`：** log[(c + 0.5) / (32 + 0.5·|𝒜_t|)]，c = 32 份首批草稿中解析为 chosen 的数量（D-008）。
- **`entropy_sampled`：** 库 1 草稿中合法且不重复动作的频率熵（CLAUDE.md §4 特征 16）。
- **W1：** 取自 `data/peek_alts/`（D-029）。五个 peek = factual + 库 1 的 4 个替代。
- **peek 态：** 隐状态取自 `hidden_peek/`（bf16，prompt_end，与 `hidden/` 同一流程）；打分取自 `data/scores_peek_fp32.parquet`（peek 态合法集，bare context）。
- **G：** 取自 `data/siblings/`（每任务 7 条，D-043）。状态匹配 = 当前观察文本完全相同；兄弟轨迹的任一 turn 均可匹配，不要求 turn 相同。匹配率按"turn 相同"与"turn 不同"分别报告。

## 3. 特征清单与先验符号（FREEZE ADDENDUM §3 第 1–2 项确认）

方向约定：
- "+" = 值越大，越可能是标签 1。B1 的标签 1 = Live；B2 的标签 1 = Fork（het）；B3 的标签 1 = factual 更好（A_3456 > 0）。
- "0" = 无先验，报告原始 AUROC，不翻转。
- "C" = 控制列，进入多特征模型，但不作单特征检验。

| # | 特征 | 家族 | 可得性 / 访问 | B1 | B2 | B3 | 理由（简） |
|---|---|---|---|---|---|---|---|
| 1 | `margin_raw` | F1 | forward / θ_old | − | − | + | 小 margin = 犹豫 → 活、分叉；大 margin 时策略选对 |
| 2 | `margin_mean` | F1 | 同上 | − | − | + | 同 1 |
| 3 | `margin_detask` | F5 | 同上 | 0 | + | 0 | 任务条件化差越大，该步越与目标相关 |
| 4 | `entropy_q` | F2 | 同上 | + | + | 0 | CARL 方向；P1 预测 B2 上效应为 0 |
| 5 | `eff_size` | F2 | 同上 | + | + | 0 | 同 4 |
| 6 | `chosen_prob` | F1 | 同上 | − | − | + | 同 1 |
| 7 | `runner_same_type` | F3 | 同上 | 0 | − | 0 | 次选与 chosen 同类型 → 替代相似 → indifferent |
| 8 | `top3_n_types` | F3 | 同上 | + | + | 0 | 候选类型越多样 → 动作后果越可能不同 |
| 9 | `loop_chosen` | F4 | 同上 | − | 0 | − | 循环 → 多半已经是全败；循环动作更差 |
| 10 | `loop_runner` | F4 | 同上 | − | 0 | + | 次选是重复动作 → 替代更差 |
| 11 | `alt_revisit` | F4 | **hindsight** | 0 | 0 | 0 | 单列；只进 "forward + hindsight" 版本 |
| 12 | `js_prev` | F4 | forward | + | + | 0 | 分布在变 → 局面在动 |
| 13 | `sem_dispersion` | F3 | forward | + | + | 0 | 候选语义越散，后果越可能不同 |
| 14 | `sem_dist_chosen_runner` | F3 | forward | 0 | + | 0 | 同 13 |
| 15 | `turn_index`、`n_admissible`、`chosen_n_tokens`、`type(chosen)` | 控制 | forward | C | C | C | `turn_index` 与标签的关联已在 D-035/D-042/D-044 中出现 |
| 16 | `entropy_sampled` | F2 | forward / 库 1 草稿 | + | + | 0 | 同 4（"CARL 复现"行） |
| 17 | `chosen_conf` | F1 内对照 | forward | − | − | + | 同 6；臂 A 的分数取它的负值 |
| 18 | `chosen_freq` | F1 | forward / 草稿 | − | − | + | 同 6 |
| 19 | `chosen_admissible` | 控制 | forward | C | C | C | — |
| 20 | `budget_left` | 控制（交互基底） | forward | C | C | C | 同 15 |
| 21 | `w1_nothing_frac` | F6 | forward / env_access | 0 | + | + | 替代多为空动作 → chosen 相对更好、差别更大 |
| 22 | `w1_obs_len_sd` | F6 | 同上 | 0 | + | 0 | 回应差异越大，越可能分叉 |
| 23 | `w1_adm_delta_sd` | F6 | 同上 | 0 | + | 0 | 同 22 |
| 24 | `w1_task_sim_max` | F6 | 同上 | + | + | 0 | 有 peek 与任务相关 → 局面未定、可能分叉 |
| 25 | `w1_task_sim_mean` | F6 | 同上 | + | + | 0 | 同 24 |
| 26 | `target_revealed` | F6 | 同上 | + | + | 0 | 信息揭示（P2）；不是 chosen 专属，所以 B3 无先验 |
| 27 | `w1_div_emb` | F6 | 同上 | 0 | + | 0 | 同 22 |
| 28 | `w1_div_n_distinct` | F6 | 同上 | 0 | + | 0 | 同 22 |
| 29 | `w1_V_sd` | F7 | forward / env + θ_old | + | + | 0 | 后继价值差越大，越可能分叉 |
| 30 | `w1_V_chosen_minus_mean` | F7 | 同上 | 0 | 0 | + | chosen 后继价值高于替代均值 → A > 0 |
| 31 | `w1_P_entropy_mean` | F7 | 同上 | + | 0 | 0 | — |
| 32 | `w1_J_progress_frac` | F7 | 同上（W1-J 调用） | + | + | 0 | — |
| 33 | `w1_J_regress_frac` | F7 | 同上 | 0 | + | 0 | — |
| 34 | `w1_div_emb × budget_left` | F8 | forward | 0 | 0 | 0 | 作用见 P4 |
| 35 | `w1_task_sim_max × budget_left` | F8 | forward | 0 | 0 | 0 | 同 34 |
| 36 | `target_revealed × budget_left` | F8 | forward | 0 | 0 | 0 | 同 34 |
| 37 | `g_cover` | F9 | forward（GRPO 组内） / siblings | 0 | 0 | 0 | — |
| 38 | `g_succ_state` | F9 | 同上 | 0 | 0 | 0 | 与 Live 是 U 形关系（0/1 都意味着 Dead），无单调先验 |
| 39 | `g_succ_alt_sd` | F9 | 同上 | 0 | + | 0 | 兄弟在该状态选不同动作、结局不同 → 分叉 |
| 40 | `g_singleton` | F9 | 同上 | 0 | 0 | 0 | 覆盖空洞，只用于分层 |

**特征 26 的目标词抽取：**
- 任务描述转小写、去掉句末句号。记 D = `(?:a |an |some |two |the )?`，依次匹配下面四个正则，取第一个命中的捕获组 X（单个词）：
  1. `^(?:clean|heat|cool) D(\w+) and put`
  2. `^find D(\w+) and put`
  3. `^put D(?:clean |hot |cool |cold )?(\w+) (?:in|on|into)\b`
  4. `^(?:look at|examine) D(\w+) (?:under|with)\b`
- 在 100 个任务上 100/100 抽中。
- 任一 peek 观察匹配 `\bX \d+\b` 即记 1。抽不出时记 NaN，并报告计数。

**特征计算细节（冻结）：**
- 21：比较时去掉首尾空白，与 "Nothing happens." 完全相同才算。分母为替代 peek 数（4）。
- 22、23、27、28：在五个 peek（factual + 4 替代）上计算。
  - 23 的"变化" = |peek 后合法集大小| − |s_t 合法集大小|（均去掉 help）。
  - 27 用 `bge-small-en-v1.5` 的归一化嵌入，取 10 对余弦距离的均值。
- 24、25：每个 peek 观察与任务描述（`common.extract_task`）的 `bge-small` 余弦相似度，在五个 peek 上取 max 与 mean。
- 29、30：W1-V 见 §4。30 = V̂(factual peek 态) − 四个替代 peek 态 V̂ 的均值。
- 31：每个 peek 态上候选（peek 态合法集）logp_sum 的 softmax 熵，五个取均值。
- 32、33：五个 peek 中 label = progress / regress 的比例。分母只计已解析的 peek；`unparsed` 计数报告；全部未解析时记 NaN。
- 34–36：原值相乘（不先标准化）；多特征模型在训练折内统一标准化。
- 37–39：G 在每个点上用"同任务 7 条兄弟轨迹中，turn 前的观察文本 = 当前点 `obs_before` 的所有兄弟步"计算。
  - 37 = 匹配到的兄弟步数。
  - 38 = 这些兄弟步所在轨迹的成功率。
  - 39 = 按兄弟在该步所选动作分组后，各组成功率的标准差（≥2 组才定义）。

**缺失值：**
- 单特征 AUROC 只在非缺失点上计算，并报告 n。
- 多特征模型在训练折内，用中位数插补并加缺失指示列。
- 会出现缺失的特征：`js_prev`（t = 1）、`target_revealed`、`w1_J_*`、`g_succ_state`、`g_succ_alt_sd`。

## 4. 臂与学习型分数（FREEZE ADDENDUM §3 第 4–5 项确认）

**通用设置：**
- 所有学习型分数取 GroupKFold（按轨迹，5 折，`GroupKFold(n_splits=5)` 默认确定性划分）的 out-of-fold 预测。
- 分类用 L2 logistic（`liblinear`，class_weight=None），连续目标用 ridge。
- 正则系数只调一个：内层按轨迹做 5 折 CV，网格 C = 1/λ ∈ 10^{−4, −3, …, 4}，按 AUROC 选（ridge 按 MSE 选）。
- 特征在训练折内标准化。
- 每个多特征模型都包含控制列 `turn_index`、`n_admissible`。

| 臂 / 分数 | 输入 | 边界 |
|---|---|---|
| A | −conf（单分数） | B1 / B2 / B3 / L2-sign |
| H1 | ρ_t（单分数） | 同上 |
| `chosen_freq` | 单分数（独立基线行） | 同上 |
| V | `turn_index`、`n_admissible` + 臂 C 位置 (i) 隐状态；层由内层 CV 在 {14, 21, 26} 中选；目标 = 续跑池平均成功率（库 1、库 2 各一次，交叉拟合）；ridge | B1 / B2 / B3（out-of-fold 预测作独立对手与偏回归协变量） |
| B 多特征 | 1–10、12–19（forward 版）；另报 + 11（hindsight 版） | 每条边界一个模型 |
| B 家族 | F1–F5 各自 family-only + 控制列；全模型 leave-one-family-out | 指认规则（CLAUDE.md §4） |
| C | 每层 × 每位置一个 probe（6 个） | 每条边界；报告每个 probe 与最佳层 |
| W1-pure 多特征 | 20–28 + 34–36（交互项归 W1-pure：budget 可观测，属世界侧） | B1 / B2 / B3 |
| W1-model 多特征 | 20、29–33 | 同上 |
| G 多特征 | 37–40 | 同上 |
| B+C | B 多特征 + C probe（位置 prompt_end；层由训练折内的内层 CV 在 {14, 21, 26} 中选）的 out-of-fold 输出 | 同上 |
| C→B 分解 | C 最佳层输出对 B 特征的线性回归 R² | — |

**W1-V（特征 29、30）：**
- 用 V 的同一模型，作用在 peek 态隐状态上。
- 对点 t，所用的 V 模型来自不含 t 所在轨迹的训练折（与 V 的 out-of-fold 预测同折）。
- 层与 V 选中的层相同。

## 5. 评估与判定（CLAUDE.md §6.3、§8）

**每条边界、每个分数报告：**
- AUROC，以及按轨迹 cluster bootstrap（10,000 次）的 95% CI。
- 分层 shuffle 对照（1,000 次；B2/B3 按轨迹内进度三分位分层，B1 不分层）。
- 对 V、对臂 A 的 paired 差值 CI。
- 归一化效应 = (AUROC − 0.5)/(天花板 − 0.5)。天花板（`eval/ceilings.json`）：B1 = 0.930（m = 3）；B2 = 0.763（het，m = 6 的五臂 G）；B3 = 0.883（170 点，m = 6 的 A）；Credit-pivotal 报告行 = 0.661。
- REV：使 B_m（库 1+2 子抽样 200 次均值，m ∈ {1, 2, 3, 6}）的 AUROC ≥ 该臂 AUROC 的最小 m；超过 m = 6 时记 ">6"。

**过线：** CI 下界 > shuffle 97.5% 分位，且 paired-vs-V CI 不含 0，且 paired-vs-A CI 不含 0。

**门：**
- 复现门：臂 A 在 pivotal_cwgt 上 13.1% 路由召回的 CI 含 13.1%；ρ_t 的 T3 与 [0.647, 0.793] 相容；HF–vLLM 秩序一致（已过）；M5 同量级。
- 标签门：B1 两库一致率 0.902；|B3| = 170 ≥ 150；B3 按 §1 的复现率规则处理。

**读法：** R-LADDER / R-MODEL / R-WORLD / R-REPLAY-ONLY / R-NULL，条件见 CLAUDE.md §8。
- **优先级（FREEZE ADDENDUM §3 第 6 项）：** R-MODEL 与 R-LADDER 都满足时，算模型侧多特征与 W1-pure 多特征在 B2 上的 paired 差值 CI：
  - 模型显著更高 → R-MODEL；
  - W1 显著更高 → R-LADDER；
  - CI 含 0 → **R-EITHER**（论文以 REV 更低者为主叙事）。
- 其余读法互斥，按表序取第一个满足者。
- "模型侧多特征" = B+C 联合模型（§4；单一预设分数，不在 B 与 C 之间按结果取最大）。

**预注册预测（P1–P4 中的 B2 一律指 Fork / het；逐条对账，错了照写）：**
- **P1：** `entropy_q` 在 B1 上过线，在 B2 上归一化效应 CI 含 0。
- **P2：** `target_revealed` 在 B2 上的归一化效应 ≥ W1-pure 多特征的 50%。
- **P3：** W1-pure 多特征在 B2 上的归一化效应 ≥ W1-model 多特征的 50%。
- **P4：** 从 W1-pure 多特征中去掉 34–36 后，B2 上的 AUROC 下降，paired 差值 CI 不含 0。

**已知事实 F1（不是预测）：** 64.2% [59.2, 69.0] 的分叉（Fork）在 chosen-vs-mean 信用下不可见（Masked 305 / 475，D-042）。

**保留报告项：** M2、M3、M4（D 与 𝓛 各一次）、V 偏回归、M1 / M5（CWGT 对比）、CARL δ（`crit_sd`）、T-ref、零质量界、三域表（含零假设期望计数）。

## 6. E1：fork 预算分配（CLAUDE.md §6.6）

- 每轨迹选 k ∈ {1, 2, 4, 8} 个点。轨迹点数不足 k 时取全部。
- **选择器（只用 forward 量）：**
  - uniform：按轨迹内进度等间隔取 k 点；
  - random：每轨迹 200 次重复取均值；
  - entropy-top-k：按 `entropy_q`；
  - B-top-k：按 B 多特征在 B1 上的 out-of-fold P(Live)；
  - B→W1-pure：先取 P(Live) 最高的 2k 个点，再按 W1-pure 多特征在 B2 上的 out-of-fold 分数取前 k；
  - oracle：gold / 库 3–6 标签。
- **评估：**
  - 选中集覆盖的 Fork 与 Credit-pivotal 比例；
  - 选中集上，用库 1+2（每臂 3 次子抽样）估计的 A 与库 3–6 的 A 之间的 Pearson；
  - 每轨迹捕获的 |A_3456| 总量 / k。
  - 统计：轨迹层面 paired bootstrap。
  - 横轴：§7 的成本向量。
- **`exploratory_adaptive_replay`（FREEZE ADDENDUM §4，不进主表、不进判定）：**
  - 离线自适应回放：每点先从库 1–2 取每臂 1 次，之后按轮次从库 1–4 依次追加。追加规则：当前续跑池全同的点停止追加；其余点按"五臂 G 统计量"降序优先追加，直到用完同等续跑预算。
  - 标签用库 5–6。

## 7. 成本向量（每臂，写入 `audit_metadata`；部署口径与实验口径并报）

成本向量 = (环境步, 模型前向 token, LLM 调用数, 回放续跑数)，按每点平均计。
- **A、B：** 前向 token = 候选打分的 prefix 加命令 token。
- **C、V：** 一次前向。
- **W1-pure：** 部署口径为每点 5 个环境步（执行时 fork）；实验口径另报重放步数（reset + 前缀重放 + 1 步）。
- **W1-model：** 在 W1-pure 之上，加 5 次 peek 态前向；W1-J 再加 5 次 LLM 调用。
- **G：** 7 条兄弟 rollout 的生成步，均摊到点。
- **B_m：** 5m 次续跑。

## 8. W1-J prompt（FREEZE ADDENDUM §3 第 10 项批准；sha256 `9804c516f3fe3d1b6ec9080f9e6a4e15cd09e31e9062dcfa51dd9c7a27e233b1`）

```
You are evaluating one step of an agent that is carrying out a text-based task.

Task: {task}

Observation before the step:
{obs_before}

Action taken: {action}

Environment response:
{peek_obs}

Judge only this step. Compared with the situation before the action, does the environment response show that the agent got closer to completing the task (progress), stayed about as far from completing it as before (neutral), or got farther from completing it (regress)?
Answer with exactly one word: progress, neutral, or regress.
```

调用设置：冻结的 Qwen2.5-7B-Instruct，chat template，温度 0，max_tokens 4。首词不在 {progress, neutral, regress} 内的记 `unparsed`，计数报告；计算比例时不计入分母。模板文件为 `prompts/w1_judge_template.txt`。

## 9. L3（可选，不进判定）

按 CLAUDE.md §6.4 原样执行。选择器只换成"臂 B 多特征在 B3 上的 out-of-fold P(A < 0) 最大者"。

---

## 确认记录

- 草稿十项全部按 `RULING_2026-10-26_FREEZE_ADDENDUM.md` §3 处理：第 1–5、7–10 项确认；第 6 项（读法优先级）改为 R-EITHER 规则。
- §1 标签与 §5 预测表按补充裁决 §1 重写。
- 起草时补写、未在草稿中出现的实现细节，均为确定性口径，不涉及任何结局：
  - §3 "特征计算细节"；
  - §4 的求解器与默认划分；
  - §5 的"模型侧多特征"取法；
  - §6 的 k 不足处理与 `exploratory_adaptive_replay` 的追加规则。

  记录于 D-044。
