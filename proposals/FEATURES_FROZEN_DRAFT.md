# FEATURES_FROZEN — **DRAFT（不是冻结文件；研究者确认后复制到根目录 `FEATURES_FROZEN.md` 并单独提交才算冻结）**

> - **依据：** `CLAUDE.md` @ 602b23f（FINAL RULING 2026-10-26 已写入），`RULING_2026-10-26_FINAL.md` §3、§5、§6.5。
> - **提交方式：** 研究者确认后，本文件单独提交 `FEATURES_FROZEN.md: freeze`，commit hash 记入 DECISIONS。此后目标定义、特征清单、判定规则不再变动。唯二例外：§6.3a 的 gold 机械切换；`exploratory_` 前缀的追加，单独报告。
> - **冻结前状态：** 截至起草，没有计算过任何特征与任何结局的关联。已知的只有标签自身的统计（D-031、D-035、D-039、D-042）。其中 D-035 / D-042 的"按进度分层"一项，本身就是 `turn_index` / `budget_left` 与标签的关联，所以这两列只作控制列，不作检验对象（见 §3 注）。
> - **标 ⚠ 的条目是起草时的提议，需研究者确认。**

---

## 1. 决策点与标签（CLAUDE.md §6.0，原样冻结）

**集合：**
- D = cf_defined（库 1），n = 3,038。
- Live(𝓛) = 库 1+2 合并 30 次续跑池非常量，n = 1,478。Dead = D \ 𝓛。

**标签：**
- **Pivotal_agg（主）：** 库 3–6，factual(12) vs 替代池(48)，Newcombe（method 10 hybrid score）90% 区间不含 0；n = 229。
  - 敏感性规则：|A_3456| ≥ 0.25 且 A_34、A_56 同号非零；n = 193。
- **Pivotal_het：** 五臂 G 检验，置换 10,000 次，α = 0.10；n = 475（D-042）。
- **B3 标签：** sign(A_3456)，在 Pivotal_agg 内。
- **L2-sign：** 𝓛 \ {A_3456 = 0} 上的 sign(A_3456)，只报告。
- **CWGT 复现标签：** pivotal_cwgt = 1[库 1 A_replay ≠ 0]，只用于复现门与 M1。

**Gold 机械切换：** 若 G-candidate 上 Pivotal_agg 的精确率 < 0.70，全文改用敏感性规则（CLAUDE.md §6.3a）。
- 精确率口径：32/臂合并后，90% 区间仍不含 0 的比例。
- ⚠ 同时报告"只用 gold 新增的 20 次独立续跑"的精确率（D-042 flag 2）。切换只按前一口径执行，除非研究者另行裁决。

## 2. 打分与构造口径（不变；后端 D-041）

- **主口径：** HF fp32 + 共享前缀 KV cache（`data/scores_fp32.parquet`、`data/scores_chosen_fp32.parquet`）。vLLM bf16 打分只作敏感性行。
- **候选集：** 𝒜_t = 合法动作 ∪ {chosen} \ {help}（`scoring.candidate_set`，D-021）。q_t = 𝒜_t 上 logp_sum 的 softmax。
- **臂 A：** conf_t = chosen 动作命令 token 的平均 logπ。
  - context = chat(prompt) 加 generation prompt；teacher-force `<action>{a}</action>`，只计 `{a}` 的 token（D-003）。
  - 主分数 s = −conf。敏感性行：+conf；emptythink 口径；realized 口径只报告一行（D-020）。
- **去任务 prompt：** `scoring.detask`，删除任务模板行，以及观察中所有任务陈述的副本。
- **臂 H1：** q_t = exp(Σ logp_cmd / (5.0 · n_tokens))，在 hindsight pass 上计算；hindsight pass = prompt 加一行 `Final outcome: … Final observation: {final_obs[:500]}`。
  - ρ_t = clip(q_t / 同轨迹全部已评分 turn 的 q 均值, 0.8, 1.2)。
- **`chosen_freq`：** log[(c + 0.5) / (32 + 0.5·|𝒜_t|)]，c = 32 份首批草稿中解析为 chosen 的数量（D-008）。
- **`entropy_sampled`：** 库 1 草稿中合法且不重复动作的频率熵（CLAUDE.md §4 特征 16）。
- **W1：** 取自 `data/peek_alts/`（D-029）。五个 peek = factual + 库 1 的 4 个替代。
- **peek 态：** 隐状态取自 `hidden_peek/`（bf16，prompt_end），打分取自 `data/scores_peek_fp32.parquet`。
- **G：** 取自 `data/siblings/`（每任务 7 条，D-043）。状态匹配 = 当前观察文本完全相同。
  - ⚠ 匹配范围：兄弟轨迹的任一 turn 均可匹配，不要求 turn 相同。

## 3. 特征清单与先验符号

方向约定：
- "+" = 值越大，越可能是标签 1。B1 的标签 1 = Live；B2 的标签 1 = Pivotal；B3 的标签 1 = factual 更好（A > 0）。
- "0" = 无先验，报告原始 AUROC，不翻转。
- "C" = 控制列，进入多特征模型，但不作单特征检验。

⚠ 下表所有先验符号都是提议。

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
| 15 | `turn_index`、`n_admissible`、`chosen_n_tokens`、`type(chosen)` | 控制 | forward | C | C | C | `turn_index` 与标签的关联已在 D-035/D-042 中出现 |
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
- ⚠ 规则：任务描述转小写、去掉句末句号后，记 D = `(?:a |an |some |two |the )?`，依次匹配下面四个正则，取第一个命中的捕获组 X（单个词）：
  1. `^(?:clean|heat|cool) D(\w+) and put`
  2. `^find D(\w+) and put`
  3. `^put D(?:clean |hot |cool |cold )?(\w+) (?:in|on|into)\b`
  4. `^(?:look at|examine) D(\w+) (?:under|with)\b`
- 在 100 个任务上 100/100 抽中，无代词或形容词残留（核查只读任务文本，不涉及标签）。
- 任一 peek 观察匹配 `\bX \d+\b`（ALFWorld 物体带编号，例如 "ladle 1"；词边界避免 "pot" 误中 "potato"、"knife" 误中 "butterknife"）即记 1。抽不出时记 NaN，并报告计数。

**缺失值：**
- 单特征 AUROC 只在非缺失点上计算，并报告 n。
- 多特征模型在训练折内，用中位数插补并加缺失指示列。
- 会出现缺失的特征：`js_prev`（t = 1）、`target_revealed`、`g_succ_state`、`g_succ_alt_sd`。

## 4. 臂与学习型分数

**通用设置：**
- 所有学习型分数取 GroupKFold（按轨迹，5 折）的 out-of-fold 预测。
- 分类用 L2 logistic，连续目标用 ridge。
- 正则系数只调一个：内层按轨迹做 5 折 CV，网格 λ ∈ 10^{−4, −3, …, 4}，按 AUROC 选（ridge 按 MSE 选）。
- 特征在训练折内标准化。
- ⚠ 每个多特征模型都包含控制列 `turn_index`、`n_admissible`，确保增益不只来自位置。

| 臂 / 分数 | 输入 | 边界 |
|---|---|---|
| A | −conf（单分数） | B1/B2/B3/L2-sign |
| H1 | ρ_t（单分数） | 同上 |
| `chosen_freq` | 单分数（独立基线行） | 同上 |
| V | `turn_index`、`n_admissible` + 臂 C 位置 (i) 隐状态；层由内层 CV 在 {14, 21, 26} 中选；目标 = 续跑池平均成功率（库 1、库 2 各一次，交叉拟合）；ridge | B1/B2/B3（out-of-fold 预测作独立对手与偏回归协变量） |
| B 多特征 | 1–10、12–19（forward 版）；另报 + 11（hindsight 版） | 每条边界一个模型 |
| B 家族 | F1–F5 各自 family-only + 控制列；全模型 leave-one-family-out | 指认规则（CLAUDE.md §4） |
| C | 每层 × 每位置一个 probe（6 个） | 每条边界；报告每个 probe 与最佳层 |
| ⚠ W1-pure 多特征 | 20–28 + 34–36 | B1/B2/B3 |
| ⚠ W1-model 多特征 | 20、29–33 | 同上 |
| G 多特征 | 37–40（缺失按上节处理） | 同上 |
| B+C | B 多特征 + C 最佳层 probe 的 out-of-fold 输出 | 同上 |
| C→B 分解 | C 最佳层输出对 B 特征的线性回归 R² | — |

**W1-V（特征 29、30）：**
- 用 V 的同一模型，作用在 peek 态隐状态上。
- 对点 t，所用的 V 模型来自不含 t 所在轨迹的训练折（与 V 的 out-of-fold 预测同折）。
- 层与 V 选中的层相同。

## 5. 评估（CLAUDE.md §6.3，原样冻结）

**每条边界、每个分数报告：**
- AUROC，以及按轨迹 cluster bootstrap（10,000 次）的 95% CI；
- 分层 shuffle（1,000 次；B2/B3 按轨迹内进度三分位分层，B1 不分层）；
- 对 V、对臂 A 的 paired 差值 CI；
- 归一化效应 = (AUROC − 0.5)/(天花板 − 0.5)，天花板为库 1+2 回放（m = 6；B1 用 m = 3）；
- REV（B_m 取库 1+2 子抽样 200 次的均值，m ∈ {1, 2, 3, 6}）。

**过线：** CI 下界 > shuffle 97.5% 分位，且 paired-vs-V CI 不含 0，且 paired-vs-A CI 不含 0。

**判定：** CLAUDE.md §8（门 + R-LADDER / R-MODEL / R-WORLD / R-REPLAY-ONLY / R-NULL + "(het)" 后缀）。
- ⚠ 多个读法同时成立时，按表中自上而下的顺序取第一个满足者。这条需研究者确认（裁决未写明）。

**预注册预测（逐条对账，错了照写）：**
- **P1：** `entropy_q` 在 B1 上过线，在 B2 上归一化效应 CI 含 0。
- **P2：** `target_revealed` 在 B2 上的归一化效应 ≥ W1-pure 多特征的 50%。
- **P3：** W1-pure 多特征在 B2 上的归一化效应 ≥ W1-model 多特征的 50%。
- **P4：** 从 W1-pure 多特征中去掉 34–36 后，B2 上的 AUROC 下降，paired 差值 CI 不含 0。
- **P5：** 被掩盖的分叉比例 / |Pivotal_het| ≥ 0.15。
  - **注：** P5 是纯标签量，冻结前已算出（0.642，D-042），所以它是已知事实，不是前向预测。

**保留报告项：** M2、M3、M4（D 与 𝓛 各一次）、V 偏回归、M1 / M5（CWGT 对比）、CARL δ（`crit_sd`）。

## 6. E1：fork 预算分配（CLAUDE.md §6.6，原样冻结）

- 每轨迹选 k ∈ {1, 2, 4, 8} 个点。
- **选择器（只用 forward 量）：**
  - uniform：⚠ 按轨迹内进度等间隔取 k 点；
  - random：每轨迹 200 次重复取均值；
  - entropy-top-k：按 `entropy_q`；
  - B-top-k：按 B 多特征在 B1 上的 out-of-fold P(Live)；
  - B→W1-pure：⚠ 先取 P(Live) 最高的 2k 个点，再按 W1-pure 多特征在 B2 上的 out-of-fold 分数取前 k；
  - oracle：gold / 库 3–6 标签。
- **评估：**
  - 选中集覆盖的 Pivotal_agg 与 Pivotal_het 比例；
  - 选中集上，用库 1+2（每臂 3 次子抽样）估计的 A 与库 3–6 的 A 之间的 Pearson；
  - 每轨迹捕获的 |A_3456| 总量 / k。
  - 统计：轨迹层面 paired bootstrap。
  - 横轴：§7 的成本向量。

## 7. 成本向量（每臂，写入 `audit_metadata`）

成本向量 = (环境步, 模型前向 token, LLM 调用数, 回放续跑数)，按每点平均计。
- **A、B：** 前向 token = 候选打分的 prefix 加命令 token。
- **C、V：** 一次前向。
- **W1-pure：** 每点 5 个 peek，每个 = 1 次 reset + 前缀重放 + 1 步。
  - ⚠ 部署口径只计 5 个环境步（执行时 fork），实验口径另报重放步数。
- **W1-model：** 在 W1-pure 之上，加 5 次 peek 态前向；W1-J 再加 5 次 LLM 调用。
- **G：** 7 条兄弟 rollout 的生成步，均摊到点。
- **B_m：** 5m 次续跑。

## 8. W1-J prompt（原文；sha256 `9804c516f3fe3d1b6ec9080f9e6a4e15cd09e31e9062dcfa51dd9c7a27e233b1`）

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

调用设置：冻结的 Qwen2.5-7B-Instruct，chat template，温度 0，max_tokens 4。首词不在 {progress, neutral, regress} 内的记 `unparsed`，计数报告；计算比例时不计入分母。

## 9. L3（可选，不进判定）

按 CLAUDE.md §6.4 原样执行。选择器只换成"臂 B 多特征在 B3 上的 out-of-fold P(A < 0) 最大者"。

---

## 待研究者确认的条目（⚠ 汇总）

1. §3 全部先验符号。
2. 特征 26 的目标词正则。
3. G 匹配是否不要求 turn 相同。
4. 多特征模型是否都带 `turn_index` / `n_admissible`。
5. W1-pure / W1-model 多特征的组成（交互项归入 pure）。
6. 读法优先级。
7. gold 精确率同时报告独立 20 次口径。
8. E1 的 uniform 定义与 B→W1 的 2k 预筛。
9. 成本向量的部署口径。
10. W1-J prompt 原文（批准后写入 `logs/W1J_APPROVED`）。
