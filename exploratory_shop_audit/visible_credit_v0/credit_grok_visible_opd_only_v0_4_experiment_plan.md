# credit-grok 实验方案 v0.4：只做 on-policy 特权蒸馏与学生可见投影（VC-OPD）

**2026-11-20 更新：执行参数与判定以 §16 为准。**

**用途：交给服务器 agent 的可执行实验协议。日期：2026-11-20。版本：v0.4（生效版本，替代 v0.3；新增特权包有用性预检）。**

> 本版取代 `credit_grok_visible_opsd_v0_experiment_plan.md` 中的 **GRPO 训练部分**，保留其特权老师、学生候选 token、中心化相对评分、rank=8 低秩双线性投影器、按题跨折诊断及主要工程审计。**本轮不运行 GRPO，不使用任何环境 reward/advantage 更新参数。**
>
> **v0.4 生效约束**：保留 `T_cal≥512×1`、埋信号功效检查、rank=8 主投影器、动作级容量参照、ASIN 隔离、在线交叉拟合；**取消按 PASS 类别跳过 token 的硬门控和 R-masked**。P 使用按类别 OOF 校准收缩 `b_c·hat_d`，R-β（按类别冻结 β）替代 R-masked，保留 P-shuffled；R/P/R-β/P-shuffled 均在相同的**学生可见结构性有效 response 位置规则**下训练。§12–§15 是补充与变更记录；正文已同步替换，不再执行 v0.2 旧 mask 规则。**新增 P-pre 特权包预检：在 `T_pre` 上比较无包、`z_full/z_product/z_evidence/z_placeholder` 的行为和 token 信号；仅选定一个通过门槛的包，才允许进入 `T_cal` 大规模离线投影。**
>
> 此处将 **OPD** 定义为“学生使用自己的 on-policy 轨迹、在学生可见输入上拟合特权老师的下一 token 分布”；由于老师与学生由同一更新前模型快照构造，亦可称为同模特权 **OPSD** 的一种实现。为控制成本、保持 Raw/Projected 可比，**本版主要蒸馏的是学生候选集上的条件分布，而非全词表 KL**。这一限定必须贯穿代码、图表和报告，不得将候选集实验称为完整 OPD 的效果。

## 0. 边界与仓库预检

- 目标仓库：`melancholiac99/ForkAudit`，`credit-grok` 分支。此前方案记录的提交是 `9eb97150496f1e716df6fbee0909e9bc170ba69a`；实际执行前记录服务器 `git rev-parse HEAD`、工作树改动、模型和 LoRA checkpoint SHA，不能假定服务器仍停在该提交。
- 可复用代码锚点：`exploratory_shop_audit/grpo_forced_audit/{train.py,native.py,worker_hooks.py,rollout.py,interface.py,RUNBOOK.md}`。**只复用 rollout、prompt/interface、环境交互、模型加载、分布式数据流及评测基础设施；不得复用 GRPO/PPO loss、`compute_advantage`、group reward 归一化或 PPO clipping 来实现本实验。**
- 原有基线条件供冻结配置参考：`verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8`，Qwen2.5-7B-Instruct，LoRA rank/alpha=64/64，4 GPU，原配置每题 8 rollout、每更新 16 题、最多 35 环境动作、每次生成不超过 512 token。是否能以相同吞吐执行蒸馏，须在服务器实际计时，不得照搬 GRPO 耗时。
- 环境：`YYHDBL/shopping-grpo-longhorizon@13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5`。不改任务、观察模板、动作是否合法、终止条件、原生 Reward v3 或验证任务集合。**Reward v3 只用于评测，不进入训练 loss、数据筛选或样本权重。**
- 旧 GRPO 100 更新成绩、E1 选臂结果以及 ALFWorld 的 `exploratory_credit_projection/` 只可作工程背景，不是本轮控制组，也不代表购物 token 投影效果。旧 BF16/vLLM 概率不一致、checkpoint 恢复及 GPU 时钟账本问题必须重新检查。
- 不覆盖旧目录。新代码建议放 `exploratory_shop_audit/visible_opd_v0/`，运行结果放 `.runtime/runs/shop_visible_opd_v0_<run_id>/`；Git 只提交代码、配置、摘要、hash 和不含任务答案的报告。

## 1. 本轮要回答什么

主问题：

1. 在 **不使用 GRPO、只使用 OPD** 的情况下，同模特权老师是否能给当前无特权学生提供有效的训练信号，带来新任务、**未见过 gold 商品**上的行为收益？
2. 使用学生可见信息预测的 token 相对修正，是否比直接使用特权老师的相对修正更有利于学生泛化？在**相同训练位置**下，收益是否仅因**蒸馏强度较弱或随机扰动**？
3. 原样保留的 **rank=8 低秩双线性投影器** 是否足够；若失败，是离线功效不足、表示/函数类容量限制，还是在当前特征下没有检测到稳定的跨题可预测信号？

严格区分六层证据：（0）同模特权老师是否实际利用特权包改善行为，且 token 偏移区别于占位模板；（a）Raw 老师是否改变候选 token 相对分布；（b）离线试验是否**有功效检测已知可见信号**；（c）真实老师修正能否从学生表示题外预测，以及更强的**动作级只读参照**能否预测；（d）Gold 与满足全部要求的等价商品之间的偏好是否被投影削弱；（e）蒸馏后学生在无特权、未见商品上的 Reward v3、严格成功率及**满足全部要求的购买率**是否改善。

本轮**保留原版 rank=8 主模型，不引入 MLP、rank 扩大或独立 Critic**。动作级满秩岭回归只做诊断，不蒸馏、不训练 actor；此举不构成修改原 OPD 方法。

不做：GRPO、PPO、REINFORCE、环境奖励加权、oracle 动作强制训练、SFT 参考答案、额外大模型、人工中间奖励、改原生 reward、事后根据 V 调参；也不把当前候选集 KL 冒充完整词表 KL。

## 2. 数据与隔离：任务、gold ASIN 与离线功效

保持原始 train `T`、开发验证 `V`、blind/final 的边界。执行前用 `goal_index`、规范化 task content hash 去重，并**仅在数据划分/评测审计侧**读取冻结目标的 gold ASIN。不得将这些 gold 字段传入 actor、可见特征、投影器输入，亦不得用 V 的 gold 信息生成老师评分包。

从 `T` 事先冻结：

- `T_pre`：**特权包有用性预检任务池**，建议约 100 个不同任务，尽可能与 `T_cal/T_opd` 的任务及 gold ASIN 不重叠；预先抽取并冻结，不许按结果补换任务。五个包/无包组用**相同的 `T_pre`、冻结模型快照及对齐的随机种子方案**；只在此池选定特权包，绝不用 V。若 train 规模使 ASIN 严格隔离不可行，先报告交叉和风险。`T_pre` 不得用来训练投影器、actor 或拟合 `b_c/β_c/γ_c`。
- `T_cal`：**不少于 512 个不同任务，每题 1 条学生原生 rollout**；覆盖页面/动作类别。先不以每题重复 rollout 代替新任务。需要追加样本以提高功效时，必须先报告 `UNDERPOWERED` 和新增成本，获得明确授权后按预注册方式追加新的**不同任务**；不得看 V 结果决定采样。
- `T_opd`：actor 的 on-policy 训练池，与 `T_cal` 在 task ID/content hash 上不交叉；**优先按 gold ASIN 分组**使两池商品不重叠，若训练池规模不足而无法做到，记录冲突数和原因。
- `V`：冻结的开发验证集，**actor 行为评测始终无特权**，绝不用于 projector 拟合、校准斜率/β/γ、模型选择或超参调整；blind/final 不接触。**唯一例外**：§7 的“等价商品偏好”可在所有训练与判定规则冻结之后，使用独立隔离的只读审计器对 V 做一次特权评分。该审计的 `z`、输出、V gold 不得进入训练端、投影器刷新或决策门；审计不能用于选 run。若无法保证物理/权限隔离，V 上不做该审计，只报告 `T_cal` OOF 与预先留出的 train-only、ASIN-disjoint 审计池，不冒称 V 结果。
- `T_opd_eval`：在训练启动前固定，从 `T_opd` 抽取一份代表性任务列表，仅用于与 V 同检查点、同采样协议的**无特权**行为评测；训练本身可使用这些任务，评测 reward 不回传训练。

**ASIN 分组与分层（强制记录）**：

1. 保存 `T_pre/T_cal/T_opd/V` 之间 gold ASIN 的 pairwise 重合量和比例：任务比例、去重 ASIN 比例，均注明分母。还要分别记录 `V` 的 gold ASIN 是否出现在 `T_cal`、`T_opd` 以及其并集。
2. 优先让 `T_cal/T_opd` 与训练相关开发评测商品按 ASIN 分组隔离，**不得偷偷修改既有 V 定义**。如果原 V 与 T 共享商品，只能将 V 固定切成 `seen` 与 `unseen` 子层；主泛化结论依据 **`unseen = gold ASIN 不在 T_cal ∪ T_opd`**。另外报告相对于仅 `T_opd` 的 seen/unseen 作为补充。
3. 若 V 的 `unseen` 样本过少，必须标 `UNIDENTIFIABLE/INSUFFICIENT_UNSEEN`，不许拿 seen 层结果替代新商品泛化；可以在训练前从 T 另预留 **ASIN-disjoint 的内部测试池**，但须明确不是原始 V 或 blind，并报告其与模型选择的隔离。
4. `T_cal` 的 4-fold **优先按 gold ASIN 分组**，同 ASIN 下所有题、rollout、token 不跨 fold。也保存按 task 分组的辅助指标以对照旧设定；若 ASIN 分组不可行要登记及解释。所有标准化和容量诊断的超参选择都限训练 folds。

**训练—新任务差距**：B/R/P/R-β/P-shuffled 在相同检查点上分别评估 `T_opd_eval` 与冻结 V（特别是 V-unseen），报告 `Gap = Reward(T_opd_eval) − Reward(V)`，以及相对 B 的 gap 变化，严格成功率也同报。配对 bootstrap 对任务及重复商品做簇聚合。这个 gap 是**记忆/分布差异的风险读数**，会受任务难度与分布影响，**不是特权泄漏的单独因果证明**。

`T_cal` 的 OOF 仅供离线诊断；actor 的在线 `T_opd` 预测来自不包含当前任务、尽可能也不包含当前 ASIN 的校准模型，不得把在线预测叫作 `T_cal` 的 OOF。

## 3. 特权老师与原始 token 修正：保持原方法

每次 actor 更新前冻结学生快照 `bar_theta_k`。学生只凭可见历史 `h_{i,t}`（题目、已收到的环境反馈和已生成前缀）采样轨迹；老师用**同一权重、同一 tokenizer、同一输出 token 前缀**做 teacher-forced 评分：

- `student`：`pi_{bar_theta_k}(v | h_{i,t})`，无特权。
- `privileged`：`pi_{bar_theta_k}(v | h_{i,t}, z_i)`，仅评分端有特权包 `z_i`。

训练用特权包 **先通过 §3.1 的 P-pre 预检选定一种模板与数据来源，再冻结为 `z_selected`**；默认候选 `z_full` 沿用 `gold ASIN`、`goal_options` 和可核验约束；`z_product` 包含 z_full 全部字段（含 gold ASIN 与价格约束），另取冻结 train 商品目录中的 gold 商品标题与关键属性；`z_evidence` 仅使用**当前时刻**真实结果页候选商品的属性/规格且不标注 gold。不得加入未来观察、实际 rollout 结果、E0/E1 oracle 或 V/final 答案。证据包上限 1024 tokens，其余真实包上限 192 tokens；各模式均报告实际字段数、token 数、截断/溢出率；信息超过上限时按预先冻结规则记 `PACKET_OVERFLOW`/不可用，不能暗中省略困难商品或为了择优临时扩上下文。**`teacher_packet_manifest.json` 必须逐字段列出 `field_name/source/already_visible_to_student(yes|partial|no)/token_count`，并报告各字段 token 数及总长度分布**，检验老师额外拿到的究竟是什么（预期 gold ASIN 是主要差异，但不得先验断言）。不能为加入 `z` 删改原学生历史；超上下文记 NA 并分层报告覆盖。

在**任何 GPU 预检前**冻结全部候选包的角色、插入位置、文字模板、占位模板、分隔符、tokenizer 和模板哈希，写入 `teacher_packet_manifest.json`；例如将 `z` 作为“完整购物要求”消息之后的独立固定格式段落，但最终必须依据真实 prompt 验证。P-pre 只能在这些**事前固定**的候选中做一次选择，不准根据结果改写包内容。`z=空` 时**整个额外段落完全不出现**（不能保留空 system 段），保证 student 与 empty-z 的输入 token IDs 完全相同。不得在运行中为改善信号改模板。V 的一次性只读审计例外按 §2 的隔离规定执行，不产生训练用 packet。

### 3.1 P-pre：特权包有用性预检（在 `T_cal` 评分/OOF 之前；对应讨论中的 §16′）

**动机与停损**：老师提前看到答案或商品证据，并不意味着同一 7B 模型真正用得上这些信息；即使老师自己做题更好，候选集内尤其是 click/规格等**动作 token**的概率也可能几乎不变。必须先核验**行为增益**和**可蒸馏的输出层偏移**，否则不得直接在 `T_cal` 投入 512×1 的投影审计与随后在线训练。此预检**只选输入包，不修改 actor，不改 rank=8、top-8、中心化或 KL 目标**；环境 Reward v3 此时仅作 train-only 行为评测与预注册的一次性包选择，绝不进入任何梯度、轨迹权重或 actor 训练样本筛选。

**五种 teacher-as-policy 设置（约 100 个 `T_pre` 任务，每种 sample×2 + greedy×1）**：

| 设置 | 老师额外获得什么 | 与学生视角的边界 |
|---|---|---|
| `z_none`（学生基线） | 完全无附加段落，原学生原生 prompt | 评测无包行为与 token/logit 基线 |
| `z_full` | 原版包：gold ASIN、冻结目标规格/价格等可核验字段 | 最贴近原 v0.3 设计；字段逐项核对哪些已对学生可见 |
| `z_product` | z_full 全部字段，含 gold ASIN，另加 gold 商品标题与关键属性 | 只来自 train 冻结商品目录；不能从实际未来轨迹提取 |
| `z_evidence` | 当前结果页**各个当前可见候选商品**的属性/规格证据，不带 gold 标记 | 必须在每个真实环境状态重新构建，禁止引用未出现的下一页/未来反馈；与学生已看到的重合部分单列 |
| `z_placeholder` | 与 `z_full` **同角色、同插入位置、同字段和固定文本格式**，内容值全为“未提供” | 控制有无附加段落造成的模板/格式效应；`z_none` 才是严格空包控制 |

这五组共约 `5×100×(2+1)=1,500` 条完整环境轨迹（若各设定均完成），必须先用 8–16 题测吞吐、失败率和 GPU-hour，**并先验证 student/empty-z 前缀逐 token 一致、各模式评分对齐及占位模板正确**，登记预算并获授权；**不可把它称作无需 GPU 的阶段 A**。在相同任务上报告 Reward v3、严格成功率、`all_requirements_purchase_rate`（满足全部要求购买率）、reward_type 全分布、合法动作率和 search/click/规格漏斗；`sample×2` 和 `greedy×1` **分别比较**，不得把三条视为独立任务。统计差值以任务配对/ASIN 簇 bootstrap 给区间；对 `z_full` 额外报告相对同模板 `z_placeholder` 的 teacher-as-policy 行为差，对 `z_product/z_evidence` 仅有评分侧的匹配占位对照，因此其**行为改善相对无包不能完全归因于信息内容**。V/blind/final 不参与筛包。若某包不可从冻结 train 数据或当前可见页可靠构造，应记 `PACKET_UNAVAILABLE` 与覆盖率，不得合成假商品属性。

**学生轨迹上的 teacher-forced 对照（不能只看 teacher 自己的成功率）**：在同一冻结 `z_none` 学生自己生成的轨迹、同一 `h_t` 与同一 `C_t=student top-8 + sampled token` 上，分别以 `z_none/z_full/z_product/z_evidence/z_placeholder` 做评分。对每包 `x` 记 `q_x=softmax_C(s+d_x)`；报告各页面与冻结解析器类别上的：

- `KL(q_x||q_S)` 的均值/分位数与非零 token 比例；`mass_S(C_t)`、`mass_x(C_t)`，不能让低候选 mass 假装成老师没作用。
- 实际采样 token 的 `Δ_x(y_t)`、中心化 `d_x(y_t)` 的分布；对 click/规格等**动作内容 token**报告 `|Δ_x|` 与 `|d_x|` 超过**预先按评分精度/空包数值误差登记**的非零阈值 `τ_delta/τ_d` 的比例，以及 token/任务/商品覆盖。
- **格式效应与内容效应分开**：对 `z_full` 以 `z_placeholder` 为成对格式对照，报告 `δ_info,full(v)=d_full(v)−d_placeholder(v)`、`KL(q_full||q_placeholder)` 及 `KL(q_full||q_S)−KL(q_placeholder||q_S)`（后者是配对指标差，**不是** KL 的代数分解）。`z_none` 与 `z_placeholder` 的差刻画**附加段落本身**的影响。对 `z_product/z_evidence`，除报告其相对公共 `z_placeholder` 的对照外，还须在**评分侧**各建立同结构、同插入点、对应字段都为“未提供”的 `z_placeholder_product/z_placeholder_evidence`，以对应模板成对差隔离内容效应；它们**不新增 teacher-as-policy 的主五组**。必须报告 placeholder 和真实包的长度差，不能声称占位消除了所有 token 长度/注意力影响。
- `z_evidence` 对应每个决策点的**当前可见结果页**构造；若该位置不是结果页/无可用候选，就记录 NA 与覆盖率，不得用未来页或后台 gold 代填。相同原始学生轨迹上所有组评分类别和位置须对齐，失败统一记 `SCORER_FAILURE/PACKET_UNAVAILABLE`。

**一次性择包与停损（在看任何 P1/在线结果前冻结算法、阈值）**：预注册动作内容 `τ_delta/τ_d`、候选 mass 有效范围、最低可评估动作位置/不同任务数（由阶段 A 的资源与标签审计定值），并按如下规则执行：

1. 对每种真实包 `x∈{full,product,evidence}`，在同一 `T_pre` 上相对 `z_none` 计算**满足全部要求购买率的正改善**（主用 `sample×2` 的按任务平均；greedy 另报），同时要求 **其动作 token 在真实内容与**`x` **对应模板的占位版本之间，有超过冻结数值阈值的非零分布偏移**（`δ_info,x`/`KL(q_x||q_placeholder_x)` 以及可评价位置比例）。预先冻结并报告改善点估计、任务簇区间和达到非零阈值的任务数；若样本过少、区间太宽，记 `PREFLIGHT_INCONCLUSIVE`，不能写成没有特权效应。
2. 只有满足上述两项者进入候选，取**全要求购买率相对 `z_none` 提升最大**的一个；平局按事先登记的固定顺序 `z_full > z_product > z_evidence`（此顺序只是可复现的工程裁决，不是结果推断）。只允许用 `T_pre` 选**一次**；不允许用 `T_cal` 的 OOF、V、在线曲线重新挑包、改提示词或根据 test 选模型。
3. 若三种真实包都没有同时达到基本行为改善与动作 token 内容偏移，记 `PRIVILEGE_NOT_USEFUL`，**停止 v0 的正式 `T_cal` OOF 与 R/P/R-β/P-shuffled 在线训练**，只提交预检报告和实际成本；若仅仅缺数据/功效则记 `PREFLIGHT_INCONCLUSIVE` 并停在签收点，不以“失败”替代“未检出”。
4. 选定模式、模板 SHA、字段来源/可见度、token 长度分布、`z_none`/公共 placeholder/匹配 placeholder 的指标、差值/区间、`T_pre` task/ASIN hash 存入 `packet_selection.json`，从此所有 R、P、R-β、P-shuffled 的 `T_cal` 监督和在线评分采用**同一种 `z_selected`**。原 `z_full` 若未中选，不得偷偷继续当主方法老师；所有 `z_selected` 都不得在 actor 输入或 student-only 投影器特征中出现。**占位相减只作为预检诊断，不擅自把主训练的 `d_x` 换成 `d_x−d_placeholder`**，否则属于改算法。

**研究问题边界**：若选中 `z_evidence`，应检查其中的证据是否其实已完整显示在学生观察中；若是，研究内容转向“可见材料的额外整理/提示”，不能仍把它说成按 gold ID 过滤题目特有特权。若只在 teacher-as-policy 上有益、token 偏移仍为零，也不能用一个看起来成功的老师推断 OPSD 蒸馏可行。

评分温度 `T_score=1`：

```text
s_t(v)      = student logits(v | h_t)
p_t(v)      = privileged logits(v | h_t,z)
Delta_t(v)  = log pi_priv(v | h_t,z) - log pi_student(v | h_t)
C_t         = student top-8 + actual sampled y_t   # 去重后最多9个
q_S_t(v)    = softmax_{v in C_t}(s_t(v))
d_t(v)     = Delta_t(v) - sum_{u in C_t} q_S_t(u) * Delta_t(u)
```

老师**不能**提名候选；保留整个候选集合的 `d_t(C_t)`，不只采样 token 的分数。中心化去除共同 logit 偏移，**不消除老师携带的答案信息**。数值异常、前缀错位和超长文一律 NA，不能默认置 0。

### 候选集质量

在同一评分前向中计算、仅保存聚合统计：

```text
mass_S(C_t) = sum_{v in C_t} pi_student(v | h_t)
mass_T(C_t) = sum_{v in C_t} pi_priv(v | h_t,z)
```

如果老师重视的 token 不在 student top-8，候选集条件 KL 无法蒸馏其**集合外概率质量**；低覆盖不能当作 OPD 或投影无效的证据。

### token 分类在执行前冻结

由与真实 rollout/interface 对齐的解析器将 response token 唯一归入：**思考文本 / search 内容 / click 内容 / 规格值 / 格式符号**（若无法判定则 `unknown`，单独计数，禁止静默丢弃）。必须规定 `Action:` 边界、`search[...]` 查询跨度、`click[...]` ASIN 子词、规格键值与特殊符号如何处理；分类解析器、版本和测试例在 P0 之前冻结。

所有 OOF、功效、候选 mass、target KL、有效 mask 和 reward 漏斗都按类别及页面类型报告。**思考文本**特别报告 R/P 对答案、ASIN 等字符串的目标偏移和可见性；若老师在推理文本中直接说出答案，标记为高泄漏风险，不把格式/思考预测成功冒充动作信用成功。

## 4. 学生可见投影：保留 rank=8，补功效与容量参照

同一 student-only 冻结模型前向提取最后一层 `r_t`，并取学生 LM head 在候选 token 上的行向量 `e_v`。主投影器结构**完全不变**：

```text
f_phi(r_t,v) = (U^T norm(r_t)) dot (V^T norm(e_v))
rank(U,V) = 8    # 本轮固定，不做大 rank、MLP 或 Critic 替换
hat_d_t(v) = f_phi(r_t,v) - sum_{u in C_t} q_S_t(u) f_phi(r_t,u)
```

在 `T_cal` 上用候选集 `d_t(v)` 的鲁棒平方损失拟合 `U,V`，任务/轨迹/token 权重防止长轨迹和格式 token 主导。训练配置、损失、标准化、优化器、早停和随机 seed 必须在看 V 前冻结。特权 `z`、goal ID、gold/后台字段、老师 hidden 不能作预测器特征。

### 4.1 真正的 OOF 可见性诊断（检验非零信号，不追求高解释比例）

按 **4-fold ASIN 分组（优先）/任务分组（备份且注明限制）** 交叉拟合，同一任务的所有轨迹/位置/候选 token 始终在一折。OOF 用任务簇（另报商品簇）bootstrap 比较 `hat_d` 与恒零、训练折均值、token-only 最强简单基线，分别报：MSE 改进**绝对量/相对量及其区间**、方向一致率（仅诊断）、token/任务覆盖、决策 token 与格式/思考类别，以及页面分层。

**目的不是把 `hat_d` 拟合得与每个单次特权 `d` 很接近，而是检验是否存在稳定非零的学生可见条件平均修正。** 若 `d=g(r,e)+epsilon`，且 `E[epsilon|r,e]=0`，对恒零基线的期望 MSE 改善满足：

```text
E[d^2] - E[(d-hat_d)^2] = 2E[d*hat_d] - E[hat_d^2]
                                  = 2E[g*hat_d] - E[hat_d^2]
```

因此**不设 explained variance/R² 最低比例门**；大残差会降低检验功效/扩大区间，并不会自动使该改善量的期望偏小（前提是跨题残差与预测无关且分布匹配）。`d` 与 `hat_d` 的单次同号率仅报告、不作 P1 通过阈值。

“折间预测符号稳定”定义为：在**同一批未用于拟合的留出 token**上，用不同训练折子集单独拟合的多个预测器，其预测的同号比例；接近零的预测按预先冻结阈值作为 near-zero，另列而不是硬判为符号错误。该统计不等于 `sign(hat_d)==sign(d)`。仅当 P1 功效合格、OOF MSE 改善对最强简单基线的任务簇 CI 下界 >0，且折间稳定性未呈明显崩溃，才记为真实可见性 `PASS`；`PASS` **不再控制哪个 token 参与 OPD**。

### 4.2 埋信号的离线检验功效（强制；不调主架构）

在同一批真实 `T_cal` 冻结 `r_t/e_v/C_t/q_S` 上做**已知真值的可检出性检查**：

1. 在每个页面×token 类别内，将真实 `d` 的**任务级标签对应关系**打乱；打乱单位必须是任务，不能按单 token 造成伪独立。若不同候选长度/位置不能逐行对应，采用按类别匹配的标量标签重配并在接收位置按其 `q_S` 重新中心化；记录无法匹配的比例。打乱不得使用评测 V。
2. 用固定随机种子、**rank≤8 且只依赖 `(r_t,e_v)`** 的已知函数 `g_*(r_t,v)` 在每个候选集合内中心化。基于每折训练数据选取注入系数 `alpha`，使注入成分 `alpha*g_*` 约占合成标签方差的 **50%**；验证折不参与选系数。合成标签 `d_syn=d_shuffle+alpha*g_*`。
3. 仍使用**原 rank=8 投影器和相同 folds/训练流程**拟合 `d_syn`，并与只用 `d_shuffle` 训练的 null 预测器比较。评估真实已知成分的 OOF **回收系数** `recovery = Cov(hat_d_syn - hat_d_null, alpha*g_*) / Var(alpha*g_*)` 以及 bootstrap 区间，并报告 OOF MSE 改进、不同类别的独立任务数。
4. **预注册功效规则**：关键决策类别 `recovery` 点估计 ≥0.50 且任务簇 bootstrap 区间下界 >0，才视为对“可预测份额约 0.5”的假设有基本检出能力。若未达到，P1 该类别记 `UNDERPOWERED`，**不能写成学生信息不可学或投影信号阴性**。不把在可实现 rank≤8 埋信号上成功，误称为对任意复杂可见规律都有功效。

可另外用可见份额 0.25 做敏感性分析，但不在试验后替换 0.50 的事先门槛。若功效不足，先提交成本和增加不同任务数的建议；不得悄悄改 rank/top-k/MLP 直到“过关”。

### 4.3 动作级容量参照（只读诊断，不参与 OPD）

**在同一 `T_cal` 决策点上**登记一个比 rank=8 token 预测更强的比较器：冻结学生，在“**学生可见上下文 + 完整候选动作字符串**”上 teacher-force，提取动作末端/约定聚合的隐藏表示，以**满秩岭回归**预测该完整动作的特权-无特权评分修正，按任务/商品分组 OOF。

- 候选动作只能来自学生可见的当前页面合法动作及**无特权 student-only 提议**；不得用 gold ASIN、环境 oracle 或特权老师构造候选。冻结候选数量、生成/过滤规则、动作长度归一化和表示提取位置，缺候选时报告覆盖。
- 标签建议在**相同候选动作**上计算 `log pi_priv(action|h,z) − log pi_student(action|h)` 的长度归一化值，再按 student 候选相对权重中心化；必须记录该标签与 token 级 `d_t(v)` **不是同一个监督量**。亦可对已生成完整动作聚合 token `d_t(y_t)` 做额外一致性诊断，但不得混称。
- 岭回归正则强度和所有标准化仅由训练 folds 确定；若能直接复用仓库的 `visibility_projection.py`，必须先核验其真实输入/标签/折分语义，**不能假定它天然适配购物动作**。
- 报 OOF 可解释比例/MSE 相对基线和行动类别覆盖。**若动作参照题外能预测、rank=8 token 不能预测，支持“结构或目标粒度可能是瓶颈”，但因输入和监督粒度不同，并不能单独证明学生必然可在 token 级学会该信号**。

### 4.4 P1 状态与按类别校准收缩（取代硬门控）

对每个页面/类别先判断埋信号功效，再判断真实 OOF 可见性：

- `PASS`：功效达到 §4.2，真实 OOF 相比最强简单基线有正改进且任务簇 CI 下界 >0，折间预测符号稳定。含义是“检测到稳定非零可见信号”，**不设高 R² 门槛，也不是允许训练 token 的布尔 mask**。
- `FAIL_POWERED`：功效合格但真实 OOF 无可靠改善，表示当前特征/规模/模型下未检测到信号。
- `UNDERPOWERED`：埋信号回收不足，不能将真实 OOF 差写成学生不可学。
- `BLOCKED`：数据串漏、数值错位、候选质量严重问题使该类结果无法评价，必须解决 P0/数据问题。

**P 在所有与 R 相同的结构性有效 assistant response 位置上计算 KL，绝不因为某类 P1 未 `PASS` 就把它硬屏蔽。** 为避免在低信号类把拟合噪声当成修正，预先从 `T_cal` 的**真实标签 OOF 预测**估计类别校准斜率：

```text
b_raw,c = Cov_task_weighted(hat_d_OOF,c, d_c) / Var_task_weighted(hat_d_OOF,c)
b_c     = clip(b_raw,c, 0, 1)
d_P,t(v) = b_class(t) * hat_d_t(v)
```

计算 Cov/Var 时用与 OOF 诊断一致的任务与 token 权重，必须将同位置的全部候选 token 一起统计，不能把同题 token 当独立任务。类别内预测方差近零或非有限时，`b_c=0` 并标记 `UNSTABLE_SLOPE`，不得用极大斜率补偿。每类报告**截断前斜率、截断后 `b_c`、任务/商品簇 bootstrap 95% CI、有效任务数、零修正比例**；为判断是否包含 0，同时保存截断前 bootstrap 区间，避免截断边界制造假显著性。低信号类 `b_c` 自动接近/归零，但它们**仍在训练 mask 内**；`q_P≈q_S` 不等于跳过 token。

**`b_c` 只在 `T_cal` OOF 上估计，在在线 OPD 启动前冻结**，不得依据 `T_opd` 当前任务、V、reward 或 actor 训练曲线重新校准。100 更新投影器可按 §4.5 刷新参数，`b_c` 仍沿用冻结值；报告快照漂移下的校准失配，这属于 v0.4 限制。

在线实验停损：如果所有**决策内容类别（search/click/规格）**的 `b_c` 任务簇区间都包含 0，或因 `UNDERPOWERED/BLOCKED` 无法得出可靠判断，P 只做 1-update 工程 smoke，**不投入 P 的 10/100 更新**；保留离线结果并申请增加独立任务/修复阻塞。不能把这个结果包装成投影没有价值。反之若有至少一个决策内容类别 P1 `PASS`、且其正校准斜率可信，则 P/R/R-β/P-shuffled 从 10-update 同时开始；其他类别照样训练，靠 `b_c` 收缩而非 mask 过滤。

### 4.5 快照与 100 更新的在线交叉拟合

10-update pilot 可用 `T_cal` 初始快照 `phi_0` 与已冻结 `b_c` 做 **frozen-projector pilot**，但每次 teacher/student 目标及候选来自当次 `bar_theta_k`；逐步记录 `actor_sha/scorer_sha/projector_fit_sha/projector_age_updates`，不声称严格同快照投影。

**若获批进入 100 更新，投影器（P 与 P-shuffled）必须采用按时间因果隔离的在线交叉拟合，不允许仍只依赖初始 `T_cal`**；校准收缩 `b_c` 与 R-β 的系数只在初始 `T_cal` 上估计并冻结：

1. 建立仅包含**此前**已结束批次轨迹的校准 buffer，允许在批次结束、actor 更新完成后，独立评分其 privileged 标签用于**未来**投影器拟合；**当前批次 P 蒸馏目标不可使用当前任务的 `z` 或标签**。
2. 当前任务按照冻结的 task/ASIN fold 分配，只用其他 fold 的先前任务拟合其预测器；同任务（最好同 ASIN）即使在更早批次出现过，也不可进入当前 fold 的拟合数据。每次刷新保存 cutoff update、训练任务/商品 hash、fold manifest，使用先前 `T_cal` 作为冷启动且同样执行隔离。
3. 每次刷新时，优先对历史缓存轨迹用**该次冻结 actor 快照**重新 teacher-force score 并提取 `r/e` 后重拟合 rank=8；否则必须标明历史快照混合导致的 stale-snapshot approximation。默认至多每 10 更新刷新一次，并按成本决定能否更频繁；刷新间隔仍有版本滞后，不得说“每一步都严格同模型”。P-shuffled 同步用先前任务的乱标标签刷新，强度在**只含历史训练数据**的预注册校准规则下匹配当前 P；不可用当前任务的 gold、标签或 V 来拟合/校准。
4. 100 更新开始前实测该刷新成本，并获明确 GPU 预算授权。若无法承担在线跨折刷新，**暂缓完整 P-100 对照**，不能以冻结 0-step projector 冒充符合本版协议的正式结果。

## 5. 核心训练：相同 token 位置的候选集条件 OPD（无 GRPO，无硬门控）

**旧方案的 `A^RL/w/eta`、GRPO/PPO actor hook 全部禁止。** 每个 on-policy 迭代，以**更新前 actor 当前权重**为同模学生/老师冻结快照 `bar_theta_k`，生成学生轨迹、同前缀评分或 student-only 投影，然后对同一批新数据做**1 个 actor optimization epoch**。teacher、snapshot、projection 和所有 target 都 detach。

对每个结构性有效 assistant response 位置，定义共享候选集 `C_t`，快照 student logits `s_t(v)`，原始中心化老师修正 `d_t(v)`，rank=8 预测修正 `hat_d_t(v)`，`c(t)` 为冻结的 token 类别：

```text
q_S_t(v)   = softmax_{C_t}(s_t(v))
q_R_t(v)   = softmax_{C_t}(s_t(v) + d_t(v))
q_P_t(v)   = softmax_{C_t}(s_t(v) + b_c(t) * hat_d_t(v))
q_Rbeta_t(v) = softmax_{C_t}(s_t(v) + beta_c(t) * d_t(v))
q_Shuf_t(v)  = softmax_{C_t}(s_t(v) + gamma_c(t) * hat_d_shuf_t(v))
q_actor_theta,t(v) = softmax_{C_t}(a_theta,t(v))
L_group = mean_same_valid_response_positions KL(stopgrad(q_group,t) || q_actor_theta,t)
```

- `beta_c`：R-β 按类别冻结的强度匹配系数；§6 说明为何采用一组类别系数而不是假定一个全局 β 能同时匹配五类分布。若只允许单一全局 β，也要在预定准则下最小化所有类别分位数误差，并如实报告失配；**主协议使用按类别冻结的 β**。
- `gamma_c`：P-shuffled 的训练侧强度匹配系数，仅来自 `T_cal`/合规的历史 train-only 校准数据；不能让其借真实 `d` 学习映射，见 §6。
- 对类别内近零的 `b_c`，`q_P≈q_S`，该位置仍计入同一个训练分母，而不是 P-specific token 跳过。

**共享有效位置规则**：R/P/R-β/P-shuffled 都使用同一冻结的**学生可见结构条件**：assistant response token、同一 `C_t` 生成规则、合法位置、长度/对齐/精度检查、非 padding、非工具输出。不得用 P1 `PASS`、预测值大小、是否 gold、teacher 是否偏好某 token 来决定 P 的 active mask。若原始老师评分失败或数据无法对齐，逐项审计；为了维持比较的同一目标域，成对离线/首步数据上须严格用共同有效位置；在线轨迹分叉后只能保证规则相同，不能谎称不同策略的 token ID 完全相同。遇到评分缺失不能无声置零或只对某组有选择地丢弃难样本；若无法在同等规则下处理，停止该批并记 `SCORER_FAILURE`，报告分组失败覆盖。

关键实现：

1. **只更新 actor LoRA**；快照 teacher、`q_R/q_P/q_Rbeta/q_Shuf`、projector 参数、候选 ID 与校准系数全部不反传。投影器只用 §4 的 OOF/历史数据单独拟合。
2. 每个 on-policy 迭代使用当次 actor 作为 snapshot，**只做 1 个优化 epoch**，固定 minibatch/optimizer 日程，避免多轮反复追逐陈旧目标。1 epoch 内不同 minibatch 的先后更新仍可能使后续零修正位置产生自蒸馏梯度，必须量化。
3. **所有主组在同一结构性有效 response token 上计算 KL，分母规则完全相同**：推荐每条轨迹平均再每任务平均，防止长轨迹/多响应主导。没有有效 token 时统一处理并报告，不得把 `hat_d=0` 当作缺失值。
4. 原 rollout 采样温度可用 `T_sampling=0.7`；学生/老师同前缀评分与 q 构造统一 `T_score=1`，禁止将 vLLM temperature processed logprob 与 HF 原生 logprob 相减。
5. **不再执行按类别可靠门的 hard mask。** `PASS/FAIL_POWERED/UNDERPOWERED` 只用于诊断和是否值得**整体启动在线 P 对照**，token 层是否更新由实际 KL 及 `b_c` 自然控制。
6. `q_P` 及 P-shuffled **不能读取当前 `T_opd` 任务的特权包/原始标签**。Raw/R-β 评分端可访问当前训练任务 `z`；actor 前向从不接触 `z`。独立只读评测审计的 V `z` 按 §2 执行。
7. 本 loss 为 **candidate-conditional forward KL**：只约束 `C_t` 内相对概率，不约束候选集合外总质量，不等于全词表 KL。`q_R=softmax_C(s+d)` 与特权老师在 `C_t` 上重新归一化完全等价。
8. 统计每类 `KL(q_group||q_S)`、`|shift|`、KL 梯度 proxy、**零修正位置占比**（基于冻结数值容差，`b_c=0` 类单列）、零修正在 epoch 内的自蒸馏 KL。目标强度相近不保证 actor 梯度完全相同，必须报告其差异。

**为何保持这个目标**：主投影器直接拟合老师对候选 token 的**相对 logit 修正**，不重建老师隐藏状态；R 与 P 在同样的 target 空间比较。注意 `softmax(s+E[d|h])` 一般**不等于** `E[softmax(s+d)|h]`，因此 P 并非严格在分布空间重现前向 KL 的条件平均老师。不可预测残差也不自动等于有害信用。P 能否在泛化上占优，须以 §6–§7 的强度/扰动控制与未见商品行为结果判定。

## 6. 组别与对照：用 R-β 替换 R-masked

| 组 | 训练目标（同一有效 response 规则） | 回答的问题 |
|---|---|---|
| **B：No-Update** | 不训练，同一起点 actor | 不训练的原始表现 |
| **R：Raw-OPD** | `KL(q_R || q_actor)`，完整原始修正 `d` | 特权蒸馏是否有效 |
| **P：Projected-OPD** | `KL(q_P || q_actor)`，按类别 OOF 校准 `b_c·hat_d` | 可见条件修正是否促进泛化 |
| **R-β：Strength-Matched Raw** | `KL(q_Rbeta || q_actor)`，按类别冻结 `beta_c·d` | P 是否只是因为教师变温和 |
| **P-shuffled：Null Projector** | 用按任务打乱标签拟合同一 rank=8 结构，按类别 `gamma_c` 匹配 P 强度 | P 是否只是随机扰动/正则化 |
| **N：Empty-z/Self** | 仅单测/1-update，`z=空`，`q_R=q_S` | 模板、detach、零修正数值负对照 |

**R-masked 不再存在。** R/P/R-β/P-shuffled 从第一次 10-update pilot 起共同运行（以 P1/P0 和整体预算门通过为条件）；不以 `P>R` 或 V@10 作为加入对照的条件。B 只评测，不更新。

### 6.1 R-β：在 `T_cal` 上预先匹配 P 的蒸馏强度

在 `T_cal` 的**OOF 对应位置**上，先由正常 rank=8 投影器及已冻结 `b_c` 构造 `q_P`，记录各 token 类别的 `KL(q_P||q_S)` 分布。对每个类别仅从预先登记的有界参数域 `beta_c∈[0,1]` 选择固定值，最小化 `KL(softmax_C(s+beta_c*d)||q_S)` 相对于 P 的**中位数及预指定分位数（建议 P25/P75/P90）**的距离（类别权重、拟合目标、离散候选网格/优化器及匹配容差均在 P0 前冻结）。**不能用 V 或 OPD 的 reward 调 β。**

称整组冻结系数 `beta={beta_c}` 为 **R-β**；之所以允许按类别 β，是因为单一全局 β 通常无法同时匹配 search/click/规格/格式/思考的 KL 分布。不能保证一维缩放使全部分位数完全相同，必须保存按类别的分位数差异并标 `STRENGTH_MISMATCH`；不允许无界调强度来造匹配。同步比较实际 token-level `|q−q_S|`、KL 梯度 proxy、任务覆盖，避免只以 KL 中位数宣称完全等强度。所有 β 在在线训练前冻结；100 更新因 actor 漂移产生的失配继续报告而不调参。

### 6.2 P-shuffled：任务级乱标与强度匹配

在 `T_cal` 上按任务打乱原始 `d`（页面/类别匹配、候选内重中心化），拟合同架构 rank=8 的 `hat_d_shuf`。为保持“只有输入→真实老师修正关系被破坏”，**不对乱标投影器按真实老师标签重新做正向校准**（否则校准接近 0，失去强度匹配意义）；以按类别冻结的 `gamma_c` 将其 `KL(q_Shuf||q_S)` 分布匹配 P，使用同一 `T_cal` OOF、预注册目标函数与有界缩放（初始上限 4 倍）。无法达到 P 强度则报 `STRENGTH_MISMATCH`，不能通过 V 调整。

投影器的随机目标和输出必须只依赖 train-only 乱标方案，不允许 `gold ASIN`、当前任务原始 `d` 或 V 进入 projector 输入/拟合；匹配阶段只读 `T_cal` 的目标强度**聚合量**，不能拿正常标签监督乱标预测器。100 更新时沿相同历史刷新 schedule 和 as-of 隔离，用合规历史训练数据做 P-shuffled 强度匹配，不允许读取当前任务特权标签以调整修正。

### 6.3 预注册主比较

- **主要**：`P−R-β`（排除仅靠温和老师）与 `P−P-shuffled`（排除随机扰动/正则化）。
- **同时报告**：`P−R`、`R−B`、`R-β−R`、`P−B` 和 seen/unseen、train−V gap、购买率及 reward_type 分布；`P−R` 不再是唯一主因果比较。
- R/P/R-β/P-shuffled 使用统一初始 Qwen2.5-7B-Instruct、相同 `T_opd`、温度、接口、LoRA 超参、迭代数、任务顺序与 seed 计划；策略更新后 rollout 分叉是必然的，只承诺**选择位置的规则相同**，不承诺逐 token 相同。单 seed 是探索；有明确收益再从原始起点按固定方案进行至少 3 seed 确认。
- 未见商品是泛化主层，但不能仅凭 P 胜出称为“去特权”成功；还需 §7 的等价商品偏好诊断与强度/随机控制。不可预测残差可能提供有用监督，其效果由实验裁定。

## 7. 预注册门槛、离线过滤诊断与行为指标

### P-pre：特权包可用性门（早于 P0/P1 的大规模工作）

- 在 `T_pre`、冻结初始快照上完成 §3.1 的五种 teacher-as-policy 对照与同一学生轨迹的 placeholder-controlled token 评分；只看训练来源 T，不看 V。
- 通过条件：至少一种包令全要求购买率对 `z_none` 正改善，且在 click/规格等动作内容上产生相对**对应占位模板**超出数值阈值的非零变化，覆盖足够。冻结 `z_selected`；`PRIVILEGE_NOT_USEFUL/PREFLIGHT_INCONCLUSIVE/PACKET_UNAVAILABLE` 都停止下一阶段，不能以低功效宣称不存在信息作用。
- 预检奖励仅用于选择**固定包**，不得进入后续 actor loss。模板中的空段、真实包、内容占位必须严格区分；若所有包失败，阶段 B 的 `T_cal` OOF 不得自动启动。

### P0：工程/数值门（必要）

- `z=空` 时评分输入 token IDs 与 student **逐 token 完全一致**，不插入空的 extra system message；student/empty-z logits、`d`、`KL(q_R||q_S)` 在冻结 dtype 容差内近 0；`softmax_C(s+d)==softmax_C(p_priv)`。
- `C_t` 只来自 student top-8 + 实际采样 token；R/P/R-β/P-shuffled 同集合，teacher 不能提名；比较采样与 teacher-forced 精度/温度路径，特别核对既有 BF16/vLLM `click[...]` 概率翻转风险。
- actor 输入和投影器特征没有 `z`、gold 等字段；response mask 不含 prompt/工具输出/padding；teacher/target detach；多 rank 权重同步；NaN/Inf、错位和不合理零梯度立刻停并报告。
- P-pre 前就冻结全部候选包/对应 placeholder 的插入位置、角色、固定文字模板及 hash；P-pre 后锁定 `z_selected`；在 P0 前复验选中包的字段可见度（yes/partial/no）、token 数及版本；测试 `click[ASIN]` 子词、search 参数、规格值、思考文本和 `z=空` 段落完全消失。

### P1：离线信号、统计功效与冻结校准

1. `packet_selection.json` 已记录 P-pre 通过、`z_selected` 已冻结，且 `T_pre` 不用于此处拟合；`T_cal>=512` 不同任务 × 1 条 student rollout，优先 ASIN 分折；报告每页面×类别的**任务/商品/token 数**，不能用海量同题 token 冒充独立样本。
2. 埋入 0.50 可预测份额的功效检查：`recovery>=0.50` 且任务簇 bootstrap CI 下界 >0 为 `POWER_OK`，否则 `UNDERPOWERED`，不能做真实信号阴性推断。
3. 真实标签 OOF：只需相对恒零、训练折均值及 token-only 中**最强简单基线**有正 MSE 改善，且任务簇 CI 下界 >0、预测符号在相同留出 token 上折间稳定。**不设解释方差/R² 最低比例、也不要求单次 `d` 与 `hat_d` 同号率达标。**
4. 用 OOF 的 `hat_d/d` 按类别估计 `b_raw,c`、截断的 `b_c` 与任务/商品簇区间；记录 `UNSTABLE_SLOPE`、零修正占比。`b_c` 在在线开始前冻结，**P1 PASS 只作结论，不变成 token mask**。
5. 同批做 full-action 冻结表示+满秩岭回归容量参照，并报告 `mass_S(C)/mass_T(C)`、Raw/P KL/entropy、候选有效覆盖；不能将动作级标签和 token 级监督混为一谈。
6. 若所有 search/click/规格决策类别的 `b_c` CI 均包含 0，或缺少可评价的 `POWER_OK/PASS` 决策类别，P 仅做 1-update smoke，停止其 10/100 主实验，并报 `UNDERPOWERED/FAIL_POWERED/BLOCKED` 的具体原因；不自动增大 rank、改损失或在 V 上调参。
7. **冻结 R-β 与 P-shuffled 的强度匹配系数与误差**：均仅以 `T_cal` OOF 为依据，保存按类别 `KL(q||q_S)` 的中位数、P25/P75/P90、零信号比例和幅度匹配；拟合达不到时标 `STRENGTH_MISMATCH`，不可隐瞒。

### P1-D：等价商品 gold 偏好诊断（评测/审计侧，绝不训练）

**目的**：测量老师是否偏向“商品编号恰好等于 gold”，而不是更一般的“满足全部要求”。在 click 决策时，仅用**学生当下可见的商品页面与合法候选**列出候选，随后在独立只读审计器用 gold 和冻结规格/价格/约束判断哪些商品**全部满足要求**。仅纳入同一状态下至少有 **2 件满足全部要求的不同商品**，其中一件是 gold 的状态；若 gold 尚未作为合法可见候选，不能为了凑样本把它插到学生候选集。报告每类覆盖、任务数和未纳入原因。

对 gold `a_g` 与等价替代品 `a_alt` 的共同可见动作做**只读比较**：使用相同的冻结学生前缀与老师包，计算 Raw 特权老师对完整 `click[...]` 的 teacher-forced 动作对数似然差；P 端在相同各动作前缀上，用 `s_t(v)+b_{c(t)}·f_phi(r_t,v)`（每个动作 token 按冻结解析器确定类别）形成**诊断专用全词表扩展**，计算相同动作的对数似然差。记录长度归一化差、未归一化差及 `raw_margin−projected_margin`，并做任务/ASIN 簇区间。**同时必须报告相对学生基线的额外 gold 偏好**：`excess_R = margin_R−margin_S`、`excess_P = margin_P−margin_S`，其中 `margin_S` 由同一无特权 student 在同一动作对上计算。因为两件商品即使都符合要求，学生可见信息仍可能使其中一件更自然/更受偏好，故“P 在 unseen 上接近 0”的合理预期针对的是 `excess_P`，**不是无条件要求 `margin_P=0`**。**这一全词表扩展只用于比较完整动作，不参与训练；其 token 可能在主训练 top-8 外，须另报 out-of-support 比例，不能把其值冒充原候选集 `q_P` 的直接测量。** 另在 gold/alt 首个不同 token 都处于同一原始 `C_t` 的可评价位置，报告**严格基于训练候选集 `q_R/q_P` 的条件 log-odds**作为高可信子集；没有覆盖时标 `INSUFFICIENT_PAIRS`，不能填零。

先在 **`T_cal` 的 OOF** 上做上述计算；若按 ASIN 严格分折，所有被留出 gold 对其拟合折都属于 **unseen**，不能在这里凭空制造 seen 子组；seen/unseen 比较主要放在冻结 V（或预留内部测试池）上。若 §2 的 V 只读隔离条件成立，**只能在全部协议、系数、训练及选 run 规则冻结后**对 V 进行一次诊断评分，按训练期间 gold 商品 seen/unseen 分层；V 的原始 `z`/gold/老师输出严禁进入训练或下一轮调参。若这一步违反现有 V 禁止 teacher scoring 的工程隔离，就**不**在 V 进行，改用预先冻结的 train-only ASIN-disjoint 审计池，不把它称为 V。期望模式是 Raw 对 gold 的**基线以上额外偏好**高于 P，尤其新商品上 `excess_P` 更接近 0；**这只是预注册预测，不是必须出现的结论**。

### P2：1-update 与 10-update OPD-only pilot

- 1-update：N/R/P/R-β/P-shuffled 检查相同结构性 token mask、target detach、LoRA 实际更新、`z=空` 自蒸馏梯度、分布式同步和 GPU 成本；在**同一离线轨迹/第 0 步**严格检查四个训练组的有效位置相同，更新后在线轨迹分叉仅要求规则一致。
- 10-update：通过 P1 后共同运行 R/P/R-β/P-shuffled（B 不更新）；每个 on-policy 迭代以当次 actor 为快照，只做 1 optimization epoch。在**同一批固定 detached target** 上测量每步 actor 更新前/后 KL 变化，报告合法动作、任务覆盖、非零校准修正比例、分类别 target KL/熵与梯度 proxy、`b_c/β_c/γ_c`、projection age、自蒸馏 KL、成本。不同批次的目标 KL 不可直接比较来宣称单步改善。
- 预注册 100-update 继续门：P0 全通过；P1 至少一个决策内容类别有稳定非零可见预测且 `b_c` 区间不含 0；pilot 大多数更新的固定目标 KL 下降、无数值/隔离异常；10 更新按 §16.3 为 search/click/规格各核验至少 N_c 个有效 token 和至少 50 个不同任务（这是任务覆盖的工程 sanity 门，**不是 P active mask**），并单独报告 `KL(q_P||q_S)` 非零的决策 token 比例及 0 修正比例。预算明确授权。阈值在 pilot 前冻结；若低于要求标 `LOW_DECISION_COVERAGE` 或 `LOW_SIGNAL`，不能用 V@10 reward 调它。
- 在 `V@0/10` 与 `T_opd_eval@0/10` 进行无特权行为评测；报 Reward v3、严格成功率、**满足全部要求的购买率**、reward_type 全分布、合法率、search/click/规格漏斗、轨迹长度，以及未见商品的**click/规格决策 token 学生熵**、满足全部要求购买率/严格成功率之比。V@10 只描述，**不能作为是否进行 100 更新的门槛**。

### P3：有条件的完整 100 更新

- P0/P1/P2 与成本门通过、获得额外授权后，从同一 0-step checkpoint **重新**启动正式 B/R/P/R-β/P-shuffled 比较，按 0/10/20/.../100 评测。终点 V 每题 4 sample + 1 greedy，同协议评估 `T_opd_eval`；训练不使用 reward。
- 正式 P/P-shuffled 依 §4.5 **在线历史任务/商品隔离交叉拟合**刷新 rank=8；`b_c/β_c` 仍来自冻结 `T_cal`；P-shuffled 的 `gamma_c` 匹配**算法**在 P0 前冻结；pilot 的 `gamma_c` 数值取 `T_cal` OOF，正式在线刷新时可按该算法只用合规历史 train buffer 重新得到版本化 `gamma_c`，不得读当前批次标签或 V。无法承担在线刷新/主对照的成本时暂缓正式 100 更新，不可把 frozen projector 运行说成已遵守协议。
- **主要泛化终点**：gold ASIN 未见过的 V 层（不足则预留 ASIN-disjoint 内部测试池、如实标注来源），同报 Reward v3、严格成功率、**满足全部要求的购买率**（`reward_type∈{gold_purchase, valid_alternative_purchase}`）及**完整 reward_type 频数/比例**。Reward v3 对 gold 购买给 `1.0`、满足要求的替代购买给 `0.55`（须以服务器真实 Reward v3 代码核验并记录）；因此单看 gold 严格成功率会混入商品 ID 的差别。严格成功率具体实现若与 gold-only 不一致须核验并写清，不得偷换。
- **主比较** `P−R-β`、`P−P-shuffled`，同时报告 `P−R`、`R−B`、`P−B`。任务配对、ASIN 簇 bootstrap、3 seed 复验（资源许可且有正向探索证据），加报告 V seen、`T_opd_eval−V` gap、决策 token 学生熵、`all_requirements_purchase_rate / strict_success_rate`（分母为 0 时 NA，附两项原始率）、购买类型与任务漏斗、候选 mass、训练目标 KL、信号零修正/cover 与 GPU-hour。
- V-unseen 太少则不能宣称“未见商品泛化改善”，即便 overall V 有提升也不能替代。**熵塌缩与 R 在 seen 上更好、P 在 unseen 上更好，都仅为待验证预测**，不是预设结果。不得看 V 调 rank、β、b、γ、门槛或选择训练 checkpoint。

解释边界：

- **不可预测残差不等于有害信息。** 对前向 KL 而言，目标分布的样本波动在期望风险下会被平均；Raw 的潜在问题更可能是有限样本方差、记忆 gold 编号、容量/优化效应。且 logit 条件平均与分布条件平均并不相同（§5），P 不保证胜 R。
- MSE 只是“非零可预测修正”的检测统计量，不要求 `hat_d` 与单次老师标签高度吻合；OOF 方差很大时依赖埋信号功效审计。`P1 PASS` 不等于去特权证明。
- `T_opd_eval−V` gap 与 seen/unseen 是泄漏风险读数，不是单独因果识别。Gold/等价替代偏好审计直接一些，但全词表扩展本身超出候选训练支持，需要与候选内子集一同解释。
- 即使 R-β 匹配了 KL 分布，也未必匹配方向/梯度；P-shuffled 也可能在强度匹配后出现其他分布差异，须报告残余混杂。单 seed、低候选 mass、弱统计功效均限制结论。

## 8. 工程实现清单与最低测试

建议目录 `exploratory_shop_audit/visible_opd_v0/`，不覆盖旧实验：

1. `dataset_split.py` / `asin_audit.py`：冻结 `T_pre/T_cal/T_opd/T_opd_eval/V`、task hash、gold ASIN overlap、ASIN 分折、seen/unseen；如需评测隔离池，在训练前冻结并做权限分离。
2. `teacher_packet.py`：仅 train 用 `z_full/z_product/z_evidence` 与各模板占位包生成、P-pre 后的 `z_selected` 锁定、动态 evidence 当前状态来源、字段可见度 yes/partial/no、字段 token 数/包长度/超限、插入位置/固定模板/hash、empty-z 段落完全不存在；如启用 V 的特权只读审计，必须为不同权限/目录的 **audit-only** 工作流，不能用训练 packet 生成器。
3. `token_parser.py`：冻结思考/search/click/规格/格式 + unknown 分类及边界测试。
4. `token_scorer.py`：同快照 teacher/student 同前缀评分、`C_t/d_t(C)/q_S/q_R`、候选 mass、精度/版本/成本，强制记录评分失败与对齐错误。
5. `projection.py`：**唯一主投影器 rank=8**、按 ASIN/任务 OOF、类别 `b_c` 估计与 bootstrap、`q_P`、online 版本缓存；**删除 hard PASS mask**，不写 MLP/Critic。
6. `power_check.py`：任务级打乱+已知 rank≤8 埋信号/0.50 方差份额注入、功效回收与任务/商品簇 bootstrap。
7. `capacity_probe.py`：冻结学生对上下文+完整候选动作 hidden 的满秩岭回归，只有 OOF 审计输出，不生成 actor 蒸馏目标。
8. `opd_loss.py`：候选集 KL、target detach、**四训练组共享有效 response mask 与分母规则**、1 epoch 更新、零修正位置自蒸馏 KL 和 KL 梯度 proxy；无 reward/advantage。
9. `controls.py`：**R-β 按类别 beta** 在 `T_cal` OOF 上匹配 P 分位数；P-shuffled 按任务乱标拟合、按类别 `gamma` 匹配 P 强度；强度/目标和失配诊断，旧 `R-masked` 代码禁止在主训练入口启用。
10. `equivalent_product_audit.py`：只读、isolated gold/full-requirement 审计，合法可见候选中等价商品对筛选、gold vs alt full-action margin（明确 P 全词表扩展的诊断性质）、原 `C_t` 内分叉 token log-odds、seen/unseen 和覆盖率；输出绝不反馈训练。
11. `refresh_projector.py`：历史 buffer、as-of cutoff、fold/ASIN 排除、当次快照重评分与 stale 标记；P-shuffled 同日程刷新，`b_c/β_c` 仍冻结。
12. `train_opd.py`：独立纯 KL 训练：on-policy rollout → current snapshot teacher/投影 targets → LoRA 1 epoch；无 GRPO/PPO/环境 reward。
13. `evaluate.py` / `report.py`：B/R/P/R-β/P-shuffled 的 train/V seen/unseen、Reward v3、严格成功、全部要求购买率、reward_type 全分布、决策熵、比值、ASIN bootstrap、KL/cover 与 GPU ledger。
14. `privilege_preflight.py`：在 `T_pre` 同任务五种 teacher-as-policy `sample×2/greedy×1`、学生轨迹五种 teacher-forced 评分、每模板 placeholder 内容对照与冻结阈值择包；输出 `preflight_utility_report.json/privilege_delta_report.json/packet_selection.json/template_effect_report.json`，预检本身绝不更新模型。
15. `test_*.py`：CPU 合成+真实前向 smoke；默认命令不自动启动 100-update GPU 训练。

最低断言：

- P-pre 选包的 `T_pre` task/ASIN hash 不得进入 T_cal 投影或 T_opd 梯度；真实包相对同格式 placeholder 发生变化时才算内容效果；`z_none` 严格不插段落，所有包保持原 student 历史逐 token 不变。`z_evidence` 只包含实际当前页信息、不标 gold；包缺失/过长、类覆盖不足与预检预算问题不得静默忽略。
- `softmax_C(s+d)==softmax_C(p_priv)`；`d` 加常数不影响分布；`z=空` 无额外段落、输入 token IDs 完全不变、`q_R≈q_S`。
- teacher/student 真前缀、tokenizer、精度/温度/候选对齐；尤其 BF16/vLLM `click[...]` 风险。target 与 `U/V/b_c/β_c/γ_c` 均 detach，不读当前任务 reward。
- `T_cal/T_opd` task hash 分离，OOF 同商品/同题不得跨 fold；online 当前批次/同题/同商品不得进入对应投影器的拟合/校准。
- OOF 功效 toy：`d=x+10z` 且 `z` 独立于学生特征时不恢复 `10z`；`d=x` 能检测非零可见成分；动作 probe 与 token 标签不可混用。
- **相同离线轨迹/第 0 步上 R/P/R-β/P-shuffled 结构性有效 mask 完全一致**，即便 `b_c=0` 也不跳 P token。强度校准只读 `T_cal` 聚合，乱标模型不能用真实标签直接监督。
- `beta_c=0`/`gamma_c=0` 产生 snapshot target，actor=snapshot 且 step 开始时 KL 梯度近零；1 epoch 过程中 self-distill KL 可非零须记录。
- 等价商品对只能从**可见合法动作**提名；gold 仅用于只读审核是否等价，不允许 gold 拓宽训练候选或改 actor prompt；报告缺少 gold 候选、支持集外 token、V 隔离证据。
- 每个 checkpoint 恢复包含 actor/optimizer/scheduler/RNG、任务顺序、teacher/student snapshot hash、投影器/校准参数与 as-of fold/版本；缺项不得声称精确恢复。

## 9. 执行顺序与计算预算

**阶段 A：只读仓库 + CPU 预检（不改原训练）**

- 核验真实分支/环境/checkpoint/rollout/reward/token mask 代码，输出 `architecture_map.md/environment_manifest.json/preflight_report.json`；CPU 先算 V-unseen gold ASIN 数，若不足（例如 <100 任务）在训练前按 §2 冻结 ASIN-disjoint 内部测试池，不能事后补。
- 冻结 `T_pre≈100`、`T_cal>=512×1`、`T_opd`、`T_opd_eval`、V、ASIN 分层/fold、三个真实包+对应占位包每字段可见度与 token 数、模板位置/文字/hash、P-pre 阈值/择包规则、top-8、rank=8、token parser、OOF/功效/容量 probe、`b_c/β_c/γ_c` 的**计算与强度匹配算法**（数值要经阶段 B 由 T_cal 得到）、等价商品只读审计规则。写入 `PROTOCOL.md`。

**阶段 A.5：GPU 特权包预检（在阶段 B 之前独立签收）**

- 先 8–16 题核验 teacher-as-policy 的五包提示词/模板/合法动作、student/empty-z 逐 token 一致与评分精度、`z_evidence` 动态来源和 GPU 成本；预算获批后才按冻结 `T_pre≈100` 执行各包 `sample×2+greedy×1`，以及**同一份学生轨迹**的五组 teacher-forced 评分（product/evidence 另有不更新 actor 的配对占位评分）。
- 输出 `preflight_utility_report.json`、`privilege_delta_report.json`、`template_effect_report.json`、`packet_selection.json` 与 `preflight_cost_ledger.jsonl`；先暂停签收。`PRIVILEGE_NOT_USEFUL/PREFLIGHT_INCONCLUSIVE` 均不进入阶段 B，不自动追加包或增大任务数；后续探索须另获授权。

**阶段 B：GPU 对齐 + 已选 `z_selected` 的 `T_cal` 离线诊断（执行后暂停签收）**

- 已验收 `packet_selection.json` 的选中包后，8–16 个不同任务 P0 scorer 烟测；完整 `T_cal` rank=8 OOF、按类别/页面任务和 ASIN bootstrap、0.50 埋信号功效、动作级岭回归 probe、候选 mass。估计并冻结 `b_c/β_c/γ_c` 与其区间/强度误差；做 `T_cal` OOF 等价商品偏好诊断，不能访问 V 训练标签。
- **阶段 B 完成后先暂停，不自动进入多组在线训练**，提交 `projection_report.json`、`power_injection_report.json`、`capacity_action_ridge_report.json`、`candidate_mass_report.json`，并附 `calibration_strength_report.json`、`equivalent_product_bias_report.json`、成本及 `P1_decision.json` 供人工核验。未获得继续授权，不自动开始阶段 C/D/E。

**阶段 C：OPD 代码单测 + 1-update smoke**

- N/R/P/R-β/P-shuffled 同位置 loss/版本/强度/detach 的 CPU/真实 GPU smoke；验证训练 reward 不入梯度、LoRA 权重确实改变、多 rank 同步、零修正自蒸馏行为。
- 如果 P1 没有可靠决策内容信号，P 只运行工程 smoke，不将 R/B 结果声称为“P 失败”。

**阶段 D：10-update OPD-only pilot（P1 满足时四训练组同时跑）**

- R、P、R-β、P-shuffled 从相同 checkpoint 和预注册任务顺序出发，B 不更新；冻结 `b_c/β_c`，P/P-shuffled 的初始 projector 可采用 `frozen-projector` 并明确滞后。
- 训练侧汇报固定 target 的 KL 前后变化、decision token 比例、零修正/自蒸馏 KL、强度匹配误差、任务/ASIN 覆盖及成本；V@0/10 和 T_opd_eval@0/10 的行为数据用于描述，**不用于决定 100 更新去留**。

**阶段 E：获明确新增 GPU 预算授权才做 100-update 在线投影刷新**

- 重新从同一 0-step checkpoint 跑各组，P/P-shuffled 用§4.5 的严格时间隔离与跨题/ASIN online projection；`b_c/β_c` 不随 V/训练结果更改。不能负担刷新/对照时暂缓，不许偷换成固定 v0 投影器。
- 同步评测 V seen/unseen、`T_opd_eval`、全要求购买率及 reward_type，100-step 终点 4 samples + greedy；按任务/商品簇与 seed 给出配对区间，记 GPU-hour，未运行项必须注明 `not run`。

## 10. 交付与研究表述

运行目录至少保留：

```text
PROTOCOL.md
preflight_utility_report.json
privilege_delta_report.json
template_effect_report.json
packet_selection.json
preflight_cost_ledger.jsonl
architecture_map.md
env_manifest.json
splits_hash.json
asin_overlap_report.json
asin_split_manifest.json
teacher_packet_manifest.json
token_parser_manifest.json
score_alignment.json
candidate_mass_report.json
projection_oof.parquet
projection_report.json
power_injection_report.json
capacity_action_ridge_report.json
calibration_strength_report.json
calibration_bootstrap_report.json
equivalent_product_bias_report.json
P1_decision.json
projector_versions/
projector_refresh_ledger.jsonl
control_strength_report.json
opd_targets_audit/
opd_training_metrics.jsonl
rollouts/
checkpoints/
evaluation/reward_type_distribution.json
evaluation/all_requirements_purchase.json
evaluation/decision_entropy.json
evaluation/train_vs_val_gap.json
evaluation/seen_unseen_report.json
cost_ledger.jsonl
pilot_decision.json
metrics.json
REPORT.md
RUNBOOK.md
```

原始逐题、含 gold ASIN 的训练用特权包、V 只读审计输出与可能反推出答案的逐 token 记录只保留在服务器隔离目录；Git 只存脱敏摘要、代码、协议和 hash。未经验证标 `unverified`，未运行标 `not run`，功效不足标 `UNDERPOWERED`，等价商品不足标 `INSUFFICIENT_PAIRS`。

`REPORT.md` 固定回答：

1. **P-pre**：五种 teacher-as-policy 在 `T_pre` 上有没有相对无包改善全要求购买率？`z_full/product/evidence` 的动作 token 相对**匹配占位模板**有没有可检测的内容效应，还是只有模板效应？选择的 `z_selected`、覆盖、成本、阈值和理由是什么？`PRIVILEGE_NOT_USEFUL` 是否触发？
2. Raw 老师是否给 student top-8+sample token 非平凡相对偏好？学生/老师候选 mass 有多大，分页面/思考/search/click/规格/格式如何？
3. `T_cal≥512×1` 的 rank=8 OOF 是否检测出**非零可见信号**，0.50 埋信号回收率如何；`b_c`、任务簇区间及零修正比是多少？为什么不是追求高 MSE 解释率？
4. 动作级满秩岭回归只读参照是否跨题可预测，与 token 信号的不同粒度是否分开解释？
5. R/P/R-β/P-shuffled 是否在同样结构性 token 位置训练？`beta_c/gamma_c` 的候选目标 KL 强度与梯度 proxy 匹配是否充分？P 相对 R-β/P-shuffled 能否改善 V-unseen 全要求购买率、Reward v3 和严格成功？
6. `T_cal/T_opd/V` ASIN 重合率、seen/unseen、`T_opd_eval−V` gap 与 decision token 学生熵如何？gold vs 等价替代点击偏好是否变弱，有多少可观测商品对，P 的 audit full-vocab 扩展支持性如何？
7. 特权包逐字段学生是否已看到（yes/partial/no）、token 数/固定插入模板是否按 P0 冻结？online projector 是否只用历史其他任务/商品、是否重评分，快照/候选/精度/成本限制是什么？

### 解释边界

- “学生可见”相对于冻结的学生表示、rank=8 函数类、OOF 数据与任务/商品分组，是操作性定义；并非形式化证明去除了所有特权信息。
- Raw 前向 KL 在期望上会平均老师**分布**的条件波动；P 是 logit 修正条件估计再 softmax，二者不完全相同。不可预测残差不能直接标作有害噪声；P 的主机制假设是**降有限样本方差/减少 gold 记忆，提高无特权泛化**，而非 Raw 梯度必然错误。
- 埋信号功效只验证已知 rank≤8 可表达成分的检测能力，动作 probe 成功不代表 token 投影器必定能够学习。
- P 与 R 在相同有效位置训练，但不同策略轨迹更新后会分叉；R-β 只匹配目标强度统计，并非逐梯度相等；P-shuffled 控制乱标扰动，仍须报告强度失配。
- V 的全要求购买率与未见商品回报是行为目标，等价商品 audit 更直接检验 gold 偏好但其全词表扩展超出主训练候选集；如果 V 只读隔离做不到，禁止在 V 评估带特权老师。
- 本轮**不回答**信用是否改善 GRPO；不得中途改 rank/MLP/Critic、奖励、full KL 或根据 V 筛模型。

## 11. 执行前签收检查（强制，不是建议）

- [ ] **P-pre**：`T_pre≈100`/5 模式 teacher-as-policy 与 student 轨迹占位评分完成；冻结全部包/占位模板、token 阈值、成本与 `packet_selection.json`；未通过则不运行 512×1 离线拟合。
- [ ] `T_cal≥512` 不同任务 × 1 rollout；任务/商品簇 OOF、0.50 埋信号检验与 `UNDERPOWERED` 判定已预注册。
- [ ] 同样决策点的只读 full-action ridge 容量参照已定义，不参加 actor 训练。
- [ ] Gold ASIN overlap、V seen/unseen、ASIN-disjoint 内部测试池需求在训练前冻结；`T_opd_eval−V` gap 评测方案固定。
- [ ] `teacher_packet_manifest` 包含字段来源、是否学生可见、各字段 token 数、插入位置/模板/hash；空 `z` 整段消失。
- [ ] parser 冻结思考/search/click/规格/格式/unknown，能正确处理 ASIN 多 token 和 action boundaries。
- [ ] **无 hard PASS token mask；R/P/R-β/P-shuffled 同一有效 response 规则**，P 使用离线 OOF 截断斜率 `b_c`；R-masked **取消**。
- [ ] R-β 的 `beta_c` 只用 `T_cal` OOF 校准并冻结；P-shuffled 的 `gamma_c` 匹配算法与初始值由 `T_cal` 冻结（在线刷新仅用合规历史数据依预注册算法版本化更新），量化 KL 分布差异、梯度 proxy 及匹配失败。
- [ ] Gold/等价商品只读 click 偏好诊断已注册；如在 V 做，必须与训练/调参彻底隔离；否则仅 T_cal OOF/内部测试池。
- [ ] Reward v3/严格成功/全部要求购买率（含 reward_type 全分布）同时报告；V-unseen 决策 token 熵、全要求购买/严格成功比值为次要指标。
- [ ] P1 PASS 不要求高解释率、`sign(hat_d)==sign(d)`；折间稳定用相同留出位置多次拟合的同号比例，不能与单次标签同号混淆。
- [ ] 每步当前 actor 快照 + 1 optimization epoch，零修正位置的自蒸馏 KL 实测；100-step projector 只从历史其他任务/商品更新，`b_c/β_c` 按预注册冻结。
- [ ] 100-step 继续决策只看 P0/P1、固定 target KL、decision token 覆盖、数值稳定和预算，不看 V@10 reward。
- [ ] **阶段 A.5 先暂停签收**，提交 `preflight_utility_report.json/privilege_delta_report.json/template_effect_report.json/packet_selection.json`；通过后才进入 B。
- [ ] **阶段 B 结束暂停签收**，至少交 `projection_report.json/power_injection_report.json/capacity_action_ridge_report.json/candidate_mass_report.json` 与校准/成本审计，未经授权不自动继续在线训练。

## 12. 补充（2026-11-20；已合入 §2/§3/§7/§8）

**12.1 行为评测口径。** 主表除 Reward v3 和严格成功率，还需同时展示**满足全部要求的购买率**：`reward_type ∈ {gold_purchase, valid_alternative_purchase}` 的任务/轨迹比例，以及 **reward_type 的完整分布**。根据方案的 Reward v3 假定，gold = `1.0`、满足全部要求的替代品 = `0.55`，但实际执行须用服务器真实奖励代码核验。只看严格成功率可能把 gold 编号差异误读为没有学会购物要求。

**12.2 特权信息差清单。** `teacher_packet_manifest` 对每个字段记录“该字段内容是否已在学生可见任务文本中出现（是/部分/否）”和字段 token 数、来源及模板版本，报告老师额外信息的真实构成。

**12.3 模板冻结。** 特权包的角色、插入位置、固定文案、分隔符在 P0 前冻结、登记 SHA；空 `z` 时完全删除这一段落，不保留空角色或空 system 消息；执行过程中不得改。

## 13. 判定口径补充（2026-11-20；已合入 §4/§7）

**13.1 P1 信号解释。** `PASS` 检验稳定的、非零的学生可见修正，**不设解释比例门槛**。若残差条件均值为零，则恒零基线 OOF 改善量 `2E[d*hat_d]-E[hat_d²]` 仅由可预测的条件平均部分决定，残差主要扩大抽样区间。`sign(hat_d)==sign(d)` 只报告；折间稳定是在**同一批留出 token**上不同独立拟合的预测器预测同号比例。

**13.2 去 gold 偏好。** 在学生**可见合法 click 候选**中存在至少 2 件全部满足要求的商品、且 gold 可见时，审计 Raw vs Projected 对 gold 相对等价替代品的 logprob 差，以及**相对无特权 student baseline 的额外偏好**；优先 `T_cal` OOF / ASIN-disjoint，只读隔离允许则额外 V seen/unseen。完整动作需诊断专用 full-vocab 评分扩展；仅限原始 `C_t` 的部分另报高可信条件 log-odds 和可评价覆盖，不将扩展混称主训练目标。期望 Raw 的额外 gold 偏好高于 P，P 在 unseen 的**额外偏好**趋近 0，**不将预期写成事实**。

**13.3 泛化次要指标。** V-unseen 的 click/规格 student entropy 与 `all_requirements_purchase_rate / strict_success_rate`（分母 0 → NA）分别报告；R 可能更过度自信只是预先假设，不是训练门槛。

**13.4 假设边界。** 学生可见的条件平均修正可能比完整特权修正更利于泛化；原始残差不可直接定性为有害。必须联合 `P−R-β`、`P−P-shuffled`、`P−R` 及 V-unseen、等价商品 bias 诊断判断，不能只看训练 loss 或单个指标。

## 14. 去除硬门控（2026-11-20；已替换旧 §4.4/§5/§6/§7）

**14.1 同位置。** P、R、R-β、P-shuffled 对所有结构性有效 response token 计算 KL；不再以类别 PASS/FAIL 跳位置。`R-masked` 取消。

**14.2 类别斜率收缩。** `b_c=clip(Cov(hat_d_OOF,d)/Var(hat_d_OOF),0,1)`，只在 `T_cal` OOF 估计、启动前冻结并按任务/ASIN cluster bootstrap 报区间；`q_P=softmax_C(s+b_c·hat_d)`，同时记录斜率近零和 snapshot 变化的失配。

**14.3 强度匹配。** R-masked 改为 **R-β**：`q_Rbeta=softmax_C(s+beta_c·d)`，按类别在 `T_cal` 冻结系数，以 KL 的中位数和预注册分位数拟合 P 的强度；P-shuffled 也只在合规训练校准数据上匹配强度。分布无法完全匹配须标 `STRENGTH_MISMATCH`，不能宣称已经完全消除强度混杂。

**14.4 对照与主比较。** 训练组 `R/P/R-β/P-shuffled`（B 不更新），主比较 `P−R-β` 和 `P−P-shuffled`；`P−R`、`R−B` 同报。无需以 P 赢 R 为增加对照的条件。

**14.5 P1 仍是训练阶段停损，不是 token mask。** `PASS/FAIL_POWERED/UNDERPOWERED/BLOCKED` 保留用于解释；若所有决策内容类别 `b_c` 区间都包含零，或没有可信 P1 决策信号，则 P 只做 smoke，不花 10/100 更新预算。**单个类别 `FAIL_POWERED` 不再屏蔽其 token**，由 `b_c` 决定其修正幅度。

**14.6 每步当前快照、1 epoch。** 每次 on-policy 迭代以当次 actor 为 teacher/student snapshot，只做 1 个优化 epoch；报告 q_P 修正为零的位置占比和这些位置在 epoch 内的 KL/梯度（避免把先前 minibatch 引起的自蒸馏误称为“完全无更新”）。


## 15. 特权包预检修订（2026-11-20；对应外部建议的 §16′；已合入 §2/§3.1/§7/§8/§9）

**15.1 五种 teacher-as-policy 设置。** 约 100 个 train-only `T_pre` 不同任务，在同一冻结 Qwen2.5-7B 快照、同任务、相同 seed 方案上比较 `z_none/z_full/z_product/z_evidence/z_placeholder`，每种 `sample×2+greedy×1`。`z_placeholder` 与 `z_full` 同格式但全部字段值改“未提供”。`z_product` 包含 gold 商品标题/属性/目标规格（固定包含 z_full 的 gold ASIN、价格与规格）；`z_evidence` 只来自**当前**结果页的候选属性/规格，**不标 gold、不看未来**。奖励与漏斗只用于 `T_pre` 预检包选择，不进 OPD 梯度。

**15.2 同轨迹的 token 内容效应。** 在固定的无包学生轨迹上，按学生候选集比较各包相对 `z_placeholder` 的 `KL(q_x||q_S)`、动作 token 的 `|Δ|` 非零比例和 `d_x-d_placeholder` 的差；`z_product/z_evidence` 还要使用**各自同格式**的评分侧占位包，避免把不同模板导致的分数变化误称特权信息增益。区分 `z_none→placeholder` 的格式影响与 `placeholder→真实包` 的内容影响。占位对照仅审计，不改变原 `d_x` 训练标签。

**15.3 选包与停止条件。** 只从行为层全要求购买率相对无包改善**且**动作层存在超出冻结数值阈值的内容效应的包中，选改善最大者；单次选定、冻结 hash，后续不依据 `T_cal`/V/在线训练重新选择。若全部不符合，记 `PRIVILEGE_NOT_USEFUL`；低覆盖/功效无法下结论则记 `PREFLIGHT_INCONCLUSIVE`，阶段 A.5 停止。不允许为了过关改 rank、奖励、teacher 类型或提示词。

**15.4 与既有算法的关系。** `z_selected` 一旦选定，原本的学生 top-8+采样 token、候选集中心化 `d`、rank=8 双线性投影、`b_c` 校准、R/P/R-β/P-shuffled 相同位置的 KL 及 100-update 历史任务 online projector 规则**全部保留**。这项修订只决定**输入老师的特权信息是否真正有用且在 token 分布上可见**；并不声称 teacher 变强等于蒸馏必然有效。

**15.5 参考文献与研究定位（仅作背景，不能充当本实验实测）。**

- **LEAP / Better than Your Teacher**（ICLR 2025）：以额外状态辅助老师进行轨迹纠正，再以 SFT/DPO 更新学生；与同模 token KL 不同。论文项目：<https://leap-llm.github.io/>。
- **SDAR / Self-Distilled Agentic Reinforcement Learning**（2026 预印本）：将 skill 条件下的 OPSD 作为 RL 辅助目标并门控；本轮**不做 GRPO**。<https://arxiv.org/abs/2605.15155>。
- **PR-OPD / Privileged Representation On-policy Self-Distillation**（2026 预印本）：指出技能能提升老师行为，但输出层 token 变化有限，转而研究隐藏表示；其公开摘要中 WebShop 行为增益与 token/隐藏状态机制数字未必取自相同任务/检查点，**不能直接当作本环境 WebShop token 偏移结论**。<https://arxiv.org/abs/2609.36642>。
- **OCSD / Observation-Calibrated Self-Distillation**（2026 预印本）：通过匹配的 full 与 observation-ablated replay 视图，分离视图结构和观察内容引起的偏移；本协议借鉴**匹配占位格式诊断**，不采用其 GRPO 信号。<https://arxiv.org/abs/2608.04788>。
- **PINT / Privileged Information Internalization for Agentic RL**（2026 预印本，作者公开页面标注 NeurIPS 2026 PretoPost Workshop 接收、主会录用未确认）：同样关注“老师依据特权信息产生学生难以内化的监督”问题，已有方案说明把它与奖励验证、在基础上下文拟合临时老师的路线相区分；我们的技术路线是**同模相对 logit 标签 → student-only rank=8 跨任务投影 → 候选 KL**。正式论文中须进一步核对 PINT 原文方法和具体结果，当前不把 31.6% 等未核实数字写作确定结论。作者论文入口：<https://yilanchen6.github.io/>；论文链接：<https://openreview.net/forum?id=R1peldcnUL>。

**文献不能替代本次 `T_pre` 五设置预检。** 特别是采用不同尺度/不同环境/不同老师包的工作，不能推导出 Qwen2.5-7B 在当前购物环境下的 token 可蒸馏性。

## 16. 执行前定值（2026-11-20）

本节覆盖前文相应旧值。修订在旧模板 8 题预检已报告、旧正式采集已启动之后收到。修订版沿用原冻结任务与种子，独立保存产物、重新测量，不作为旧数据收集前预注册。

16.1 P-pre 以相对 z_none 的采样全要求购买率配对差为主。点估计 ≥ +10 个百分点且任务簇 95% 区间下界 > 0 为改善。区间同时包含 0 与 +10 为 PREFLIGHT_INCONCLUSIVE。点估计 < +10 且上界 < +10 为无改善。其余未明确达到改善或无改善的边界情形保留为 PREFLIGHT_INCONCLUSIVE。动作内容和覆盖要求继续独立判定。

16.2 z_evidence 包长上限 1024 tokens，其他真实包 192。匹配占位仅用于同格式诊断。记录各包溢出率。在非结果页（含商品页规格决策）z_evidence 为 NA。仅当选中 z_evidence 时，所有训练组在这些位置同样不计入该包评分覆盖。其他已选包沿用原共同结构性有效位置。

16.3 P2 使用绝对覆盖门。10 次更新合计，search、click、规格各至少 N_c 个结构性有效 token，且各来自至少 50 个不同任务。阶段 B 以 T_cal 每题一条 rollout、已选包适用域的类别计数，冻结 N_c=max(1,ceil(0.5×10×32×2×该类有效 token 总数/T_cal 不同任务数))。10-update pilot 固定并留存每类 N_c 和分母。若某类结构性不适用，标记覆盖门未满足并披露原因。

16.4 1-update smoke 前冻结 OPD 每次更新 32 个任务、每题 2 条轨迹，四训练组相同。实测吞吐后如需调整须在 pilot 前登记，禁止据 V 结果修改。

16.5 z_product 固定为 z_full 的全部字段（gold ASIN、目标规格和价格约束）加冻结商品目录的 gold 商品标题与关键属性。

§17–§20 的登记时间与版本以[执行协议](../visible_opd_v0/PROTOCOL.md)的提交历史为准。

## 17. 事后修正：按动作计内容门（2026-11-20，已见 A.5 结果）

原始判定 PRIVILEGE_NOT_USEFUL 和全部原始产物保留，与 v2 并列。研究者提出，click 和规格参数常由多个子词构成，少数动作分叉位置可能被照抄位置稀释。按每动作最多一个分叉的模型，条件结构上限为动作数/有效内容 token 数；这是假设下的上限，单凭长度不能证明每个动作最多一个可偏移位置。同时报告实际多命中动作数，避免将该假设写成定理。

只读已完成的 §16 A.5 逐位置评分，按 (goal, rep, step) 的 click 或 option 动作分组；仅采用原有效支持（student、真实包、匹配占位的 mass 均 >=0.25）。一个动作有至少一个有效位置才计入分母。同一位置上 |采样 token 的 content d|>0.02 且 KL(真实包||匹配占位)>1e-5 即为命中。动作门为至少 10% 动作命中并覆盖至少 10 题，保留原 >=100 有效位置、>=20 任务覆盖要求和 §16 行为门。分别报告每类动作 token 数、有效长度、命中数分布和条件结构上限。禁止重新生成或评分 A.5 数据。保留采样行为增益最大者，平局按原顺序；若 full/product 均过选 full。现有 evidence 仅提供首个规格轴名称，标记“未构成有效证据包测试”，不推广到其他证据包。没有合格包即停止。

## 18. 内容标签（2026-11-20，T_cal 采集、OOF 和训练前登记）

固定无包学生候选集 C=top8(S)∪{实际 token}，统一快照、精度、T_score=1。三种前向上下文是无包 S、选定包 sel 和匹配占位 ph；相同前缀、相同字段和插入位置，ph 值均为未提供。计算 Delta_content=log π_sel−log π_ph，d_content=Delta_content−E_{q_S}[Delta_content]。R 定义为内容老师 q_R=softmax_C(s+d_content)。R-β、投影器/OOF、功效埋信号、b_c、任务内位置保留的任务级乱标 P-shuffled、β/γ 强度匹配均使用该新标签。完整动作岭回归使用 sel 对 ph 的动作似然差。等价商品的内容老师以 S 为底在全词表施加 sel 对 ph 的修正，excess 相对无包学生；附报直接 sel−ph 动作 margin。原标签 sel 对 S 单独运行同规格 OOF、b_c、类别 KL，只读对照，不用于系数选择、决策或训练。

P0 检验 softmax_C(s+d_content)=softmax_C(log π_S+log π_sel−log π_ph)；以 ph 替换 sel 时内容修正在容差内为零。旧版 q_R=特权老师条件分布断言不适用于新版。三前向评分预算登记为原两前向的 1.5 倍，同时记录实际耗时；在线阶段沿用。任务池、rank=8、四训练臂、阈值、分折、暂停点不变。阶段 B 仅用 512 个冻结 T_cal 任务各 1 条原生无包 rollout。先检查 8 题的 scorer 对齐，再扩至全池，保留首批。B 完成提交投影、功效、满秩动作岭回归、候选质量、强度校准、等价商品偏好及旧标签对照，随后暂停；本次不更新 actor。

## 19. 阶段 B 诊断修正（2026-11-20，已看到阶段 B 结果后登记；仅 CPU，不改任何冻结的训练量）

19.1 动作级容量参照：原正则网格 {0.1,1,10,100} 相对标准化特征下约 6,800 的训练样本量过小，四折均选中上界，结果记为 GRID_BOUNDARY、不可解读。改为按训练样本量缩放的网格 n_train×{1e-3,1e-2,1e-1,1,10,100}，若选中值落在网格端点则向外扩一个数量级，最多 3 次；报告每折最终 alpha、是否仍在端点、扩网格次数。原结果与新结果并列报告。该参照仍只读，不产生任何训练目标。

19.2 P1 状态：功效检查只限制阴性推断。真实 OOF 相对最强基线改善区间下界 >0 且折间同号率 ≥0.7 时，记 PASS（功效达标）或 PASS_LOW_POWER（功效未达标）；未检出时，功效未达标记 UNDERPOWERED，达标记 FAIL_POWERED。原标签并列保留。P1 状态不控制 token，b_c/β/γ 与 P2 覆盖门均不变。

19.3 描述性拆分：将冻结类别“format”细分为“动作动词（search/click/finish 对应 token）”与“纯格式（括号、Action:、特殊结束符等）”，分别报告 OOF 改善、b 与 KL；不改变训练所用类别与系数。

19.4 执行：先提交本节，再以已保存的阶段 B 数据重算；不新增 GPU 采样或评分，可与阶段 C smoke 并行。

## 20. 阶段 D 预测（2026-11-20，阶段 C 进行中、无任何训练后评测时登记）

20.1 训练侧：四组在固定目标上的 KL 均随更新下降；P、R-β、P-shuffled 的目标强度相近（按 §6 匹配），R 最强。

20.2 行为幅度：10 次更新后，R 在 T_opd_eval 上的行为变化（全要求购买率、决策 token 熵）大于 P；P、R-β、P-shuffled 三组的 V 结果预计都在噪声范围内，V@10 只作描述。

20.3 泄漏读数：R 的决策 token 学生熵下降幅度大于 P；R 的 T_opd_eval−V 差距扩大幅度大于 P。

20.4 若 P 与 P-shuffled 在训练侧和行为侧都不可区分，记为“10 步内投影方向未产生可检测差异”，不作阴性结论。
