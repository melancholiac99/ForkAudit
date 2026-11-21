# VC-OPD v0.4 execution registration

The parent protocol is ../visible_credit_v0/credit_grok_visible_opd_only_v0_4_experiment_plan.md at 5a076ef5a. The current execution boundary is its eight-task throughput check. Never select a packet on this subset. All 100 preflight identities are fixed before generation, the first eight are used once and can be reused with identical code and weights. Freeze the unadapted Qwen2.5-7B-Instruct snapshot. Sampling 0.7, top-p 1, greedy separately. Seeds depend on goal index, replicate and step. Native task, environment and prompt history are preserved.

Use all current result products in page order for evidence. Give the first catalog specification-axis name for each, or 无 for an empty axis list. Evidence is unavailable off results. Use the unchanged native input at unavailable policy steps and record exposure; exclude unavailable pairs from scoring and report coverage. Overflow is never truncated. Withhold formal interpretation for incomplete packets. Freeze templates, tokenizer and parser by file hash. Teacher packets are one user turn after the initial requirements. Scoring uses raw T=1, teacher and student identically loaded, on exact generated IDs including end-of-turn. Behavior metrics aggregate sample and greedy separately, with paired task and product resampling.

For formal selection require 100 complete paired tasks, positive sampled all-requirements lift, paired product and task bootstrap lower limits above zero for a conclusive gain. Nonpositive effects whose upper bounds exceed zero are inconclusive. Eligible content positions are click/option with both candidate masses >=0.25 and matching packet support. Require >=100 positions across >=20 tasks, >=10% of eligible positions across >=10 tasks with |sampled centered content difference|>0.02 and content KL>1e-5. Report tau_delta=tau_d=0.02, independently require empty controls under 1e-5. A missing comparison/insufficient coverage stays inconclusive. Largest confirmed lift wins; ties full/product/evidence. No selection in eight-task smoke.

Offline registration follows parent §§4–7. Unit-normalize final hidden and frozen head rows, rank eight, task/trajectory then category balanced Huber squared-error (transition 1), AdamW 0.001, 10 fixed epochs seed 20261120. Four ASIN folds; training-only centering. Power injection 0.50, recovery >=0.5 and lower task CI >0. Report MSE relative to strongest baseline, require lower CI >0 and >=0.7 same-sign nonnear-zero predictions (near zero 0.02) on same held-out positions. Full-action probe uses first eight visible legal clicks in display order, normalized logprob gap and last action-token hidden, ridge selected within training folds from 0.1/1/10/100. Equivalence audit train-only: all visible legal product clicks, independent native feasibility check of every option configuration, no V privilege. Task-weighted clipped OOF covariance slope, variance <=1e-10 gives zero. Match beta/gamma on .01 grid [0,1]/[0,4] to equally weighted KL quantiles .25/.5/.75/.9 by absolute discrepancy divided by max(target quantile,1e-5); flag any relative discrepancy >.25. Frozen snapshot targets for all structurally valid positions. Online projection uses other ASIN folds from completed batches only, every 10 updates rescore history. Further stages follow parent acceptance points.

Section 16 supersedes conflicting values above. P-pre requires paired sampled lift >=10 percentage points and lower task-cluster CI >0. Upper CI <10 points together with point <10 points is no improvement. An interval covering both zero and ten points, or another unclassified boundary, is inconclusive. Content and coverage gates remain frozen. z_product includes every z_full field, title and attributes. Evidence supports up to 1024 tokens; other real packets support 192. Apply the common results-page scoring restriction across arms only if evidence wins. Freeze the 32 tasks × 2 trajectories OPD batch and the class-wise absolute coverage calculation in section16.py. A new run repeats the same frozen 100 tasks independently. Old results and all costs remain attributed to their original run. Historical template outcomes had been seen before this amendment.

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

## 21. 优化设置与诊断定稿（2026-11-20，阶段 C smoke 后、阶段 D 前登记）

21.1 待检验假设：AdamW 逐参数归一化步长，各组单步参数移动量与目标强度近似无关；对目标离学生较近的组（P、Rβ、P-shuffled），同一步长可能越过目标，表现为固定目标 KL 上升。该假设由 21.3 检验，不预先认定为噪声。注：Rβ 改变的是目标分布（非线性），不等价于缩放 R 的损失，保留为独立对照。

21.2 训练组：R、Rβ、P、P-shuffled（B 只评测）。主比较 P−Rβ、P−P-shuffled；同时报告 P−R。

21.3 诊断：重放原 smoke（同批、同初始 LoRA），保存各组更新后 LoRA 与固定候选集上的更新前后 logits。在步长倍数 {0, 1/8, 1/4, 1/2, 1, 2, 4} × 21.4 所定学习率下，对每组按原训练权重聚合报告：ΔKL = KL(q‖p0) − KL(q‖p1)、对齐量 Σ(q−p0)·log(p1/p0)、移动量 KL(p0‖p1)（恒等式 ΔKL = 对齐量 − 移动量），倍数 0 用于估计数值误差；另报各组 LoRA 更新范数与两两更新方向余弦。

21.4 优化设置沿用 GRPO 基线：AdamW、lr=1e-6 恒定、权重衰减 0、梯度裁剪 1.0、与基线相同的 mini-batch 划分规则与 1 个 epoch（服务器先核实基线每次更新的实际优化器步数，并让 OPD 每次更新的步数与之一致）。唯一的否决条件：若在此设置下任一组的对齐量 ≤ 数值误差，则按 21.3 的倍数网格依次减半，取所有组对齐量均为正的最大倍数，四组共用。弱目标组的 ΔKL < 0（移动量大于对齐量）照实报告，不作否决。不读取 V 或 T_opd_eval。

21.5 以选定设置重跑 smoke 后进入阶段 D；阶段 D、E 全程冻结优化设置。逐步报告：各组在当步训练批、当步冻结目标上的更新前后 KL 及其对齐量/移动量分解；相邻快照在固定探针批上的实际输出移动量。原 lr=1e-6 单步 smoke 结果保留。

### §21.4 执行澄清（核验 GRPO 后、步长网格前）

服务器核验首轮 GRPO 有 624 条动作记录、10 个 AdamW 步。在已保存的 282 条 smoke 记录上，研究者选择保留绝对批宽参照 64 和 1 个 epoch。设 N 为当前外层更新的真实动作记录数，K=ceil(N/64)，将记录按固定种子打乱后尽量均分成 K 份，每条恰好使用一次。四组使用同一划分规则；在共同轨迹上使用同一份划分。原 smoke 对应 57、57、56、56、56 条，共 5 步。阶段 D、E 沿用并逐次记录实际步数。按整批先在轨迹内平均响应位置，再在任务内平均轨迹，最后等权平均任务。大小 m 的随机小批损失为该小批全局加权和乘 N/m；在同一参数点上其期望梯度等于整批梯度。四组使用 AdamW、恒定 lr、权重衰减 0、梯度裁剪 1.0。本轮在新 smoke 结束后暂停。

## 22. 测量底噪修正（2026-11-20，§21 结果后、阶段 D 前登记）

22.1 发现：§21 网格中，LoRA 更新范数随学习率线性增长（×0.125 至 ×4 共 32 倍），而输出移动量 KL(p0‖p1) 在各倍数、各组间基本恒定（约 8–10e-4），逐小批同样如此；对齐量在小倍数下正负波动。判定为 BF16 前向在任何非零权重变化下产生的数值抖动底噪。§21.4 以零倍误差（=0）作噪声基准，未覆盖该底噪，“lr=1e-6 通过”记为 NOT_ESTABLISHED（既未支持也未否定）。

22.2 诊断改为 FP32：p0、p1 及固定探针 logits 用 FP32 前向计算（训练仍为 BF16，与基线一致）。

22.3 空更新对照：在同一初始 LoRA 上施加随机方向、范数与各组真实更新相同的扰动（每组 5 个随机种子），在 BF16 与 FP32 下分别报告对齐量与移动量，作为噪声分布。

22.4 以 FP32 重跑 §21.3 的 ×1、×2、×4 三档（四组）。判定：在 lr=1e-6 下，各组对齐量须高于其空更新分布的 95% 分位数；并报告对齐量对步长的斜率与移动量对步长平方的关系。通过则沿用 lr=1e-6；R 通过而弱目标组不通过时，暂停并报告，不自动改学习率。

22.5 阶段 D 的训练侧监测统一使用 FP32 探针，并以 10 次更新的累积变化判断；单步 BF16 读数仅作记录。

22.6 方向余弦（参数空间）不受输出抖动影响，§21 的余弦结果有效并保留。

## 23. 阶段 D 前的交叉对齐诊断与监测口径（2026-11-20，§22 后、阶段 D 前登记）

23.1 §22 结论登记：BF16 下约 1e-3 的输出移动量为数值底噪（范数匹配随机扰动在 FP32 下为约 2e-9）；FP32 下 lr=1e-6 四组固定目标 KL 均下降，无越过目标；§21 中“弱目标组越过目标”的观察不成立（仅在 ≥2 倍步长出现）。lr=1e-6 冻结。

23.2 交叉对齐矩阵（仅诊断，已保存的 1 倍适配器，FP32）：对更新组 u ∈ {R, Rβ, P, P-shuffled} 与目标组 t ∈ 同集合，报告把 u 的更新作用于初始模型后，在 t 的冻结目标上的 ΔKL、对齐量与移动量（4×4）。另报“u 在 t 上的 ΔKL / t 在自身目标上的 ΔKL”比值。预期：R→P 明显为正；P-shuffled→P 接近 0。结果不改变任何训练设置。

23.3 阶段 D 监测：每次更新在 FP32 下记录各组在当步冻结目标上的 ΔKL 及对齐量、移动量分解，相邻快照在固定探针上的 FP32 移动量；十次更新结束后报告首尾直接移动量。BF16 单步读数仅留档。阶段 D 完成后暂停，提交训练侧报告，以及 V@0/10、T_opd_eval@0/10 的行为描述。

## 24. 阶段 D 后的训练机制诊断（2026-11-20，已见前 7 步监测，行为评测前登记）

D 按原冻结设置跑满 10 步并完成行为评测。之后进行下述三项只读或独立重放诊断，暂停在 E 前。

1. 四组逐步列出 FP32 更新前 KL、训练精度下的目标强度和更新前 KL。核对目标评分器和轮初 actor 的适配器哈希、原生前缀与响应、有效候选 ID/掩码、候选 logits 及无包候选分布。独立还原每个轮初 actor 并在 BF16 同精度路径复算，按所有结构性有效位置汇总实际差值，区别评分器存档和重建证据。不同批次的目标强度可变，KL 的趋势单独不判快照过期。
2. 在 Rβ、P、P-shuffled 各自第 2、4、7 步的真实轮初权重上，保留原始数据、目标、分批、权重、BF16 精度、恒定学习率、裁剪与单个 epoch，仅将 AdamW 一二阶状态与内部步数一并清零。九个分支各自独立，报告 FP32 更新前后 KL、对齐量和移动量，与对应历史保留分支配对。重放不改 D 的快照或行为结果。状态清零的对比不能单独归因于一阶动量。
3. 各组每个轮初参数点上，冻结对应批次的候选和目标，对完整批次按原任务/轨迹/位置权重求未裁剪的 KL 梯度，参数坐标为同一组 LoRA 可训练张量。四组均报告 1→2 至 9→10 的相邻梯度余弦和两侧 L2 范数，点积与范数以 FP64 聚合，零范数记 NA。相邻比较同时跨越批次和快照。上述诊断独立于 V 与 T_opd_eval，不用于本轮调参。若确证过期快照，保留原结果、修复并独立重跑 D；其他归因按诊断结果讨论，E 不自动启动。

### §24 事后补充（2026-11-21，十步训练读数已知，行为评测进行中，三项诊断出数前）

保留上述三项原指标和顺序门。对四组全部十步，在完全相同的轮初适配器、原生前缀和有效候选上测量 KL(p_BF16‖p_FP32)，沿用每轮原任务和轨迹权重。由原目标 q 计算 c = log q − log softmax_C(s_BF16)，并以 q_FP32 = softmax_C(s_FP32 + c) 重锚。第 2–10 步使用逐步存档 bf16_before.pt，第 1 步采用快照核对中的重建锚点。原目标的零质量与无效候选保留零。四组十步的历史更新与指定九个重置 AdamW 分支均在原指标及重锚指标下报告更新前后 KL、对齐量和移动量。相邻梯度余弦维持原训练精度与口径。

vLLM 与 HF 的行为端对比使用每个已评测策略（B@0 以及四组 @10）的同一个固定原生评测批次（从第 992 条开始、32 条，横跨 V 和 T_opd_eval，含贪心和采样）。重放原前缀、动作和随机种子，校验原行动摘要，比较原生 vLLM 与 HF BF16、HF FP32 对所生成 token 的全词表、温度 1 对数概率，分开报告解码、类别及任务池。按实际匹配情况报告数目与差异。此项为描述性抽样。所有补充均不参与训练或学习率选择。完成行为评测、原诊断、事后补充及远端文件验收后再提前退租，保留系统盘与数据盘。

## 25. 阶段 E 的组别、移动量对照与监测（2026-11-21，事后修改：已见阶段 D 全部训练和行为结果以及 §24 诊断）

25.1 组别。E 训练 R 和 P，B 只评测，不再运行 R-β 和 P-shuffled；按 25.3 判定是否增设 R-matched。D 中 P−P-shuffled 在 T_opd_eval 和 V 上分别为 +6.62 [+4.25, +9.00] 和 +7.12 [+5.00, +9.38] 个百分点，且 P-shuffled 相对 B 有害，是负对照而非中性对照，它对应的问题 D 已回答。R-β 匹配的是目标 KL，在 Adam 下控制不了实际输出移动量（D 中 P 的累计 FP32 移动量约为 R-β 的 2.6 倍）。D 中这两组的结果原样保留，报告中引用。单种子，按 v0.4 §6.3 属探索性结果。

25.2 移动量核验（不训练，先于 E）。取 D 第 1 步的共同批次（全部由 B 生成）的全部结构性有效位置，用 HF BF16 计算四组第 10 步相对第 0 步的全词表 KL(π_0‖π_10)，按原任务和轨迹权重汇总，并分 click/商品 ID、option、search、动作动词、语法符号、其他六类位置报告。同时报告 FP32 读数，以及一个与 P 第 10 步 LoRA 改变量 Frobenius 范数相同的随机扰动在同一路径下的 KL，作为底噪参照。记 m_X 为 X 组 BF16 下全部位置的加权均值，ρ = sqrt(m_R / m_P)。

25.3 移动量匹配组。若 0.8 ≤ ρ ≤ 1.25，R 本身即作移动量对照，E 不增组。否则增设 R-matched：目标与 R 完全相同，只把学习率改为 1e-6/ρ。先从第 0 步检查点做 10 次更新的校准运行，按 25.2 的口径复测得 ρ′；若 ρ′ 不在 [0.8, 1.25] 内，按 lr ← lr/ρ′ 调整一次后冻结，此后不再调整。校准只用训练任务和上述共同前缀，不读 V 和 T_opd_eval。若最后又调整了一次学习率，E 第 10 步复核 R-matched 和 P 的实际移动量比值 sqrt(m_R-matched/m_P) 是否落在 [0.8, 1.25]，照实报告，不再调参。E 的各检查点继续报告移动量。

25.4 主比较。有 R-matched 时，主比较为 P−R-matched 和 P−B；没有时为 P−R 和 P−B。其余比较照常报告。本条取代 v0.4 §6.3 和 P3 中以 P−R-β、P−P-shuffled 为主比较的规定。

25.5 目标构造与投影器刷新。E 沿用 D 的目标构造：每次更新以当次 actor 的学生 logits 为锚，加上当次修正。R 用当次老师的内容修正，P 用当前版本投影器的 b_c·d̂。这种构造会沿同一修正方向逐次累加，报告中称为“沿修正方向反复推进”，不称为“向固定老师分布收敛”。P 的投影器保持 D 的设计（rank-8 双线性、Huber、按类别冻结的 b_c），按 v0.4 §4.5 每 10 次更新在线刷新一次。刷新时必须先用当次 actor 快照，对缓冲区里的历史轨迹重新计算老师标签和学生表示，再重新拟合，不使用混合快照近似。若对全部历史重算的成本过高，改用滑动窗口，窗口大小在 E 启动前按实测成本确定并登记。实施前先实测刷新成本。

25.6 监测（每 10 次更新一次，与评测点对齐）。在 25.2 的共同前缀上用 HF BF16 报告：相对第 0 步的全词表 KL（分六类位置）；决策位置的平均熵和 top-1 概率；KL(p_BF16‖p_FP32)；P 相邻两个投影器版本在同一批前缀上的修正差异（b_c·d̂ 的均方差和余弦）。每次更新继续报告两套读数——BF16 锚定和 FP32 重锚——下的更新前后 KL、对齐量与移动量。训练侧以 BF16 为准，FP32 重锚只作方向检查。

25.7 停止条件。若某组训练轨迹的合法动作率比第 0 步低 10 个百分点以上，且连续两个监测点如此，停止该组，以最后一个检查点作终点评测，并标记 EARLY_STOP。不得以 V 或 T_opd_eval 的结果触发停止。

25.8 LoRA 死区检查（不训练，只描述）。在 P 和 R 的第 10 步检查点上，给每个 LoRA 层挂钩，记录基座输出 y_bf16、FP32 计算的 LoRA 增量 δ，以及实际生效的增量 Δ = bf16(y_bf16 + δ) − y_bf16。按层类型报告 ‖Δ‖/‖δ‖、cos(Δ, δ)，以及 δ≠0 但 Δ=0 的元素比例。

25.9 离线并行（以 CPU 为主，不阻塞 E，不改变 E 的任何训练设置）。
(a) 把 format 类拆成动作动词和语法符号两类，重新报告阶段 B 的可预测份额；b_c 仍按原类别冻结。
(b) 分布空间投影器（作为下一版候选，不进入 E）。在 T_cal 上沿用同样的 rank-8 结构、输入和 ASIN 四折，以 KL(sg(q_R)‖softmax_C(s + f_φ)) 为损失拟合投影器；每个类别在 OOF 上按同一前向 KL，在 [0, 1] 内选一个收缩系数。与现有投影器（Huber + b_c）比较题外加权 E[KL(q_R‖q)]，给出任务簇 bootstrap 区间，并分类别报告。若分布空间版的改善区间下界 > 0，就按 D 的条件另跑一个 P-KL 试点：同一第 0 步检查点、冻结投影器、相同任务顺序和种子、10 次更新、同样的评测，结果与 D 各组第 10 步并列报告。该试点在 E 之后或有空闲卡时运行，不改变 E 的设计。
(c) 报告中“约 5% 可见份额”一律改为“当前预测器（rank-8、512 题、Huber）的题外可预测份额”。

25.10 其余设置不变：LoRA r=64、alpha=64、AdamW、lr 1e-6（R-matched 除外）、梯度裁剪 1.0、单 epoch，评测按 v0.4 P3 在第 0/10/…/100 步进行。

25.11 移动量匹配的事后修改（2026-11-21，已见 §25.3 首次十步校准，正式 E 已启动）。暂停正式 E 原学习率 1.30005955e-6 的 R-matched 分支，保留其已有记录，标记作废，排除于所有正式比较。R、P 延续既有检查点和优化器状态。§25.3 的平方根换算假设 BF16 移动量与学习率平方成正比；在本次 BF16 死区条件下，学习率约 ×2.5 时实测移动量约 ×23，呈强超线性。两点幂律预计 1.30e-6 的平方根比约为 0.63，仍在匹配带外，此数为外推模型的预测。用实测的两点（1e-6，0.001967）和（2.4966e-6，0.045216）在 log(lr) 与 log(m) 上线性插值到 m_P = 0.012261，以未舍入的存档值计算，取 lr ≈ 1.71e-6。从共同第 0 步检查点独立做第二次 10 步校准；依 §25.2 在共同前缀复测 ρ″ = sqrt(m_R-matched/m_P)。无论结果如何，冻结此学习率；若 ρ″ 不在闭区间 [0.8, 1.25]，标记 MOVEMENT_MISMATCH。然后从第 0 步检查点和全新优化器启动正式 E 的 R-matched。保留第 10 步的正式复核和其余监测，仅作报告。校准不使用 V 或 T_opd_eval。

25.12 执行顺序和布局的事后修改（2026-11-21；登记前未查看正式 E 的任何训练后行为评测）。正式 E 先单独跑 P，四张卡全部用于 P，包括 vLLM 采样和行为评测。采样和评测改为四个引擎数据并行。任务到卡的分配按长度均衡确定一次后冻结，之后所有组、所有更新、所有评测都用这同一份分配；种子规则不变。P 从第 0 步检查点和全新优化器重新开始；此前在轮流调度下跑出的 P 和 R 更新保留记录，标为作废，不进入比较。R-matched 的第二次校准照常完成，学习率按 §25.11 的校准结果冻结。B@0 在新布局下重新评测，作为所有比较的基线。P 跑满 100 次更新（或触发 §25.7 的停止条件）后，依次在同一布局下从第 0 步运行 R-matched 和 R，设置全部按已冻结的内容执行，不因 P 的结果做任何改动。提前终止规则：若 P 第 100 步在 T_opd_eval 上的采样全要求购买率不高于 B，就不再运行 R-matched 和 R，报告中注明 not run；不得以 V 作此判断。

25.13 采用已完成更新与布局切换（2026-11-21，经确认的事后修订；新布局的正式 P 尚未更新，未查看任何正式训练后的行为评测）。撤销 §25.12 对轮流调度下 P、R 前五步的作废决定，在本版正式比较中采用已完成的 P@1–5 以及条件性采用 R@1–5。P 从已保存的 P@5 LoRA 权重和 AdamW 完整状态续跑，下一次更新为第 6 步，不重跑前五步。前五步为旧布局，第 6 步起采样、评分、训练及评测按已冻结的四卡任务分配执行。保留新布局的 B@0 行为评测，作为 P@100 门和各正式比较的共同基线。若控制组获准运行，R 使用同样的旧布局前五步并从现有 R@5 续跑；R-matched 以 §25.11 冻结的学习率完成同口径旧布局前五步，从第 6 步切换到与 P、R 相同的布局。对相同学习率、初始快照、任务顺序、种子、实际执行布局及优化器轨迹均已核实一致，并保有可恢复状态的已有计算，优先采用，记录来源及签名。不能将不同学习率的权重或 AdamW 状态用作替代。首十步报告逐步注明布局；P 刷新时将已采用的历史轨迹纳入同快照重评分。训练合法动作率的第 0 步参考沿用产生实际首步训练批的 B；行为评测用新布局 B@0。§25.12 的 P → R-matched → R 顺序、仅按 T_opd_eval 采样全要求购买次数判断的 P@100 门、§25.7 停止条件及其余冻结设置均沿用。
