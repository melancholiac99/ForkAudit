# R 与 λ=1.5 的长度触顶轨迹审计

事后探索。四份归档的 SHA-256 均与评测回执一致。分母为回答，AIME24 为 30 题 × 12，MATH-500 为 500 题 × 2。样本之间共享题目。仅使用已保存的文本、官方判分函数与原分词器。

## 全量计数与预算

| 模式 / 模型 | 触顶 / 全部 | 触顶 token / 全部预算 | 触顶长度范围 | 未触顶 min / Q1 / 中位 / Q3 / P95 / max | 触顶最终答对 / 有答但错 / 缺答 | 有 explicit 正解 | 正解后 token 合计 |
|---|---|---|---|---|---|---|---|
| aime24 R@100 | 4/360 | 155,648 / 2.2% | 38912–38912 | 5,454 / 12,675 / 19,147 / 26,496 / 33,876 / 37,706 | 0 / 1 / 3 | 1 | 596 |
| aime24 R-lambda15@100 | 19/360 | 739,328 / 9.4% | 38912–38912 | 4,711 / 13,854 / 20,890 / 26,721 / 35,094 / 38,800 | 1 / 3 / 15 | 1 | 2621 |
| nonthinking R@100 | 72/1000 | 1,179,648 / 30.4% | 16384–16384 | 13 / 623 / 1,430 / 4,064 / 9,724 / 16,001 | 0 / 0 / 72 | 6 | 50140 |
| nonthinking R-lambda15@100 | 125/1000 | 2,048,000 / 52.9% | 16384–16384 | 13 / 550 / 1,018 / 2,364 / 8,221 / 16,221 | 2 / 8 / 115 | 2 | 18032 |

缺答沿用归档格式标志和缺少 boxed 的哨兵。对有答案的触顶记录，已用官方 grade_answer 复核最终分数。显式答案用原 first / box / grade_answer，逐个候选继续查到最早匹配 gold 的位置。没有被该提取器识别的口头结论、裸数、变量赋值可能漏检；中间量也可能恰好等于 gold。详见下文推导核查。token 位置是对归档文本重新分词的一基位置，剩余量按同一分词计算。归档生成长度与重分词长度分别保存。

## 触顶回答的进度统计

每个单元格依次为四个等 token 数区间。重复 5-gram 指此前出现过的原分词器五元组；段落按空行分隔、折叠空白、在段尾归档。其余列按每千 token 计。数学表达以 LaTeX 命令和二元运算符计数；反思词为 Wait、But、Hmm、Alternatively。角色标记为显式角色分隔符或行首角色名；语言切换为相继识别到的不同语言提示词（英、西、越、法、中、俄），另附脚本切换。词表代理会漏掉同一文字体系内的无标志切换。数学符号、反思词或重复率单独均不足以判断推理是否停滞。

| 模式 / 模型 | 重复 5-gram % | 重复段落 % | 数学 | 反思 | 角色 | 语言提示切换 | 文字体系切换 |
|---|---|---|---|---|---|---|---|
| aime24 R@100 | 36.05 / 42.39 / 50.15 / 57.18 | 10.55 / 9.20 / 12.15 / 13.47 | 69.34 / 89.36 / 69.85 / 90.95 | 15.29 / 13.57 / 9.56 / 10.38 | 0.00 / 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 / 0.00 |
| aime24 R-lambda15@100 | 41.76 / 55.43 / 60.39 / 63.02 | 34.17 / 46.75 / 47.40 / 49.68 | 60.41 / 74.59 / 85.66 / 75.17 | 17.97 / 13.02 / 12.31 / 13.73 | 0.00 / 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 / 0.00 |
| nonthinking R@100 | 55.37 / 73.83 / 78.86 / 81.83 | 26.37 / 63.73 / 72.95 / 76.60 | 67.74 / 67.62 / 64.07 / 62.57 | 10.11 / 7.56 / 7.00 / 6.75 | 0.00 / 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 0.74 / 0.18 / 0.10 / 0.10 |
| nonthinking R-lambda15@100 | 71.53 / 91.91 / 95.27 / 96.20 | 60.56 / 86.76 / 91.90 / 93.26 | 120.22 / 118.42 / 118.92 / 117.88 | 2.26 / 1.57 / 1.51 / 1.42 | 0.05 / 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 0.49 / 0.59 / 0.55 / 0.55 |

## 人工审阅与各类预算

AIME24 触顶全查；MATH 各模型按归档顺序、Python Random(20261124) 均匀无放回抽 8 条，不按题目、正误或重复率筛选。表中出现率以人工审阅回答为分母。预算采用粗分段估计：原文按字符四等分，各段结合中点、相邻位置和结尾判读，再把边界换算成原分词器 token 数。P 为尚在推进的有效中间步骤，N 为重复尝试且无可见进展，D 为失去实质内容的语言循环，U 为证据不足或同段混合。D 段也计入“无进展”列。一次新枚举、发现算错或尝试不同证明可以计作推进。出现率可重叠；预算是段级估计，不是逐 token 人工标签。正确答案后继续的预算独立以首次 explicit 正解末尾计算，与各段可重叠。

| 模式 / 模型 | 审阅 n | 推进：条数 (%) / token | 无进展（含失调）：条数 (%) / token | 语言失调：条数 (%) / token | 不确定：条数 (%) / token | 显式正解后：条数 (%) / token | 审阅 token 总数 |
|---|---|---|---|---|---|---|---|
| aime24 R@100 | 4 | 4 (100.0%) / 77,013 | 2 (50.0%) / 21,439 | 0 (0.0%) / 0 | 4 (100.0%) / 57,196 | 1 (25.0%) / 596 | 155,648 |
| aime24 R-lambda15@100 | 19 | 17 (89.5%) / 297,265 | 15 (78.9%) / 225,713 | 3 (15.8%) / 98,908 | 17 (89.5%) / 216,346 | 1 (5.3%) / 2,621 | 739,324 |
| nonthinking R@100 | 8 | 7 (87.5%) / 42,344 | 7 (87.5%) / 54,935 | 1 (12.5%) / 11,922 | 7 (87.5%) / 33,793 | 0 (0.0%) / 0 | 131,072 |
| nonthinking R-lambda15@100 | 8 | 2 (25.0%) / 7,544 | 8 (100.0%) / 108,179 | 1 (12.5%) / 16,384 | 4 (50.0%) / 15,349 | 0 (0.0%) / 0 | 131,072 |

## 逐条对照

题序为归档零基索引；MATH 并行引擎的原 problem_id 会重复。对照取同题同模型首个未触顶样本，若无则用另一模型，再查 B 同题归档，仍缺则明示。此对照描述同题可行路径，未按成功筛选。[原文与位置](review_evidence.json)同时给出各进度片段、正确答案上下文及对照末尾。

| 模式 | 模型 | 题序 | 原 ID | 采样 | 分段 | 判断依据 | 同题未触顶 |
|---|---|---|---|---|---|---|---|
| aime24 | R@100 | 5 | 65 | 5 | P/P/P/U | Changes coordinates and determinant representations, then assumes an unverified radius and boxes 76. | R@100 #0，34,493 token，错 |
| aime24 | R@100 | 10 | 70 | 5 | P/U/U/N | Returns to a quadratic that contradicts the rectangle geometry and substitutes guesses without resolving the contradiction. | R@100 #0，22,817 token，错 |
| aime24 | R@100 | 20 | 80 | 0 | U/P/U/N | Finds a useful divisibility relation but ultimately guesses from small bases. | R@100 #1，30,979 token，对 |
| aime24 | R@100 | 20 | 80 | 8 | P/U/P/P | Uses coprimality of consecutive factors and new CRT cases. Enumeration remains incomplete. | R@100 #1，30,979 token，对 |
| aime24 | R-lambda15@100 | 4 | 64 | 1 | P/P/U/N | The lifting congruence is useful; abandoning 17 for 41 loses the least-prime condition. | R-lambda15@100 #0，15,094 token，错 |
| aime24 | R-lambda15@100 | 5 | 65 | 0 | U/P/N/N | Reuses incompatible volume and face areas. The four faces are congruent. | R-lambda15@100 #1，30,911 token，错 |
| aime24 | R-lambda15@100 | 5 | 65 | 2 | P/P/U/N | A correct coordinate from the distance equations precedes repeated returns to the same unresolved volume. | R-lambda15@100 #1，30,911 token，错 |
| aime24 | R-lambda15@100 | 5 | 65 | 5 | U/P/U/N | Tries a determinant and repeatedly recombines earlier inconsistent surface and volume calculations. | R-lambda15@100 #1，30,911 token，错 |
| aime24 | R-lambda15@100 | 5 | 65 | 8 | P/U/P/N | Explicit vector products advance a method. The same incompatible radical is repeatedly presented in the tail. | R-lambda15@100 #1，30,911 token，错 |
| aime24 | R-lambda15@100 | 5 | 65 | 9 | P/P/U/N | Coordinates and cross products give intermediate constraints; eventually assumes a result and restarts after boxing 85. | R-lambda15@100 #1，30,911 token，错 |
| aime24 | R-lambda15@100 | 5 | 65 | 10 | P/P/U/P | Uses simple test vectors and corrects a determinant arithmetic error just before cutoff. The final radius is unresolved. | R-lambda15@100 #1，30,911 token，错 |
| aime24 | R-lambda15@100 | 5 | 65 | 11 | P/U/P/U | Checks faces and coordinates; the incongruent computed face areas contradict the given disphenoid. | R-lambda15@100 #1，30,911 token，错 |
| aime24 | R-lambda15@100 | 15 | 75 | 3 | P/U/D/D | Inclusion-exclusion gives way to a continuous bare “let’s” loop. | R-lambda15@100 #0，14,067 token，对 |
| aime24 | R-lambda15@100 | 17 | 77 | 0 | P/U/N/U | Derives symmetric constraints and checks cases, but keeps switching among incompatible counts. | R-lambda15@100 #1，26,632 token，错 |
| aime24 | R-lambda15@100 | 17 | 77 | 7 | U/P/U/N | Finds the family with one coordinate 100; counts permutations of both x and 200−x twice. | R-lambda15@100 #1，26,632 token，错 |
| aime24 | R-lambda15@100 | 20 | 80 | 1 | P/U/U/N | Contradictory counts of solutions for base 16 persist through the final guess. | R-lambda15@100 #0，21,713 token，对 |
| aime24 | R-lambda15@100 | 20 | 80 | 3 | P/P/P/U | Enumerates genuinely distinct residues and corrects a divisibility count; no proof of the first qualifying base. | R-lambda15@100 #0，21,713 token，对 |
| aime24 | R-lambda15@100 | 20 | 80 | 7 | P/U/P/N | Tests additional bases and divisibility cases before reverting to unsupported guesses. | R-lambda15@100 #0，21,713 token，对 |
| aime24 | R-lambda15@100 | 20 | 80 | 11 | P/P/U/N | Small-base calculations test the congruence but the tail recycles the same known counts. | R-lambda15@100 #0，21,713 token，对 |
| aime24 | R-lambda15@100 | 23 | 83 | 1 | D/D/D/D | Equation fragments collapse to alternating “Let me note / think / take” scaffolds. | R-lambda15@100 #0，24,385 token，对 |
| aime24 | R-lambda15@100 | 23 | 83 | 3 | D/D/D/D | An unfinished equation prompt collapses to bare “Let me think” paragraphs. | R-lambda15@100 #0，24,385 token，对 |
| aime24 | R-lambda15@100 | 23 | 83 | 11 | U/P/P/U | Enumerates constrained digit cases, reaches 45 and starts writing up the solution. | R-lambda15@100 #0，24,385 token，对 |
| aime24 | R-lambda15@100 | 27 | 87 | 6 | P/P/N/U | Tests the four digit replacements, returning to equivalent congruences; checking additional candidates alone is not a loop. | R-lambda15@100 #0，26,604 token，对 |
| nonthinking | R@100 | 9 | 2 | 1 | P/U/P/N | Eventually repeats a parenthesization verbatim, even labelling it “Already counted”. | R@100 #0，12,126 token，错 |
| nonthinking | R@100 | 26 | 6 | 0 | P/P/U/N | The gap construction is promising; conflicting counts recur without consistently eliminating rotations. | R@100 #1，1,308 token，错 |
| nonthinking | R@100 | 110 | 27 | 1 | P/U/N/N | Enumerates a fractional sequence repeatedly despite the requirement of positive integers. | B #0，8,459 token，错 |
| nonthinking | R@100 | 120 | 30 | 1 | P/D/D/D | Starts with parity of composition; tail alternates unfinished “define” and “take” prompts. | B #0，662 token，对 |
| nonthinking | R@100 | 136 | 34 | 0 | P/P/U/U | Verifies feasibility at 64, then tries to prove uniqueness. Feasibility alone is insufficient. | R-lambda15@100 #0，6,962 token，错 |
| nonthinking | R@100 | 381 | 95 | 1 | P/P/U/N | Correct region decomposition reappears alongside incompatible integrated areas. | R@100 #0，14,227 token，对 |
| nonthinking | R@100 | 401 | 100 | 1 | U/N/N/N | Vietnamese prose repeats the same tautological complex-modulus equation. | R@100 #0，4,493 token，错 |
| nonthinking | R@100 | 425 | 106 | 0 | P/U/N/N | Useful centroid identity gives way to repeated expansions of the same dot products. | R-lambda15@100 #1，8,184 token，错 |
| nonthinking | R-lambda15@100 | 9 | 2 | 0 | U/N/N/N | Repeats an inadmissible all-addition expression without adding a parenthesization. | R-lambda15@100 #1，1,522 token，错 |
| nonthinking | R-lambda15@100 | 50 | 12 | 0 | P/N/N/N | Checks an integer trial then repeatedly boxes an intermediate bag size rather than total coins. | R@100 #0，8,827 token，对 |
| nonthinking | R-lambda15@100 | 112 | 28 | 0 | N/N/N/N | The same unresolved radical identity and “Try” are copied across the response. | R-lambda15@100 #1，1,663 token，对 |
| nonthinking | R-lambda15@100 | 328 | 82 | 0 | U/N/N/N | Repeats a cases expression with identical positive-integer conditions. | R-lambda15@100 #1，3,365 token，对 |
| nonthinking | R-lambda15@100 | 333 | 83 | 1 | N/N/N/N | The Spanish transition “De aquí” introduces exactly the same continued fraction. | R@100 #0，6,679 token，错 |
| nonthinking | R-lambda15@100 | 400 | 100 | 1 | D/D/D/D | A cyclic chain of bold sentence fragments repeats the tangency description without constructing a relation. | R-lambda15@100 #0，10,666 token，错 |
| nonthinking | R-lambda15@100 | 425 | 106 | 0 | P/U/N/N | Starts with a centroid identity, then copies an identical sum of squared sides. | R-lambda15@100 #1，8,184 token，错 |
| nonthinking | R-lambda15@100 | 489 | 122 | 1 | U/N/N/N | Restates f(x)=30−x repeatedly instead of evaluating the endpoint minimum. | R-lambda15@100 #0，3,572 token，对 |

归档校验、触顶和未触顶的均值与分位数见 [lengths.json](lengths.json)。全部 220 条的题目、最终得分、缺答、首次正确和生成进度指标见 [capped.json](capped.json)。复现顺序为 cap_audit.py、supplement.py、report.py；人工判断固定于 readings.tsv。

## 正确答案有没有依据

- **推导成立，仍继续生成**。AIME24 网格题（原 ID 83，λ=1.5，采样 11）把下排写成上排各位的 9 的补数，得到上排三位之和为 8，故有 C(10,2)=45 种。三位非负且总和 8，数字上限自动满足。原 explicit 口径在 token 36,291 识别到 45，之后又写了 2,621 token。相同模型同题有一条在 24,385 token 正确结束；另两条触顶样本则在复写 “Let me think”。
- **同一数字，不同证明状态**。AIME24 进位制题（ID 80，R，采样 8）在 38,316 token 提出 211。210 有四个不同质因子，确实给出了合理候选；回答自己写明 “this is speculative”，该段没有完成最小性证明，随后又用掉 596 token。同题 R 的一个未触顶样本在 30,979 token 正确结束。
- **explicit 会漏检**。AIME24 三元组题（ID 77，λ=1.5，采样 7）在 token 30,887 算出 601。取无序代表 (100,x,200−x)，x=0…99 各有六个排列，x=100 有一个，合计正确；尾段又把对称的 x 重算，继续了 8,025 token。计数步骤有依据，局部片段不足以独立验证对所有解的完备性证明。把人工确认的这一条加入，λ=1.5 的 AIME 触顶回答中至少 2/19 已提出正确答案，两个对应尾段合计 10,646 token。机器口径与人工补充分别保留。[原文](manual_correct.json)
- **MATH 的缺答也可发生在有效解之后**。六车三车道（R，归档题序 251）已用容斥得到 3⁶−3×2⁶+3=540，之后又写了 12,984 token，最终缺答。单位换算（λ=1.5，题序 200）正确推出 42 baps = 70 yaps = 40 daps，并在 token 1,197 装框，余下 15,187 token。两例都有可核查的推导。全量 explicit 命中没有逐一证明审定。
- **也有无进展且非常昂贵的尾段**。MATH 的金币题（λ=1.5，题序见逐条表）反复装框 x=29，但题目求原有金币总数。同题未触顶样本算出 7×29=203。几何题数次出现相同的点积或根式，只有回看原文确认相同表达式没有增加约束时才计入无进展。越南语、西班牙语样本的标记依据是原式反复原样出现，单凭语言变化不能作此判断。

MATH 非思考下，λ=1.5 的未触顶中位长度比 R 短（1,018 对 1,430），触顶样本却消耗了它全部生成 token 的 52.9%。其触顶样本末四分之一的段落中，93.26% 在同一回答中已经出现过。人工抽样定位到多处整段停滞。AIME 思考下则见到有效的新枚举、算术纠错、带依据的正确答案和纯语言循环共存；λ=1.5 的 19 条触顶中有 7 条来自同一四面体题。两种评测不能共用一个触顶解释。人工预算为单次审阅的粗估，MATH 每组 8 条的分类比例不宜当成全体触顶回答的精确比例。
