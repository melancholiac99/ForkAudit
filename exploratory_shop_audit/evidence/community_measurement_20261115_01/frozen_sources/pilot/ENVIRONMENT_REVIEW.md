# ShopSimulator 环境选择复核（2026-11-15）

**正式采集继续暂停。** 旧适配／前检代码和快照保留，空 query 的旧目标快照隔离，不能用于正式采集。原版奖励未被删改。已核查官方补充数据与社区版，实际跑完三个社区环境回合，并完成独立进程恢复检查。**运行通路通过不等于原研究口径可直接迁移；正式环境选择仍需确定保留原奖励还是改用社区 Reward v3。**

## 1. query 到底是什么、从哪里来

这里至少有三个不同的量，不能混用：

| 名称 | 来源和用途 |
| --- | --- |
| 商品元数据 `product['query']` | 原 WebShop loader 从商品文件直接读取，goal 复制该字段，旧奖励比较两个商品的 query 是否相等。不是学生本步搜索文本 |
| `search[...]`／工具参数 `query` | 学生实际发出的检索字符串；来自动作，与商品元数据独立 |
| 社区文档的任务 “Query” | 用户需求／instruction 文本，不是恢复丢失元数据的来源证明 |

原 WebShop 的 loader 要求数据中已有 `query`；当前已核查源码没有从 category、domain 或任务文本生成该元数据的过程。ShopSimulator loader 改为缺失时填空，但保留 query 相等的奖励分支。**在所检查的官方发布物中没有找到可验证的 query 来源或恢复映射。** 不能把 domain_zh、类别名称、任务要求或当前检索词任意赋给它并声称恢复了原语义。[原 WebShop loader](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site/engine/engine.py)，[固定 ShopSimulator loader](https://github.com/Styleofm3/ShopSimulator/blob/475837e7a51737c44c0a9f79f9459be1e79060fa/shop_env/web_agent_site/engine/engine.py)。

实际核查：

- 固定原版 gzip：23,421 商品，全部无 query。
- 官方 Hugging Face `fine_items_train_standard.jsonl`：21,962 train 商品，全部无 query，逐项与 gzip 对应商品 **21,962／21,962 完全相同**；不是修补后的商品集。文件 SHA256 已与固定 HF revision 的 LFS 元数据核对。
- 社区随库 gzip 与原版 gzip **字节哈希完全相同**。社区没有提供一份带 query 的修正版数据。

官方数据 revision：`ab34b2898059b21e09794de17e61684f1baeb50d`。[固定官方训练补充文件](https://huggingface.co/datasets/wpei/ShopSimulator/blob/ab34b2898059b21e09794de17e61684f1baeb50d/fine_items_train_standard.jsonl)。没有读取官方 eval 轨迹或执行 eval 任务；全商品库仍可用于检索。

## 2. 商品文件和索引如何产生

社区 `scripts/setup.sh` 明确将随库 gzip 直接解压成 `items_eval_train.json`，没有补 query、重写商品或合成任务的转换步骤。解压字节 SHA256 为 `57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f`，本地与脚本预期一致。[固定安装脚本](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5/scripts/setup.sh)。

原版引用 Lucene／Pyserini，却没有附 `search_engine`。官方 [issue #1](https://github.com/ShopAgent-Team/ShopSimulator/issues/1) 也报告该缺失，检查时仍开放且无回复。作为祖先实现的 WebShop 包含转换与索引脚本：把标题、描述、首个 bullet 和规格串合成文档，调用 Pyserini Lucene 索引；它不是 ShopSimulator 已提供或核验过的中文索引。[WebShop 转换脚本](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/search_engine/convert_product_file_format.py)，[索引脚本](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/search_engine/run_indexing.sh)。

社区版使用其自己的 `scripts/build_index.py`，调用 SQLite FTS5 多字段 BM25：NFKC/casefold 规范化、中文双字切分、英文／数字 token；标题、品牌、类别、型号、属性、规格、bullet 权重分别 3、2、2、2.5、1.5、1.2、0.8。查询为 OR 组合，按 BM25 后 ASIN 升序处理并列，最多 150 商品、每页 20。本次已用完整 23,421 商品构建，SQLite 3.54.1／Python 3.11.18；索引 SHA256 `8401854ff08a2debf5c5ba33c913523e175ac6927d25a2aed23bbcb3748686b7`。[固定搜索实现](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5/environments/ShopSimulator/shop_env/web_agent_site/engine/search.py)。

因此社区版解决的是可复现检索的工程缺口，**不是证明恢复了原版的搜索排序**。

## 3. 社区版改变了什么

比较原版 `475837e…` 与社区 `13d2b636…`，完整文件差分已保存到来源审计目录。

| 维度 | 原版／原测量方案 | 社区 Environment v2.1 / Reward v3 |
| --- | --- | --- |
| 搜索 | Lucene 路径，索引缺失 | SQLite FTS5 多字段 BM25，索引配方完整 |
| 动作 | search、click、Buy Now | 保留底层动作并增加 finish；客户端使用 JSON 工具和当前观察动作守卫 |
| 工具守卫 | 保留原生非法动作处理 | 客户端会拒绝额外参数、历史页面点击等，拒绝发生在环境执行前；不是同一非法动作分布 |
| 终止 | 原方案统一 40 环境动作 | 35 步，连续重复／无新进展也提前终止；停止按证据量评分 |
| 计分价 | 加载时可能 random.uniform，done 使用 product_prices | 基础表取价格列表首值；实际购买按所选 variant 的可解析价格 |
| 预算 | 随机生成 price_upper，不追加任务 | 从任务文字解析显式预算，没有写预算则不制造上限 |
| 类别／价格 | 原连续分中的系数与价格命中项 | 硬门；错类别或超预算 −0.85，无法核验则0且 reward_valid=false |
| 软约束 | 属性／规格命中比例 | 品牌、型号、核心功能、规格按活跃维度加权 |
| 满足要求 | 原奖励可能满分 | 完全满足且目标 ASIN=1；非目标但满足=0.55；部分满足最高0.25 |
| 未购买 | 原协议按环境语义结算 | 主动停止 −0.15／−0.35，步数耗尽 −0.5，循环 −0.65 |
| 可见信息 | 原页面／实际输入 | 新增结构化观察，列表包含类别和关键属性等，客户端另有规范化渲染；会改变信息可见性 |
| query 缺失 | 旧 query_match 恒真，不能作为有效复原 | 数据仍缺，但实际 done 调用独立 evaluate_purchase；遗留旧评分函数还在文件中，并非已恢复原 query |

[社区奖励说明](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5/docs/reward-v3.md)，[固定环境配置](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5/environments/ShopSimulator/shop_env/configs/environment.json)，[工具映射](https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5/src/shopping_grpo/environment/tools.py)。

本轮没有把社区奖励嫁接到原版，也没有删除旧评分分支。社区冒烟明确执行社区原有完整合同。采用社区版后，原方案 R_no_price 公式、随机计分价不可见机制、漏斗解释、40 步设置和固定止损阈值都不能直接照用；必须作为另一环境先修订协议。

## 4. 真实环境回合与恢复结果

使用原试点清单稳定顺序的前 3 个 train 任务，不按奖励挑题。控制器只把公开任务文本作为搜索词，打开实际返回第一件商品、点击公开的首个规格、再购买。**这是脚本工程冒烟，不是模型 rollout，不报告策略成功率。** 使用真实社区引擎、完整商品库、实际搜索及实际 done 路径；没有模拟 reward。

| goal_index | 环境动作数 | 实际购买商品 | Reward v3 | 终止标签 |
| --- | ---: | --- | ---: | --- |
| 14209 | 4 | 625193209912 | 1.0 | gold_purchase |
| 15101 | 4 | 856085211783 | 1.0 | gold_purchase |
| 12375 | 4 | 520623862569 | −0.85 | wrong_purchase |

3／3 完成搜索→商品→规格→购买；第三条低奖励保留，不重跑挑高分。购买前后共享商品对象哈希不变，购买后清理会话再 reset 与初始状态一致。

第二个独立 Python／引擎进程，对三个回合的每个动作前缀重新 reset 并重放，共 **12 个前缀**。逐项比较原始观察、结构化公开观察、可点击列表、URL、完整后台会话（含目标、商品、规格、进展计数）、剩余步数、执行一步后的观察及 status：**12／12 全相同**。三次实际终止购买得分及评分明细与第一次完全相同；两进程完整价格／目标表哈希一致。首次回合和独立恢复分别约 11.86 秒与 11.97 秒，包含 CPU 引擎初始化，不能用作模型采集吞吐。

社区自带环境测试 **43／43 通过**；ForkAudit 前检测试 **7／7 通过**。未加载策略，GPU 小时 **0**。不是原方案的20状态、多规格、多会话或策略截断验收。

详见 [回合清单](evidence/community_smoke_20261115_01/episodes_manifest.json)、[独立恢复](evidence/community_smoke_20261115_01/independent_replay.json)、[版本／依赖／源码登记](evidence/community_smoke_20261115_01/registration.json)、[测试日志](evidence/community_smoke_20261115_01/upstream_tests.log)。

## 5. 仍发现的动作缺陷

原版与当前社区版都保留贪婪正则 `(.+)\[(.+)\]`。对加载规范化后的 **239,925 个规格值**逐项调用真实解析函数，**570 个不等于预期的 click/规格参数**，两版结果完全相同；其中564个值含内部左方括号。例如 `click[hm530[墨绿色定制款]]` 被解析成动作名 `click[hm530`。其余失败记录也完整保存，不能全部归因于内部括号。[官方修复 PR #2](https://github.com/ShopAgent-Team/ShopSimulator/pull/2) 检查时仍未合并；此次未应用补丁。

因此三个成功通路和43单元测试不能保证全部合法规格可达。该问题也影响原方案 r_star 的“合法可达规格”集合，不能忽略或直接按全部笛卡尔组合算最优。

## 6. 环境选择结论与冻结边界

**原方案优先保留原奖励语义，因此目前不把任一候选冻结为正式采集环境。**

- 原版：不准入。query 来源与原生索引仍未解决，空字符串不是修复。
- 社区版：三个工程回合及恢复通过；可作为一个有明确源码、商品、索引和依赖的后续新环境候选。但它不是原版的等价修复，仍有解析缺陷。
- 原版＋社区检索／猜测 query：未运行、不采用；这会产生第三种未登记环境，仍不能解决原奖励数据来源问题。

候选源码／数据／索引字节已固定并保留证据，**正式环境选择与正式采集冻结没有完成**。已向研究者确认后续研究口径：保留原奖励继续等待／核实来源，或采用社区 Reward v3 并先修解析与重新登记测量定义。等待此选择期间不继续模型采集。

## 7. 复算入口与证据位置

`review_sources.py` 仅读取已保存下载文件，重算补充数据一致性、query 缺失和规格解析计数，并导出源码差分；`community_smoke.py` 在真实社区环境生成回合和双进程恢复证据。完整命令见 README。来源机器报告：[source_review.json](evidence/source_review_20261115_01/source_review.json)。旧 preflight_20261115_01 证据保持原样。
