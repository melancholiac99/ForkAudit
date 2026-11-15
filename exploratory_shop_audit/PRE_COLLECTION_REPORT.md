# 2026-11-15 购物探索：前检交付

> 后续已按研究者要求暂停正式采集并完成官方／社区比较及三个真实环境工程回合。最新结论见 [ENVIRONMENT_REVIEW.md](ENVIRONMENT_REVIEW.md)；下面保留首次前检的历史记录，不代表后续未做环境测试。

**状态：BLOCKED，尚未进行试运行或正式测量。** 已取得固定版本的商品／任务数据，阻塞项是原生检索资产缺失。两张 A100 80GB 可见，本轮没有分配模型到 GPU，累计 GPU 小时为 **0**。首次沙箱内 GPU 查询失败，在解除沙箱后宿主机只读查询成功；不应将首次失败解读成服务器没有 GPU。

## 一页结论

| 问题 | 本次可支持的结论 |
| --- | --- |
| 环境能否稳定运行 | 未知。固定源码安装脚本引用的 `search_engine/`、转换脚本、索引脚本和索引未提供；未启动完整环境 |
| 数据是否已经取得 | 是。原始 gzip SHA256 `f51c33217061479f9c95a1068621fcd38e4883ae3d2f6a1627037bea934f2125` |
| 任务数量 | 原始商品／原始任务、原生加载后的商品／目标均为 train 21,962、eval 1,459；每个目标对应独立商品 |
| 价格项 | 15,974／23,421 商品带价格区间，68.2037%；train 任务的分母权重 1/(m+o+1) 平均 16.4613%，不是实际价格奖励贡献 |
| 规格数量 | 原始与加载后最大笛卡尔组合数均为 656；尚未验证点击可达性，也未计算精确 r_star |
| 额外评分诊断 | 原始 23,421 商品均无 query，原生加载器均置空；依照固定源码 query_match 将恒真、r_type 将恒为 1。保留原生行为，不自行补造 query |
| 截断／可见性 | 未测量，不能确认合作者的 84.6%／67.8% 或 price_upper 的实际输入不可见性 |
| CTHA／停手／回放／S-F | 全部 NOT_RUN；没有奖励分布、覆盖、机会、信度或方法收益结果 |
| 三条止损 | 全部证据不足，不作触发／未触发判断，不启动训练 |

## 已完成并保存

- [本轮协议](protocol.md)、[参数](config.json)、[实施决策](DECISIONS.md)。正式档位仍为 null，须真实试点吞吐后选定。
- [原生数据核查](evidence/preflight_20261115_01/data_audit.json)、来源／环境证据（本地 `runs/preflight_20261115_01/provenance.json`）、[完整源码函数哈希](evidence/preflight_20261115_01/native_initialization.json)。数据加载使用固定源码原函数体，未执行奖励、NLP 或检索。
- [任务清单摘要](evidence/preflight_20261115_01/tasks_summary.json)：20 个试点与 200 个正式 train 任务，目标商品组互斥，类别比例抽样，另存后备稳定顺序。eval 未产生模型请求。
- 价格快照（本地 `runs/preflight_20261115_01/product_prices_frozen.json`）、完整目标快照（本地 `runs/preflight_20261115_01/goals_frozen.json`）、[快照哈希](evidence/preflight_20261115_01/snapshot_manifest.json)。两份独立 Python 进程读取一致，[日志](evidence/preflight_20261115_01/snapshot_process_check.json)保留；**不是两引擎恢复验收**。
- [模型清单](evidence/preflight_20261115_01/model_manifest.json)：本地 Qwen2.5-7B-Instruct 各权重和 tokenizer 实际字节已哈希，与保存的上游 LFS／Git blob 元数据一致；revision `a09a35458c702b33eeacc393d103063234e8bc28`。未加载推理框架。
- [依赖](evidence/preflight_20261115_01/dependencies.json)、[成本](evidence/preflight_20261115_01/costs.json)、[进度](evidence/preflight_20261115_01/progress.jsonl)、[前检状态](evidence/preflight_20261115_01/preflight.json)。7 项 CPU 前检测试通过；这些测试不替代研究方案中的真实环境或科学估计测试。

大文件和任务文本位于被忽略的 `runs/`，没有提交模型权重或旧实验数据。既有 E4、E5、CTHA 文件与结果未改动。

## 源码核查与适配要求

固定 ShopSimulator 的 `step` 返回 `(state, status, info)`，旧 E5 适配假定四元组；重置要显式给 `idx`，不能只照搬 `session`。`split="train"` 不是 tag 过滤。包装 `MAX_HISTORY_LENGTH=42` 可能先于 40 动作截止，接入时须统一动作预算。score 用随机 product_prices，不用页面／规格价；get_reward 会改商品 price。服务额外字段必须与策略上下文隔离。

官方主仓库也未列出 `shop_env/search_engine`；公开 Hugging Face 数据仓库列出数据 JSONL，没有原生索引。未用自造 BM25、外部网页搜索或其他检索排序替换原环境。默认商品文件名与随库 gzip 不同可通过显式 file_path 处理，不是原始数据不存在。

## 尚需完成

取得与固定版本匹配的原生检索索引／生成配方，明确索引字段、中文分析器、排序参数、来源及哈希后，才能接通隔离引擎。在此之前停止实际实验，符合协议的环境前检门。

此目录目前是**前检与冻结材料实现，尚不是完整采集／分析流水线**。真实停手评分、20 前缀双进程恢复、并发／评分隔离、输入无泄漏、终止边界、三点吞吐、截断审计都未运行；环境适配、轨迹／分叉／评判采集器、完整离线统计和图表尚待实施。未创建空的正式轨迹或回放文件来冒充完成交付。

后续先通过七项试点门，按实测成本选择最大可行档位，再完成冻结测量。不能由当前前检推断任何方法胜负。
