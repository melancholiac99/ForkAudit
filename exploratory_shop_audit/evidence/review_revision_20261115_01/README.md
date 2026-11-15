# 复核修订证据

原800条轨迹及已有回放保留。新35×2 S/F输出、离线指标与核验见本目录；完整原始请求、特征清单、5705个任务—商品配置最大值及日志保存在服务器，路径和SHA256见ARTIFACT_MANIFEST.json。

REPORT.md为本次解释准据。metrics.json汇总全部新结果；offline_metrics.json含逐局形态、停手交叉表、条件漏斗、逐候选评分有效率与分层信度；early_probe_metrics.json和sf_metrics.json分别保存探针与新旧S/F对照。原证据目录未改写。

复算入口：`python exploratory_shop_audit/analyze_review.py --revision exploratory_shop_audit/runs/review_revision_20261115_01`。此入口使用服务器保存数据，仅CPU分析，不启动采集、评判或训练。
