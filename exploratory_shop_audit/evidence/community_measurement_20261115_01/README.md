# 完成的测量证据

[完整报告](REPORT.md)；[指标](metrics.json)；[数据核验](saved_data_validation.json)；[冻结档位](tier_freeze.json)。

20×2试点、200×4正式轨迹、60分叉点（35可定义）；正式A/B独立续跑1176条，试点吞吐续跑60条；70个独立S/F短生成，解析失败14个均保留。

`ARTIFACT_MANIFEST.json`记录服务器原始明细哈希；`EXPORT_MANIFEST.json`记录导出关系。`frozen_sources/`保存试点和正式采集前锁定的源文件，已与冻结哈希逐项核验。原始大规模输入/回放及权重不入Git。
