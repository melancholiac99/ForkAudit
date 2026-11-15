# 社区购物环境：测量已完成

2026-11-15 完成20题×2试运行、200题×4正式轨迹、60个分叉状态和全部可用点的S/F对照。累计0.33697 GPU小时；没有策略训练、32B实验或main合并。

[完整报告、区间及图表](evidence/community_measurement_20261115_01/REPORT.md) · [汇总指标](evidence/community_measurement_20261115_01/metrics.json) · [保存数据核验](evidence/community_measurement_20261115_01/saved_data_validation.json)

- 停手基线：覆盖52.0%，继续操作机会74.4%，值得下一轮小规模训练对照；尚未证明训练收益。
- CTHA：无奖励差异组52.0%，其中26.0%为全部满分组；预测探针有信号，失败救援证据仍不足。
- 回放噪声：rho3=0.896，95%区间[0.782,0.949]，触发停止低信度回放方向的预定规则。

环境固定为社区Reward v3、35步与原生提前终止；去预算门诊断不是旧版去价格公式。详见[当前协议](protocol.md)、[采集前修订](COMMUNITY_AMENDMENT.md)及[决策记录](DECISIONS.md)。此前环境调查保留在[采集前报告](PRE_COLLECTION_REPORT.md)。
