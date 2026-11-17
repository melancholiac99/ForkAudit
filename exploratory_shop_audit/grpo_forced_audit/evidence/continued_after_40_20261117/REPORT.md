# 第40步检查点恢复说明

研究者明确授权“从第40步恢复并继续训练（推荐）”。旧更新41/42已经执行但完整模型和优化器未保存。隔离重建在更新41之前的tokens、mask、优势、actor/behavior概率逐位相同，更新后权重哈希不同，因此不称精确恢复。

原41/42训练指标、权重哈希链、alignment诊断和训练器文本输出原样归档于recovery_0042/pre_continuation_archive；manifest.json记录原文件校验和和路径。当前训练日志从原1–40开始，后续41/42是授权重新计算，不是在原42状态上重复更新。原始环境轨迹和缓存保持不变，不挑选奖励或重新生成已完成样本。

41/42保留原padding重复项和四卡样本顺序。41–43使用原缓存；42/43的生成策略属于故障前谱系，实际行为概率保留，不能声称严格on-policy或逐位延续。新生成从缺失的后续批次开始。

TensorBoard的新日志位于tensorboard/continued_after_40，原曲线保留。正常流程仍是每10步验证、平台期检查、最多100步、最终评测和E1。执行记录与用户授权见recovery_0042/continuation_authorization.json。恢复和失败尝试的GPU占用全部计入主cost_ledger。
