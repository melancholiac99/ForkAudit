# exploratory_shop_audit

独立的社区购物环境探索测量，已完成首轮及复核修订；暂停训练，不替换main的HotpotQA/Sokoban E5。

先读[结果报告](REPORT.md)、[复核修订登记](REVIEW_REVISION.md)、[当前社区版协议](protocol.md)、[原始交接协议](ORIGINAL_PROTOCOL.md)、[采集前环境修订](COMMUNITY_AMENDMENT.md)。

固定环境：`YYHDBL/shopping-grpo-longhorizon@13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5`，Reward v3，35步；模型Qwen2.5-7B-Instruct。特殊括号解析在隔离适配层修复，上游评分未改。

完整原始数据位于服务器 `exploratory_shop_audit/runs/community_measurement_20261115_01/`；GitHub保留可审阅汇总、50条人工待复核证据、冻结源代码和核验。权重、商品全库、精确输入与原始续跑不入Git。

仅使用保存数据复算（Python3.11+）：

```bash
python -m pip install -r exploratory_shop_audit/requirements-analysis.txt
python exploratory_shop_audit/analyze.py --run exploratory_shop_audit/runs/community_measurement_20261115_01
python exploratory_shop_audit/analyze_review.py --revision exploratory_shop_audit/runs/review_revision_20261115_01
python -m unittest discover -s exploratory_shop_audit -p "test_*.py" -v
```

复核入口仅用保存数据进行CPU分析，不重新调用模型。修订原始数据在服务器`runs/review_revision_20261115_01/`，可审阅报告与指标在`evidence/review_revision_20261115_01/`。`review_recompute.py --derive`可从冻结环境离线复算有效配置最大值；`review_judge.py`是本次35×2完整重测的独立执行入口，不是复算步骤。父目录与原报告保持不变；其首轮停手训练建议、近整局探针和S/F解释已由本次报告更正。

执行入口：`run_pilot.py`先做20×2门检查；`run_measurement.py`完成3点测速、冻结档位、正式轨迹、分叉与S/F；`complete_judges.py`只补未执行的完整长上下文S/F请求；`derive_environment.py`生成精确r_star原语；`validate_saved.py`用本地tokenizer重建全部真实输入。使用前按`runtime.py`配置环境、索引、模型和两个隔离Python路径。

本轮模型执行源代码逐字归档在证据的`frozen_sources/`，并逐项验证采集前哈希。`config.json`为最新状态，运行目录初始`config.json`不覆盖；正式档位以`tier_freeze.json`为准，实际合并配置见`config_resolved.json`。后续S/F容量偏离见SHOP-009。不自动启动训练或合并main。
