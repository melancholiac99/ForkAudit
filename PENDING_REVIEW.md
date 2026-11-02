# 统一待确定版本（2026-11-02）

**状态：待合作者审阅；禁止合并 main；不代表 E5 amendment 3 已冻结。**

本分支整合最近四个工作分支的**文件内容**，不改变已完成的实验结论：
- `server/e4-results-20261101`（基底提交 `f03b5713ce126d426087c4275772c2d5b7f53097`）：完整 E4 验证结果、训练日志、证据归档、报告。
- `server/p8-preflight-20261101`：P8 CPU 审计结果、GPU 评分失败诊断、脚本与冻结采样；**P8 阶段 C 尚未完成**。
- `research/fork-selection-audit-2026-11-01`：P8 修订方案及 preflight 脚本；相应版本也包含于 P8 server 分支。
- `research/e5-amendment3-history-credit-20261102`：E5 修订 3 **草案**，路径 `proposals/E5_amendment3_history_credit_2026-11-02.md`。

另从 `main` 补齐 E5 修订 2 的完整代码和预注册、STORY_v3、P7 方案及 E4 评分器候选数兼容修复，以使待审分支成为**自包含工作树**；并没有将本分支反向并入 main。

## 审核入口
1. `exploratory_E4/E4_REPORT.txt`, `exploratory_E4/e4.json`, `exploratory_E4/PUBLICATION.txt`；注册读法 R-L-HEURISTIC。
2. `exploratory_P8/P8_CPU_REPORT.txt`, `exploratory_P8/failure_reports/`；CPU 已验证，阶段 C 失败，不应解释成科学负结果。
3. `proposals/P8_fork_selection_audit_2026-11-01.md`。
4. `exploratory_E5/PREREG_E5.md` 原修订 2 与 `proposals/E5_amendment3_history_credit_2026-11-02.md` **修订 3 v2 草案**：已纳入检索最多四步、实体级新颖性、最后一步 answer-only、移除 supported_answer 主规则，以及 E5-S(H_prompt) 与 E5-C(C_all/C_pre) 的新 H4 对照。纯数据无关的**待审参考解析器** `proposals/e5_amendment3_history_rule.py` 及**合成自测脚本** `proposals/test_e5_amendment3_history_rule.py` 已附，且同 Git blob 哈希的本地合成测试 8/8 通过；仍待服务器零 pilot 数据核查、试点前覆盖门审定与冻结提交，未经冻结不得称 §14 修订生效。


5. `proposals/E4b_zero_replay_credit_learning_value_2026-11-02.md`：**E4b 新实验方案（PROPOSED / NOT FROZEN）**。研究当前策略的三次回放全零状态是否仍提供有效训练监督；四个训练组 A-zero / A-gap / Bfac-gap / D-zero 加 Base，对齐事实锚定偏好对、训练预算和独立任务评估。来自 E4 事后分析，待合作者审核后才可冻结、执行；不改变 E4 R-L-HEURISTIC。


6. `exploratory_E4/post_hoc/E4_LABEL_COMPOSITION.txt` 与 `exploratory_E4/post_hoc/e4_label_composition.py`：原始 E4 事后标签构成补丁中的**两份文件，按原 patch 恢复**；已核验 Git blob SHA 分别为 `a35260d`、`651da75`，与补丁声明一致。性质仍为 exploratory / post-hoc，不是 E4b 的实验结果，也不修改 E4 原注册读法。

## 决策记录冲突
E4 server 与 main/P8 server 分支从共同祖先后分别追加了**不同内容但同编号的 D-059、D-060、D-061**；不能按文件覆盖或把编号相同的记录当作同一条。
- 工作树根目录 `DECISIONS.md` 保留 E4 server 原文。
- `review_sources/DECISIONS_main_20261102.md` 保存 main 原文。
- `review_sources/DECISIONS_p8_server_20261101.md` 保存 P8 server 原文。
合并 main 前必须把三路新增决策逐条重新编号/引用，审核确认，不能静默覆盖。以上快照为出处证据，不是新的预注册。

## 后续流程
合作者只审这个分支并给出修改意见；所有修改继续追加到本分支。意见通过后，完成源文件散列及结果复核、冲突决策整合，**再**向 main 提交合并请求。原四分支暂保留作不可丢失的结果/日志历史引用和服务器远端名；在确认服务器不再引用且所有文件可追溯之后再考虑删除。
