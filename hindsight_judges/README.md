# Hindsight Judges Defend What Was Done

**独立论文方向，当前仍放在 ForkAudit 专用研究分支。** 原 P4、E4、E4b、main 以及其冻结结论均不修改。

- [EXPERIMENT1_PROTOCOL.md](EXPERIMENT1_PROTOCOL.md)：实验一的研究问题、386 个 Fork、10 个配对条件、主终点、统计与解释边界。
- [exp1.py](exp1.py)：`selftest`、`prepare`、`score`、`analyze` 端到端实验一代码。
- [exploratory/anchoring_reanalysis.py](exploratory/anchoring_reanalysis.py)：研究者提供的原始事后复分析脚本，不属于新实验已验证结果。

## 服务器执行（建议独立 worktree，不打断 E4b）

```bash
cd /path/to/ForkAudit
git fetch origin main research/hindsight-judges research/deep-discovery-2026-10-27
git worktree add ../ForkAudit-hindsight origin/research/hindsight-judges
cd ../ForkAudit-hindsight

mkdir -p /tmp/forkaudit_hindsight
git show origin/research/deep-discovery-2026-10-27:exploratory_deep_discovery/exploratory_arms.parquet \
  > /tmp/forkaudit_hindsight/exploratory_arms.parquet

python hindsight_judges/exp1.py selftest
python hindsight_judges/exp1.py prepare --arms /tmp/forkaudit_hindsight/exploratory_arms.parquet

# 同机器、同评分后端、同模型，包含原 S/F baseline 的重新评分；不得与旧 GPU 分数直接相减
python hindsight_judges/exp1.py score \
  --arms /tmp/forkaudit_hindsight/exploratory_arms.parquet \
  --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct --name 32b

python hindsight_judges/exp1.py analyze \
  --arms /tmp/forkaudit_hindsight/exploratory_arms.parquet --name 32b
```

建议同一批请求再用 7B 打一遍。7B 评委就是原策略的冻结 checkpoint，可以和 32B 对照“自己的动作”与“别的模型的动作”；原 P4 E2 的 7B 用时约为 32B 的四分之一：

```bash
python hindsight_judges/exp1.py score \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --name 7b \
  --revision a09a35458c702b33eeacc393d103063234e8bc28
python hindsight_judges/exp1.py analyze \
  --arms /tmp/forkaudit_hindsight/exploratory_arms.parquet --name 7b
```

32B 峰值显存约 74 GB，不要和 E4b 训练放在同一块卡上。

依赖：P4 原始 `strata.parquet`、`e2_requests.jsonl.gz`、`e2_preparation.json`、`e2_point_audit.json`、`e2.json`、`e2_32b/run.json`；与其 SHA 一致的 deep-discovery arms；numpy + P4 自带 minipq；GPU 阶段需要原 P4 的 torch/transformers 环境和原始 Qwen2.5-32B-Instruct。**不要用其他人群替换缺失文件。** 分数每候选五臂计算，总计 386×10=3,860 条请求，不重跑 Agent 环境。

结果：`hindsight_judges/results/exp1_p4/`，其中含输入/哈希 manifest、pointmeta、请求、`32b/scores.jsonl`、数值自检、分析 JSON 和文本报告。结果目录被 .gitignore 忽略；跑完核对并另行归档。

## 既有探索性复分析

```bash
# E4X 是 E4 非权重归档已解压的根目录，含 data/replay/
python hindsight_judges/exploratory/anchoring_reanalysis.py \
  "$PWD" /tmp/forkaudit_hindsight/exploratory_arms.parquet \
  /path/to/E4_extracted /tmp/forkaudit_hindsight/anchoring_reanalysis.txt
```

用户提交的旧脚本按原样保留：P4 用 q12 分层、独立 q3456 / Gold 评价，属于合理的探索性对照；但它本身没有强制检查全部输入 SHA。E4 部分用同一套每臂仅 3 次续跑来定义“坏动作”和判断优劣，不能当独立价值真值。

**何时新开独立 repo：** 待身份机制、信用评判价值损害、训练影响和去偏方法得到实验证据后，再提取干净的数据和代码独立发布；当前优先避免复制大量旧实验与冻结文件。
