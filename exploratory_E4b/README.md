# E4b：回放零信用状态的训练价值 — 运行指南

**预注册已获研究者批准（D1–D4）**；冻结版本见 [PREREG_E4b.md](PREREG_E4b.md)，冻结 commit SHA 及来源记录见根目录 `DECISIONS.md`。原 E4 结果和读法 R-L-HEURISTIC 不受影响。E4b 仍无 GPU 训练结果。

## 必备环境与原始数据

在服务器的 `m0` Python 环境、仓库根目录执行。必须已有 E4 原始实验数据 `exploratory_E4/data/`，至少包含 `points/`、`judge/32b_F/scores.jsonl`、`replay/`、`points_manifest.json` 和 `replay_manifest.json`，并可读取原 E4 的模型权重。GitHub 已归档 E4 结果，但它们也包含打包压缩文件；**仓库更新不等于服务器已经自动解压出原始数据**。

先同步 `main`（如服务器本地有其他未提交修改，不要强制覆盖），确认 `exploratory_E4/data` 存在后，运行：

```bash
cd exploratory_E4b
python test_e4b.py
E4B_E4DATA=../exploratory_E4/data python test_e4b.py
mkdir -p data
nohup bash run_e4b.sh > data/run_e4b.log 2>&1 &
```

`run_e4b.sh` 首先重新生成每臂 960 对事实锚定偏好对，并**逐一核对** `e4b_gate.json` 中冻结的训练文件 SHA256，不一致立即退出。随后训练 Azero/Agap/Dzero/Dgap 三种子，使用 valid_unseen 输出主要 2×2 读法，再训练 Bfacgap，最后对 valid_seen 进行评估。中断后可从原脚本继续运行，切勿根据中间效果更改设置。

预计单卡 H800 约 **7.7 GPU 小时**，是根据 E4 已测吞吐推算，不是已执行 E4b 的实际时间。日志在 `data/logs/`，最终结果在 `e4b.json`、`E4b_REPORT.txt`。

## 文件

- `PREREG_E4b.md`：冻结人群、配额、偏好结构、训练配置、主/次终点、非劣 δ = 4 pp 及读法。
- `e4b_common.py`：分类、偏好对路由、配平抽样、预定结果读法。
- `e4b_pairs.py`：构造五臂 960 对样本及哈希清单。
- `e4b_run.py`：复用原 E4 冻结训练/评估代码，仅重定向数据目录。
- `e4b_analyze.py`：任务级配对统计、10000 次 bootstrap 与分阶段汇报。
- `test_e4b.py`、`e4b_gate.json`：CPU 自测与经过 E4 原结果数据核对的样本门。
- `m0_zero_check.py` / `m0_zero_check.json`：仅辅助解释的既有 M0 深度回放分析，不进入 E4b 主读法。

**边界**：Z3 是当前策略每臂 3 次回放的观测全零，不等于真实 Q = 0。Z3/V3 的轨迹结局与回合位置不可配平；主比较仅解释数据筛选策略的价值，不宣称零信用本身的因果效应。
