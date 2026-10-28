# 补齐目标位置线索：最小机制实验

这是冻结后 exploratory 实验。代码和输出都在本目录，原始数据与正式主结果不改。

## 开卡后运行

```bash
cd /root/autodl-tmp/idea

# 先跑 2 个点 × 2 动作 × 2 条件 × 2 次 = 16 条，验证 GPU 路径
bash exploratory_deep_discovery/opportunity_completion/run.sh smoke

# 正式：32 个点 × 2 动作 × 2 条件 × 16 次 = 2048 条
bash exploratory_deep_discovery/opportunity_completion/run.sh full
```

两个命令都在前台运行。需要关终端时可放在 tmux 中运行；脚本不会启动后台守护进程、自动扩量或关机。
中断后重复同一命令，会跳过已经完整保存的续跑。不完整的一条从头重跑。

默认使用 `/root/miniconda3/envs/m0/bin/python`、原 Qwen2.5-7B 模型、温度 0.7、最多 50 步、两轮历史。
通过 `M0_MODEL` 修改模型位置，通过 `OPPORTUNITY_PYTHON` 修改 Python 路径。
`OPPORTUNITY_WORKERS=4`、`OPPORTUNITY_CONCURRENCY=64` 是默认 CPU worker / 同时续跑数。
调整 worker/concurrency 可原地续跑；更换模型、协议代码或重复次数须使用新的 run name。

不使用 GPU 的环境核对和管道检查：

```bash
bash exploratory_deep_discovery/opportunity_completion/run.sh check
bash exploratory_deep_discovery/opportunity_completion/run.sh dry
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/opportunity_completion/test_protocol.py
```

## 实际比较什么

每个点有一个 peek 显示目标所在位置的动作 `revealing`，另一个未显示目标的动作 `other`。
两条分支各跑 `native` 和 `completed` 两种条件：

- native：完全使用原来的真实观察。
- completed：执行原动作后，在真实观察末尾补上同一条真实信息，例如 `Additional location information: mug 1 is at countertop 1.`。

位置线索从已经采集的 peek 文本提取；环境 facts 只用于验证它在两条分支中都为真。
不替 Agent 执行动作，不提供成功率、正确策略或未来结果，不改变合法动作与预算。
只注入一次，后面按原有两轮记忆自然保留/移出，没有额外的持续记忆。
各分支的物理位置、容器开关状态、历史动作仍可能不同，所以不预设差距会完全消失。

## 样本已准备

`design.json` 固定 32 个不同任务，24 个原来检出的 Fork、8 个 Live 未检出 Fork，覆盖六类任务。
详细案例见 [design.md](design.md)。全部目标实例在此前 factual 历史（含合法动作）中未出现。
两动作如何配对不读取动作成功率或 Gold；使用固定随机顺序选点，并在每层平衡任务类型。
这是面向机制的富集样本，不用来重估全体 Agent 的 Fork 比例。

`prepare.py --verify-env` 能重建一份新样本并核对真实环境。已有 design 默认不覆盖；
如确需另选样本，用 `--out 新文件.json`，并把它交给 `run.py --design`。

## 跑完看哪里

正式输出在 `runs/main_32x16/`：

- `summary.md`：直接读的结果表，含全部点和分层结果。
- `summary_figure.png` / `.svg`：补齐前后两条分支的成功率，以及逐点差距变化。
- `summary_points.csv`：逐点四个成功率、两分支各自的增益。
- `summary_paired_outcomes.csv`：每次重复的四格结局。
- `results/*.json`：每条续跑的真实动作、观察、模型输出、种子和首次完整提示。
- `progress.json`、`console.log`：进度与本次启动消耗。

主比较是原 Fork 层的有符号差距变化：

```
native_gap    = q(revealing, native)    - q(other, native)
completed_gap = q(revealing, completed) - q(other, completed)
gap_reduction = native_gap - completed_gap
```

同时必须看 `other_lift` 和 `revealing_lift`。只有原差距确实存在，且另一条分支被补齐后变好，
才支持“补回信息机会”；单纯让 revealing 分支变差不能叫补救。负结果、差距反转都照常输出。
不把差距变化除以接近零的原差距，也不只展示成功例子。
区间按任务 bootstrap；native/completed 使用配对随机数，分析不将四个条件当独立样本。

CPU dry 输出强制分到 `runs/dry_*`，所有报告与图标记为非科学结果。
若任务尚未结束，默认分析拒绝出完整报告；确需查看进度时：

```bash
/root/miniconda3/envs/m0/bin/python exploratory_deep_discovery/opportunity_completion/analyze.py \
  exploratory_deep_discovery/opportunity_completion/runs/main_32x16 --allow-partial
```

此时写入独立的 `partial_summary.*`。最终完整跑完，脚本自动生成正式的 exploratory 汇总。

## 成本与范围

正式固定 2048 条续跑；当前选点最多 72,384 次模型决策（实际取决于提前完成）。额外代价只有线索文本 token。
先用 smoke 的实际吞吐判断需要开卡多久。本实验不训练策略、不运行 WebShop、不追加四因素消融。
输入哈希、设计哈希和运行代码哈希用于防止续跑时混入不同版本结果。

## 当前验证状态（2026-10-28）

32 个点的 64 条真实环境分支均已通过前缀、观察、合法动作、完整 facts 与线索真实性核对，
见 `design.preflight.json`。7 项针对性测试通过；一个真实前缀的四格随机策略续跑、配对种子、
50 步截断、断点恢复和自动表图生成均已验证。原始输入文件哈希未改变。
GPU 的 16 条 `smoke` 与正式 2048 条已全部完成，汇总与图在 `runs/main_32x16/`。
解读见 [实验结果报告](../../reports/opportunity_completion_results_2026-10-28.md)。启动脚本已补齐 conda / CUDA 工具路径，
无需手动激活环境。CPU 随机策略结果不构成科学发现，GPU smoke 仅用于检查运行路径。
