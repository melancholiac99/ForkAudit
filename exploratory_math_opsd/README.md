# Qwen3-1.7B 数学 OPSD 的四卡入口

运行目录默认为 `/mnt/data/ForkAudit/math-opsd`。模型、数据、独立 Python 3.10 conda 环境、上游源码、训练检查点和评测结果均保存在该目录。[登记方案](../proposals/OPSD_R_T0_Protocol.md)

准备新机器时运行以下脚本。它固定上游提交并应用随附补丁，按官方环境文件安装依赖，逐文件验证权重及数据，再执行同题提示和回答掩码的 CPU 核对。基座与各数据集的修订号在 `prepare_assets.py` 中冻结。当前服务器使用独立的 CUDA 12.8 编译工具配合 PyTorch 2.8 和 DeepSpeed 的 CPU 优化器。

```bash
bash exploratory_math_opsd/prepare_server.sh /mnt/data/ForkAudit/math-opsd
```

在四张 A100 就绪后，从 ForkAudit 仓库根目录运行以下命令。它依次训练 R、T0，先写 AIME24 快报，再按实测耗时完成正式评测。共用训练种子 42 和 30 epoch 学习率日程。回调在第 100 个优化器步触发保存并停止。每个模型保存 @25、@50、@75、@100。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash exploratory_math_opsd/run_four.sh all
```

保留盘上的独立环境使用压缩镜像时，该入口会在新机第一次运行时挂载镜像及可写层，随后使用同一环境路径。可先运行 `bash exploratory_math_opsd/run_four.sh env-check` 做 CPU 导入核对。

只训练时将 `all` 改为 `train`。也可以分别使用 `R`、`T0`、`quick` 或 `evaluate`。再次运行会复用已验证的完整评测单元，训练可从最近一个完整检查点续跑。快报位于 `run/evaluation/quick.json`，全部逐题结果和解码长度保存在各模型与基准的子目录，题级配对区间和决策在 `run/evaluation/final.json`。训练日志每组单独落盘，W&B 采用离线模式。
