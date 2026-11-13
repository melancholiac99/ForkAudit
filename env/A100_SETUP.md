# A100 实验环境

本机使用两张 A100-SXM4 80GB。所有模型、数据、缓存、日志和环境放在 Git 忽略的 `.runtime/` 中。准备检查不执行优化器更新，也不启动正式 P1/P2 或自动关机。

## 本机验收结果（2026-11-13）

基础准备已通过：模型四个 SHA256 匹配、23 项 CTHA 测试通过、v3 合成回归通过，714 个固定任务文件齐全，四个划分均通过真实 reset/step。两卡均通过 BF16 前后向、vLLM 重复生成及 HF/LoRA 检查；合成样本平均 logprob 差为 `9.31e-7`，无优化器更新。

完整报告：`.runtime/preparation_report.json`；验收证据：`.runtime/preflight/20261113T044230Z/`。旧 P0/P1 原始分卷仍待迁入，完整冷启动复验尚未通过，不能据此宣告 CTHA 方法有效。

## 使用环境

```bash
cd /root/ForkAudit
source env/a100.sh
python --version
nvidia-smi
```

Python 位于 `.runtime/conda/bin/python`。`env/requirements-a100.txt` 固定了旧 CTHA P1 记录中的核心依赖版本；验收目录中的 `requirements-lock.txt` 记录本机全部解析版本。

本机另存有来自官方 PyPI 的完整哈希锁定清单 `.runtime/requirements-hashed.txt`。阿里云镜像测速更快，可在需要补装时使用：

```bash
UV_CACHE_DIR="$PWD/.runtime/cache/uv" .runtime/tools/uv pip install \
  --python .runtime/conda/bin/python \
  --index https://mirrors.aliyun.com/pypi/simple \
  --default-index https://pypi.org/simple --index-strategy unsafe-first-match \
  --require-hashes -r .runtime/requirements-hashed.txt
```

部分新版本尚未同步到镜像，因此允许回退官方源；所有版本及包哈希仍由官方锁定清单约束。

重新安装依赖、下载资源：

```bash
python -m pip install -r env/requirements-a100.txt
python scripts/prepare_a100_assets.py --model-direct
```

模型使用 `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`，四个权重文件按 `env/qwen_sha256.txt` 校验。ALFWorld 使用 0.4.2 官方下载脚本所列的三个文本数据压缩包，不下载视觉检测器。资源清单位于 `.runtime/assets_manifest.json`。

本机测速显示 Hugging Face 直连更快，`--model-direct` 仅对模型文件绕过代理；数据下载仍沿用当前网络配置。已安装 aria2 时使用多连接断点续传，否则回退到 curl。

## 准备验收

```bash
bash scripts/check_a100.sh
```

检查依赖、CTHA 单元测试、v3 合成回归、两卡 BF16 前后向、固定任务文件、真实环境 reset/step，以及每卡 vLLM 重复生成、HF 概率一致性和 LoRA 反向传播。结果保存在 `.runtime/preflight/<UTC时间>/`，首次失败即停止；合成概率检查不能替代真实轨迹上的 v3 正式门槛。

依赖检查使用 `uv pip check`：TextWorld 1.7.0 官方 wheel 的 `WHEEL` 文件在 `Tag:` 前有空行，新版 `pip check` 因此误报不支持平台。实际标签与本机匹配；保留原包，不修改第三方元数据，并以导入和真实环境交互检查补充验证。

## 导入旧数据

把旧服务器的两个分卷放到 `.runtime/imports/incoming/`：

- `raw.tar.gz.part000`：83,886,080 字节。
- `raw.tar.gz.part001`：66,050,606 字节。

```bash
python scripts/import_ctha_archive.py
python exploratory_ctha/v3/check_cpu.py
```

导入脚本按已发布 `archive_manifest.json` 校验分卷及每个文件，拒绝覆盖不同内容。默认导入位置是 `.runtime/imports/p0_base/` 和 `.runtime/imports/ctha_s0/`。没有旧轨迹时，只能通过 `check_cpu.py --synthetic-only` 做合成检查，不能获得完整冷启动验收。

## 路径与后续运行

`M0_MODEL`、`ALFWORLD_DATA`、`CTHA_P0_ROOT`、`CTHA_LEGACY_ROUND` 和 `CTHA_RUN_ROOT` 可在 source 前设置。v3 子进程使用当前 Python。路径适配会改变源码校验和，因此后续实验必须使用新的注册记录与输出目录，不复用旧验收清单。

单卡任务用 `CUDA_VISIBLE_DEVICES=0` 或 `1` 隔离。两卡可以分配独立实验或种子；正式实验仍须遵守原有协议、数据划分和数值门槛。
