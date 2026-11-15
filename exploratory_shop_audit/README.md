# Shop audit：前检与冻结材料

本目录执行 2026-11-15 复核修订版购物探索方案，与已有 E5、CTHA 分离。
**正式采集暂停。当前含数据前检、来源复核和社区真实环境的三回合／双进程恢复冒烟；没有正式采集器或完整统计分析器。** 最新结论见 [ENVIRONMENT_REVIEW.md](ENVIRONMENT_REVIEW.md)。原空 query 快照保留但隔离，禁止作为已验收正式环境使用。

GitHub 上可直接查看 [复核证据](evidence/README.md)；大体积原始表和逐步回放留在服务器，报告中的在线链接指向已发布摘要和校验记录。

`protocol.md` 登记本轮科学约束，`config.json` 登记数值参数，`DECISIONS.md` 说明前检实现。
原生搜索资产缺失时前检退出 2，并保存报告；没有用替代搜索、模拟轨迹或奖励继续。

```bash
python3 -m unittest discover -s exploratory_shop_audit -p 'test_*.py' -v

.runtime/conda/bin/python exploratory_shop_audit/preflight.py \
  --source-root .runtime/external/shop-audit-upstream/ShopSimulator-475837e7a51737c44c0a9f79f9459be1e79060fa \
  --source-archive .runtime/external/shop-audit-source/source.tar.gz \
  --data .runtime/external/shop-audit-source/fine_items_eval_train_all.json.gz \
  --model .runtime/models/Qwen2.5-7B-Instruct \
  --model-tree .runtime/models/tree.json \
  --model-revision .runtime/models/qwen_revision.json \
  --out exploratory_shop_audit/runs/preflight_20261115_01
```

输出目录必须不存在，保留旧失败证据；不能重跑覆盖快照。`runs/` 被 Git 忽略，因为包含全部任务文本。
前检生成完整价格／目标快照、train-only 的20试点／200正式任务及后备清单、模型逐文件哈希、依赖、源码哈希、成本与进度。
所有摘要来自保存的原始数据，`data_audit.json` 不含模型测量结论。两进程 JSON 读取一致不能替代两引擎恢复验收。

下一步取得匹配的原生索引／生成配方后，接入适配并依次通过协议七项真实试点门；再冻结档位、采集与分析。
规格点击可达性、价格输入路径、奖励一致性、截断、完整离线分析和试点吞吐都尚未验证。

## 环境复核与真实冒烟

社区版本固定为 `13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5`，源码原样保存在 `.runtime/external/shop-environment-review/`。使用独立 `.runtime/shop-review-venv`（Python 3.11），安装社区 `shop_env/requirements.txt`，不运行其全套训练 setup。先解压验证商品 SHA256，再调用社区 `shop_env/scripts/build_index.py` 构建完整索引。

```bash
python3 exploratory_shop_audit/review_sources.py \
  --review-root .runtime/external/shop-environment-review \
  --original-root .runtime/external/shop-audit-upstream/ShopSimulator-475837e7a51737c44c0a9f79f9459be1e79060fa \
  --community-root .runtime/external/shop-environment-review/shopping-grpo-longhorizon-13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5 \
  --out exploratory_shop_audit/runs/source_review_20261115_01

.runtime/shop-review-venv/bin/python exploratory_shop_audit/community_smoke.py \
  --shop-env .runtime/external/shop-environment-review/shopping-grpo-longhorizon-13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5/environments/ShopSimulator/shop_env \
  --products .runtime/external/shop-environment-review/community_products.json \
  --index .runtime/external/shop-environment-review/products.sqlite3 \
  --tasks exploratory_shop_audit/runs/preflight_20261115_01/tasks.json \
  --out exploratory_shop_audit/runs/community_smoke_20261115_01
```

上述输出目录已存在，复跑须另选目录保留历史。冒烟的三题来自旧清单的 train 身份，**不读取旧价格／目标快照**；实际目标和奖励由社区完整合同生成。结果是三个真实脚本购买回合和12个独立进程前缀一致，不是策略成功率或完整试点验收。
