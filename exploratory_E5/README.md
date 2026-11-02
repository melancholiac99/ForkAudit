# E5：第二环境 —— 运行顺序（修订 2）

预注册见 `PREREG_E5.md`（第 13 节为修订 2）。三个环境：

| `--env` | 名称 | 预测 |
|---|---|---|
| `search` | E5-S：HotpotQA 多跳问题，开放检索（wiki-18 + e5） | 信息受限分叉上一步反馈有效（W-0 到 W-6） |
| `search_ctx` | E5-C：同一批问题，十段上下文直接给出 | 反馈效应缺失（A-1、A-2，操纵检查 C-0） |
| `sokoban` | E5-K：Sokoban 文本版 | 反馈效应缺失（A-1、A-2，操纵检查 K-0） |
| `webshop` | 原 E5（第 1–12 节） | 暂缓；只在 E5-S 试点失败时运行 |

命令在 `exploratory_E5/` 下运行；数据写入 `exploratory_E5/data/{env}[_pilot[_level]]/`（已加入 .gitignore）；检索缓存为 `data/retrieval_cache.sqlite`（E5-S 与 E5-C 共用，`SEARCH_CACHE` 可改路径）。

```bash
python test_e5.py                                   # CPU 测试（15 项）

# ---- 准备（门 S0′，PREREG §13.8）
# 检索服务：verl-agent examples/search/retriever/retrieval_launch.sh（e5-base-v2、e5_Flat.index、wiki-18.jsonl，--faiss_gpu，端口 8000）
export RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export HOTPOT_DEV=/path/to/hotpot_dev_distractor_v1.json WIKI18=/path/to/wiki-18.jsonl
# 核对服务器上 verl-agent 的 commit（预期 20bd331）；不同则逐字节核对 prompt，写 DECISIONS

# ---- 冻结任务（各自单独提交）
python e5_run.py freeze --env search                # -> tasks_search.json（1,000 题）+ tasks_search_pilot.json（50 题）
python e5_run.py freeze --env sokoban --pilot --level L1    # 试点房间；L2、L3 按阶梯需要时再冻结

# ---- 试点门（PREREG §13.4；不读 Fork 标签与评判输出）
for env in search search_ctx; do
  python e5_run.py rollout --env $env --pilot
  for b in 1 2; do python e5_run.py bank --env $env --pilot --b $b; done
  for s in peek labels strata pilot; do python e5_run.py $s --env $env --pilot; done   # -> data/{env}_pilot/pilot.json
done
python e5_run.py rollout --env sokoban --pilot --level L1
for b in 1 2; do python e5_run.py bank --env sokoban --pilot --level L1 --b $b; done
for s in peek labels strata pilot; do python e5_run.py $s --env sokoban --pilot --level L1; done
# 不通过：按 §13.4 / §13.5 的规则处理（L2、L3 依次；E5-S 只缺数量则扩大正式集），先写 DECISIONS
python e5_run.py freeze --env sokoban --level L1    # 用试点选定的级别冻结正式房间 -> tasks_sokoban.json

# ---- 正式测量（每个通过试点的环境）
ENV=search   # 依次 search、search_ctx、sokoban
python e5_run.py rollout --env $ENV
for b in 1 2 3 4 5 6; do python e5_run.py bank --env $ENV --b $b; done
for s in peek labels strata; do python e5_run.py $s --env $ENV; done
python e5_run.py introspect --env $ENV --model $M7
python e5_run.py signals --env $ENV
python e5_run.py judge-prepare --env $ENV
for c in S L F Fx; do
  python e5_run.py judge-score --env $ENV --cond $c --model $M7  --name 7b  --revision a09a35458c702b33eeacc393d103063234e8bc28
  python e5_run.py judge-score --env $ENV --cond $c --model $M32 --name 32b --revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
done
python e5_run.py analyze --env $ENV                 # -> e5_{env}.json, E5_{env}_REPORT.txt

# ---- 跨环境对比（A-3、K-3）
python e5_run.py cross                              # -> e5_cross.json
```

`--dry` 用随机策略跑通管线（检索服务仍需可用，或在测试中用假检索器）。
