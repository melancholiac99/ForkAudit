# E5：WebShop 复现 —— 运行顺序

预注册见 `PREREG_E5.md`。命令在 `exploratory_E5/` 下运行；`WEBSHOP_ROOT` 指向 WebShop 安装目录；数据写入 `exploratory_E5/data/`（已加入 .gitignore）。

```bash
python test_e5.py                                   # CPU 测试

# 门 S0–S3（见 PREREG_E5 §9）：安装 WebShop；把 verl-agent 的 WebShop 模板逐字存为 prompts/webshop_template.txt
# 并核对 ws_common.build_prompt；核对 ws_env.py 中标注 VERIFY 的接口；记录 commit 到 DECISIONS。
python e5_run.py freeze                             # -> tasks.json，单独提交

# 门 S4：dry 管线
python e5_run.py rollout --dry --limit 3 --out_root /tmp/e5dry
for b in 1 2; do python e5_run.py bank --b $b --dry --limit 3 --out_root /tmp/e5dry; done
python e5_run.py peek --limit 3 --out_root /tmp/e5dry

# 正式测量
python e5_run.py rollout
for b in 1 2 3 4 5 6; do python e5_run.py bank --b $b; done
python e5_run.py peek
python e5_run.py labels
python e5_run.py strata
python e5_run.py introspect --model $M7          # 修订 1：置信度与隐状态（GPU）
python e5_run.py signals                            # 修订 1：死活与分叉两条边界
python e5_run.py judge-prepare
for c in S L F Fx; do
  python e5_run.py judge-score --cond $c --model $M7  --name 7b  --revision a09a35458c702b33eeacc393d103063234e8bc28
  python e5_run.py judge-score --cond $c --model $M32 --name 32b --revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
done
python e5_run.py analyze                            # -> e5.json, E5_REPORT.txt
```
