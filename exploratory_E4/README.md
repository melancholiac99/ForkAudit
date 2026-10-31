# E4：闭环学习实验 —— 运行顺序

预注册见 `PREREG_E4.md`。所有命令都在 `exploratory_E4/` 下运行，Python 用服务器的 m0 环境。大文件写入 `exploratory_E4/data/`（已加入 .gitignore），提交时只提交汇总、清单与 SHA256。

```bash
# 0. 冒烟检查 S1–S4（CPU 或少量 GPU，不读任何结局）
python test_e4.py
python e4_01_rollout.py --split train --tag dry --full --dry --limit 4 --out /tmp/e4s2
python e4_02_points.py --dry --limit 2 --rollouts /tmp/e4s2 --out /tmp/e4s3
python e4_04_replay.py --dry --limit 2 --points /tmp/e4s3 --rollouts /tmp/e4s2 --out /tmp/e4s4

# 1. 冻结游戏列表，单独提交 games.json
python e4_00_freeze.py

# 2. 训练数据
python e4_01_rollout.py --split train --tag base --full --reps 1
python e4_02_points.py
python e4_03_judge.py prepare
python e4_03_judge.py score --cond F --model $M32 --name 32b --revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd   # 含 S5 数值门
python e4_03_judge.py score --cond L --model $M32 --name 32b --revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
python e4_04_replay.py
python e4_05_pairs.py

# 3. 冒烟 S6–S7，然后删除 smoke 适配器
python e4_06_train.py --arm Aprime --seed 0 --lr 2e-5 --smoke
python e4_07_eval.py --split tune --adapters Aprime_s0_smoke --no_base --reps 1
rm -r data/adapters/Aprime_s0_smoke data/rollout/tune/Aprime_s0_smoke

# 4. 学习率选择（只用 tune），结果写入 DECISIONS
for lr in 5e-06 2e-05 5e-05; do python e4_06_train.py --arm Aprime --seed 0 --lr $lr --out data/adapters/Aprime_s0_lr$lr; done
python e4_07_eval.py --split tune --no_base --adapters Aprime_s0_lr5e-06 Aprime_s0_lr2e-05 Aprime_s0_lr5e-05
#   选择规则：e4_common.select_lr（成功率最高，并列取较小值）

# 5. 全部臂 x 3 个种子
for arm in A Aprime H Rnd D Bfull Bmatch Cmatch Esft; do for s in 0 1 2; do
  python e4_06_train.py --arm $arm --seed $s --lr $LR; done; done

# 6. 评估与分析
python e4_07_eval.py --split valid_unseen --all
python e4_07_eval.py --split valid_seen --all
python e4_08_mechanism.py --models base A_s0 A_s1 A_s2 Rnd_s0 Rnd_s1 Rnd_s2
python e4_09_analyze.py          # -> e4.json, E4_REPORT.txt
```

`$M32` 为 32B 本地路径（与 P4 E2 相同 revision）；`M0_MODEL` 环境变量为 7B 路径（默认与 M0 相同）。
