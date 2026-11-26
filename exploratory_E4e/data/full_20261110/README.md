# E4e 完整回合数据归档

原始文件保存在 ForkAudit 的归档分支 `archive/credit-grok-pre-cleanup-20261126`，固定提交为 `6190255c8acb05294edc08120c69c0d4d17e24e2`。

[浏览原始数据](https://github.com/melancholiac99/ForkAudit/tree/6190255c8acb05294edc08120c69c0d4d17e24e2/exploratory_E4e/data/full_20261110) · [查看本次实验汇总](https://github.com/melancholiac99/ForkAudit/tree/6190255c8acb05294edc08120c69c0d4d17e24e2/exploratory_E4e/results_20261110)

在仓库根目录执行以下命令，可将归档中的完整目录恢复到本地工作区。恢复命令只更新工作区文件。

```sh
git fetch origin archive/credit-grok-pre-cleanup-20261126
git restore --source=6190255c8acb05294edc08120c69c0d4d17e24e2 --worktree -- exploratory_E4e/data/full_20261110/
```
