# 隐状态置信度

登记见 proposals/OPSD_R_T0_Protocol.md 的“隐状态置信度”。使用 math-opsd 环境，固定探针沿用存档 token；MATH-500 使用六模型各自归档的非思考回答文本及参考判分。各模型之间按相同数学题文本和参考答案对齐，归档中的 problem_id 仅在单卡分片内唯一。

先运行 python -m unittest -v test_confidence，再用 python experiment.py manifest pilot B 0 1 生成固定清单，python schedule.py 启动调度。调度器利用先前评测已完成的单卡分片，并在下一批开始前回收进程。每卡一个 HF 进程。逐题结果原子写入 run/latent-confidence，可续跑。100 题测速写入 timing.json，全量统计写入 results.json 和 REPORT.md。测速时的临时方向仅用于测时。
