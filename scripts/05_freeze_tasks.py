"""Freeze the Branch-B task list (DECISIONS D-010): 50 train games, stratified 9/9/8/8/8/8 over the 6 task types
(types sorted alphabetically), numpy seed 0. seed i <-> task i."""
import os
import numpy as np
import alfenv
import common

games = alfenv.collect_games("train")
types = sorted({t for _, t in games})
quota = dict(zip(types, [9, 9, 8, 8, 8, 8]))
rng = np.random.default_rng(0)
picked = []
for t in types:
    pool = [g for g, tt in games if tt == t]
    idx = rng.choice(len(pool), size=quota[t], replace=False)
    picked += [(pool[i], t) for i in sorted(idx)]
order = rng.permutation(len(picked))  # interleave types across seeds
tasks = []
for seed, k in enumerate(order):
    gf, t = picked[k]
    tasks.append({"seed": seed, "task_id": os.path.relpath(os.path.dirname(gf), alfenv.ALFWORLD_DATA),
                  "task_type": t, "game_file": os.path.relpath(gf, alfenv.ALFWORLD_DATA),
                  "game_sha256": common.file_sha256(gf)})
common.write_json(os.path.join(common.ROOT, "data", "task_list.json"),
                  {"provenance": common.provenance(__file__), "n_train_games_pool": len(games), "tasks": tasks})
print(len(tasks), {t: sum(x["task_type"] == t for x in tasks) for t in types})
