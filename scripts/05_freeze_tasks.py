"""Freeze the Branch-B task list (DECISIONS D-010): 50 train games, stratified 9/9/8/8/8/8 over the 6 task types
(types sorted alphabetically), numpy seed 0. seed i <-> task i.

--extend (RULING 2026-10-23 §4/§7.5, CLAUDE.md Phase 1C.1): append 50 more tasks (seeds 50-99) drawn with the same
seed and the same stratification rule from the train games not already in the list. The existing entries are left
byte-identical; the extension's provenance is stored under "extensions".
"""
import json
import os
import sys

import numpy as np
import alfenv
import common

OUT = os.path.join(common.ROOT, "data", "task_list.json")


def draw(games, exclude, first_seed):
    types = sorted({t for _, t in games})
    quota = dict(zip(types, [9, 9, 8, 8, 8, 8]))
    rng = np.random.default_rng(0)
    picked = []
    for t in types:
        pool = [g for g, tt in games if tt == t and g not in exclude]
        idx = rng.choice(len(pool), size=quota[t], replace=False)
        picked += [(pool[i], t) for i in sorted(idx)]
    order = rng.permutation(len(picked))  # interleave types across seeds
    tasks = []
    for j, k in enumerate(order):
        gf, t = picked[k]
        tasks.append({"seed": first_seed + j, "task_id": os.path.relpath(os.path.dirname(gf), alfenv.ALFWORLD_DATA),
                      "task_type": t, "game_file": os.path.relpath(gf, alfenv.ALFWORLD_DATA),
                      "game_sha256": common.file_sha256(gf)})
    return tasks, types


games = alfenv.collect_games("train")
if "--extend" in sys.argv:
    cur = json.load(open(OUT))
    used = {os.path.join(alfenv.ALFWORLD_DATA, t["game_file"]) for t in cur["tasks"]}
    first = max(t["seed"] for t in cur["tasks"]) + 1
    new, types = draw(games, used, first)
    assert not ({os.path.join(alfenv.ALFWORLD_DATA, t["game_file"]) for t in new} & used)
    cur["tasks"] += new
    cur.setdefault("extensions", []).append({"provenance": common.provenance(__file__), "seeds": [first, first + len(new) - 1],
                                             "excluded_existing": len(used), "rule": "same seed (0), same 9/9/8/8/8/8 stratification, pool = train minus existing"})
    common.write_json(OUT, cur)
    tasks = new
else:
    tasks, types = draw(games, set(), 0)
    common.write_json(OUT, {"provenance": common.provenance(__file__), "n_train_games_pool": len(games), "tasks": tasks})
print(len(tasks), {t: sum(x["task_type"] == t for x in tasks) for t in types})
