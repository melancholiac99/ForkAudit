"""Phase 0.2 smoke test: one random-policy episode (50-step cap) on task seed 0, then verify prefix re-feed determinism."""
import json
import os
import random
import alfenv
import common

tl = json.load(open(os.path.join(common.ROOT, "data", "task_list.json")))["tasks"]
gf = os.path.join(alfenv.ALFWORLD_DATA, tl[0]["game_file"])
g = alfenv.AlfGame(gf)
obs, adm, won = g.reset()
task = common.extract_task(obs)
print("TASK:", task)
print("PROMPT(step1):\n" + common.build_prompt(task, [], obs, adm))
rng = random.Random(0)
actions, observations, mem = [], [], []
for t in range(common.MAX_STEPS):
    a = rng.choice([x for x in adm if x != "help"])
    mem.append({"text_obs": obs, "action": a})
    obs, adm, won, done = g.step(a)
    actions.append(a); observations.append(obs)
    if done or won:
        break
print(f"steps={len(actions)} won={won} done={done}")
print("PROMPT(last):\n" + common.build_prompt(task, mem, obs, adm))
# determinism: re-feed full prefix in a fresh env, 3 times
divs = []
for _ in range(3):
    g2 = alfenv.AlfGame(gf)
    o, a2, w, d = g2.refeed(actions, observations)
    divs.append(d); g2.close()
print("refeed divergences over 3 fresh envs:", divs)
