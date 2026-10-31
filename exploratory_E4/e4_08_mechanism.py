"""E4 step 8 (PL-5): does training change the policy's own choices at the M0 information-limited forks?

For each model, 32 drafts (T=0.7) at every M0 Live I- and U-layer Fork point (184 + 202), from the stored M0 prompt.
Selection distribution over the five M0 arms = draft counts + 0.5 (as P4-X X2), G_u = p.q12 - mean(q12) with the
M0 bank 1+2 estimates. Draft seed = scripts/11_replay seed_of(task_seed, turn, 0, 0) + 7 * 10**12 (same for every model).
Usage: python e4_08_mechanism.py --models base A_s0 A_s1 A_s2 Rnd_s0 Rnd_s1 Rnd_s2
Output: data/mechanism.json
"""
import argparse
import json
import os

import numpy as np

import e4_common as E
import common
from post_e2_mechanism import load_parquet

OFFSET = 7 * 10 ** 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--arms", default=os.path.join(E.ROOT, "exploratory_deep_discovery", "exploratory_arms.parquet"))
    args = ap.parse_args()
    e2 = json.load(open(os.path.join(E.ROOT, "exploratory_P4", "e2.json")))
    assert E.file_sha256(args.arms) == e2["inputs"]["exploratory_deep_discovery/exploratory_arms.parquet"]
    s = load_parquet(os.path.join(E.ROOT, "exploratory_P4", "strata.parquet"))
    a = load_parquet(args.arms)
    pts = sorted((s["traj_id"][i], int(s["turn"][i]), s["stratum"][i]) for i in range(len(s["traj_id"]))
                 if s["live"][i] and s["het"][i] and s["stratum"][i] in ("I", "U"))
    assert len(pts) == 386
    arms = {}
    for i in range(len(a["traj_id"])):
        arms.setdefault((a["traj_id"][i], int(a["turn"][i])), {})[int(a["j"][i])] = (a["action"][i], a["q_12"][i])
    prompts, acts, Q, seeds = [], [], [], []
    for t, u, _ in pts:
        tr = json.load(open(os.path.join(E.ROOT, "data", "rollout", f"{t}.json")))
        st = tr["steps"][u - 1]
        prompts.append(open(os.path.join(E.ROOT, st["prompt_path"])).read())
        acts.append([arms[t, u][j][0] for j in range(5)])
        Q.append([arms[t, u][j][1] for j in range(5)])
        seeds.append(((tr["seed"] * 100 + u) * 10 + 0) * 100_000 + OFFSET)
    Q = np.array(Q, float)
    from e4_policy import LoraPolicy
    pol = LoraPolicy()
    out = dict(points=[f"{t}_t{u:02d}" for t, u, _ in pts], layer=[z for _, _, z in pts], models={})
    for k, name in enumerate(args.models, 1):
        pol.set_adapter(None if name == "base" else os.path.join(E.DATA, "adapters", name), k)
        res = pol.sample(prompts, seeds, n=32)
        G = []
        for o, ac, q in zip(res, acts, Q):
            c = np.zeros(5)
            for x in o:
                a_, valid, _ = common.parse(x["text"])
                if valid and a_ in ac:
                    c[ac.index(a_)] += 1
            p = (c + .5) / (c + .5).sum()
            G.append(float(p @ q - q.mean()))
        out["models"][name] = G
        print(name, {L: float(np.mean([g for g, z in zip(G, out["layer"]) if z == L])) for L in ("I", "U")}, flush=True)
    E.write_json(os.path.join(E.DATA, "mechanism.json"), out)
    pol.close()


if __name__ == "__main__":
    main()
