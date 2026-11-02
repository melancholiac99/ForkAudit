"""E4c gate S0 (PREREG_E4c §5). Reads no success rate.
  python s0_checks.py dry /tmp/e4c_dry     # dry-run plumbing: 2 games x 3 reps x {R, X}, step fields present
  python s0_checks.py goals                # which task types' PDDL goals mention examine / inventory / look (reported)
"""
import glob
import json
import os
import re
import sys

import e4c_common as C
from e4c_common import E


def dry(root):
    eps = [json.load(open(f)) for f in glob.glob(os.path.join(root, "rollout", "valid_unseen", "*", "s*_r*.json"))]
    assert len(eps) == 12 and all(len(e["steps"]) > 0 for e in eps), len(eps)
    assert all({"draw_index", "n_draws", "blocked", "kept_blocked"} <= set(s) for e in eps for s in e["steps"])
    x = [s for e in eps if e["condition"] == "X" for s in e["steps"]]
    assert all(not C.INSPECT.match(C.norm(s["action"])) or s["kept_blocked"] for s in x)
    print("S0 plumbing passed:", len(eps), "episodes")


def goals():
    G = json.load(open(E.GAMES_PATH))
    hits, unparsed = {}, []
    for split in ("valid_unseen", "valid_seen"):
        for g in G[split]:
            try:
                prob = json.load(open(os.path.join(G["alfworld_data"], g["game_file"])))["pddl_problem"].lower()
                goal = prob[prob.index("(:goal"):]
            except (KeyError, ValueError, json.JSONDecodeError, OSError):
                unparsed.append(g["seed"])
                continue
            if re.search(r"\b(examine|inventory|look)", goal):
                hits.setdefault(g["task_type"], []).append(g["seed"])
    print("goals mentioning examine/inventory/look, by task type:", {k: len(v) for k, v in hits.items()} or "none",
          "| unparsed games:", len(unparsed))


if __name__ == "__main__":
    {"dry": lambda: dry(sys.argv[2]), "goals": goals}[sys.argv[1]]()
