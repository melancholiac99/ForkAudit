"""E4c shared definitions (PREREG_E4c.md): non-training controls for the search habit learned in E4 / E4b.

Every rule that decides a condition's prompt, which sampled actions are blocked, how resampling is seeded, how
behaviour is summarised and how the reading is chosen lives here. Pure functions; unit-tested in test_e4c.py (CPU).
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
E4_DIR = os.path.join(ROOT, "exploratory_E4")
sys.path.insert(0, E4_DIR)
import e4_common as E                                   # noqa: E402  (puts scripts/ on sys.path)
import common                                           # noqa: E402

DATA = os.path.join(HERE, "data")
E4_DATA = os.path.join(E4_DIR, "data")
CONDITIONS = ["base", "P", "R", "PR", "X"]               # base re-run, prompt hint, no-repeat, both, strong ban
CONTROLS = ["P", "R", "PR", "X"]
TRAINED_REF = {"A": [f"A_s{s}" for s in (0, 1, 2)], "D": [f"D_s{s}" for s in (0, 1, 2)]}
DELTA = 4.0                                              # non-inferiority margin (pp), as E4b
K_RESAMPLE = 8                                           # extra draws per blocked step
RESAMPLE_OFFSET = 10 ** 13                               # > every E4 sample seed (seed_for < 10**12)
TRAINED_RANGE = dict(inspect_max=0.25, repeat_max=0.04)  # E4/E4b trained arms on valid_unseen: 8-25% / 1-4%

# ----------------------------------------------------------------------------- (a) prompt hint

HINT = ("Hint: avoid examine, look and inventory unless they are needed, and do not repeat your previous action; "
        "prefer going to locations and opening containers that you have not visited yet.")
_ANCHOR = "Your admissible actions of the current situation are: [{admissible_actions}].\n"


def hinted_template(template=None):
    """The original HCAPO template with HINT inserted as its own line right after the admissible-action line."""
    t = template if template is not None else common.TEMPLATE
    assert t.count(_ANCHOR) == 1, "template anchor not found exactly once"
    return t.replace(_ANCHOR, _ANCHOR + HINT + "\n")


def template_for(cond):
    return hinted_template() if cond in ("P", "PR") else common.TEMPLATE


# ----------------------------------------------------------------------------- (b) decoding constraints

INSPECT = re.compile(r"^(examine|look|inventory)\b")


def norm(a):
    return " ".join(a.strip().lower().split())


def is_blocked(cond, action, prev_action):
    """R / PR: the parsed action equals the previous executed action. X: that, or an examine / look / inventory
    action. base / P: never blocked."""
    a = norm(action)
    repeat = prev_action is not None and a == norm(prev_action)
    if cond in ("R", "PR"):
        return repeat
    if cond == "X":
        return repeat or bool(INSPECT.match(a))
    return False


def resample_seed(seed, k):
    """k-th extra draw (k = 1..K_RESAMPLE) for a step whose first draw used `seed`."""
    return seed + k * RESAMPLE_OFFSET


def choose(cond, draws, prev_action, parse=None):
    """draws: response texts in draw order (first = the E4 common-random-number draw). Returns (index, action, valid,
    fmt): the first draw whose parsed action is not blocked; if every draw is blocked, the LAST draw is kept."""
    parse = parse or common.parse
    for i, text in enumerate(draws):
        a, valid, fmt = parse(text)
        if not is_blocked(cond, a, prev_action):
            return i, a, valid, fmt
    a, valid, fmt = parse(draws[-1])
    return len(draws) - 1, a, valid, fmt


# ----------------------------------------------------------------------------- behaviour summaries

def target_seen(episode):
    """Proxy (as in the behaviour analysis): an instance of the task's target word appears in the initial observation or
    in any observation after an action. Unknown target word -> None."""
    w = E.target_word(episode["task"])
    if w is None:
        return None
    rx = E.target_regex(w)
    if rx.search(episode["initial_obs"].lower()):
        return True
    return any(rx.search(s["obs_after"].lower()) for s in episode["steps"])


def behaviour(episodes):
    """Step shares of inspect (examine/look/inventory) and go-to actions, immediate-repeat share, target-seen rate,
    success given seen, success, timeout share among failures (n_turns >= MAX_STEPS)."""
    n = insp = go = rep = 0
    seen = won = won_seen = fails = timeouts = known = 0
    for ep in episodes:
        prev = None
        for s in ep["steps"]:
            a = norm(s["action"])
            n += 1
            insp += bool(INSPECT.match(a))
            go += a.startswith("go to")
            rep += prev is not None and a == prev
            prev = a
        ts = target_seen(ep)
        if ts is not None:
            known += 1
            seen += ts
            won_seen += bool(ts and ep["won"])
        won += bool(ep["won"])
        if not ep["won"]:
            fails += 1
            timeouts += len(ep["steps"]) >= common.MAX_STEPS
    N = len(episodes)
    return dict(episodes=N, steps=n, inspect_share=insp / max(n, 1), goto_share=go / max(n, 1),
                repeat_share=rep / max(n, 1), target_seen=seen / max(known, 1), success_given_seen=won_seen / max(seen, 1),
                success=won / max(N, 1), timeout_share_of_failures=timeouts / max(fails, 1))


def in_trained_range(b):
    return b["inspect_share"] <= TRAINED_RANGE["inspect_max"] and b["repeat_share"] <= TRAINED_RANGE["repeat_max"]


# ----------------------------------------------------------------------------- reading (PREREG_E4c §5)

def reading(vsA, vsD, behaviour_ok):
    """vsA / vsD: {control: (ci_lo, ci_hi)} for control - A and control - D on valid_unseen (pp).
    behaviour_ok: {control: bool} (in_trained_range). First match:
      E4c-SUBSTITUTE      some control's (c - A) CI lower bound > -DELTA
      E4c-HEURISTIC-ONLY  else, some control's (c - D) CI lower bound > -DELTA
      E4c-NOMANIP         else, no control reached the trained behavioural range
      E4c-LEARNED         else, every control's (c - D) CI upper bound < 0
      E4c-MIXED           otherwise"""
    if any(vsA[c][0] > -DELTA for c in vsA):
        return "E4c-SUBSTITUTE"
    if any(vsD[c][0] > -DELTA for c in vsD):
        return "E4c-HEURISTIC-ONLY"
    if not any(behaviour_ok.values()):
        return "E4c-NOMANIP"
    if all(vsD[c][1] < 0 for c in vsD):
        return "E4c-LEARNED"
    return "E4c-MIXED"


def summary_matrix(split, tags, games, data):
    cols = []
    for tag in tags:
        sm = json.load(open(os.path.join(data, "rollout", split, tag, "summary.json")))
        cols.append(np.array([sm["per_game"][str(g["seed"])] for g in games], float))
    return np.concatenate(cols, axis=1)
