"""E4c analysis (PREREG_E4c §4-5). Per game, success is averaged over all rollouts (controls / base: 3 reps; E4 trained
arms A and D: 3 training seeds x 3 reps); differences are paired by game, 10,000-game bootstrap, 95% percentile CI
(E4's game_bootstrap_diff, unchanged). The reading uses valid_unseen only; valid_seen is reported as replication.
  python e4c_analyze.py [--splits valid_unseen]
Output: e4c.json, E4c_REPORT.txt
"""
import argparse
import glob
import json
import os

import e4c_common as C
from e4c_common import E

CONTRASTS = ([(c, "base") for c in C.CONTROLS] + [(c, "A") for c in C.CONTROLS] + [(c, "D") for c in C.CONTROLS]
             + [("A", "base"), ("D", "base"), ("A", "D")])


def episodes(split, tag, data):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(data, "rollout", split, tag, "s*_r*.json")))]


def complete(split, tags, data):
    return all(os.path.exists(os.path.join(data, "rollout", split, t, "summary.json")) for t in tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["valid_unseen", "valid_seen"])
    args = ap.parse_args()
    G = json.load(open(E.GAMES_PATH))
    res = dict(delta_pp=C.DELTA, hint=C.HINT, k_resample=C.K_RESAMPLE, trained_range=C.TRAINED_RANGE, splits={})
    for split in args.splits:
        games = G[split]
        M = {c: C.summary_matrix(split, [c], games, C.DATA) for c in C.CONDITIONS if complete(split, [c], C.DATA)}
        for ref, tags in C.TRAINED_REF.items():
            if complete(split, tags, C.E4_DATA):
                M[ref] = C.summary_matrix(split, tags, games, C.E4_DATA)
        if complete(split, ["base"], C.E4_DATA):
            M["base_E4"] = C.summary_matrix(split, ["base"], games, C.E4_DATA)
        diff = {f"{a}-{b}": E.game_bootstrap_diff(M[a], M[b]) for a, b in CONTRASTS if a in M and b in M}
        if "base" in M and "base_E4" in M:
            diff["base-base_E4"] = E.game_bootstrap_diff(M["base"], M["base_E4"])
        beh = {c: C.behaviour(episodes(split, c, C.DATA)) for c in C.CONDITIONS if c in M}
        for ref, tags in list(C.TRAINED_REF.items()) + [("base_E4", ["base"])]:
            eps = sum((episodes(split, t, C.E4_DATA) for t in tags), [])
            if eps:
                beh[ref] = C.behaviour(eps)
        redraw = {}
        for c in C.CONTROLS:
            steps = [s for ep in episodes(split, c, C.DATA) for s in ep["steps"]]
            if steps:
                redraw[c] = dict(steps=len(steps), redrawn_share=sum(s["draw_index"] > 0 for s in steps) / len(steps),
                                 fallback_blocked_share=sum(s["kept_blocked"] for s in steps) / len(steps))
        res["splits"][split] = dict(success={k: float(100 * v.mean()) for k, v in M.items()}, diff=diff, behaviour=beh,
                                    redraw=redraw, missing=[c for c in C.CONDITIONS if c not in M])
    u = res["splits"].get("valid_unseen")
    if u and all(f"{c}-{r}" in u["diff"] for c in C.CONTROLS for r in ("A", "D")) and all(c in u["behaviour"] for c in C.CONTROLS):
        vsA = {c: tuple(u["diff"][f"{c}-A"]["ci95"]) for c in C.CONTROLS}
        vsD = {c: tuple(u["diff"][f"{c}-D"]["ci95"]) for c in C.CONTROLS}
        ok = {c: C.in_trained_range(u["behaviour"][c]) for c in C.CONTROLS}
        res["reading"] = C.reading(vsA, vsD, ok)
        res["behaviour_in_trained_range"] = ok
    E.write_json(os.path.join(C.HERE, "e4c.json"), res)
    lines = [f"E4c reading: {res.get('reading')}  (delta = {C.DELTA} pp; valid_unseen)", f"hint: {C.HINT}", ""]
    for split, r in res["splits"].items():
        lines.append(f"== {split}: success (%)" + (f"   [missing: {r['missing']}]" if r["missing"] else ""))
        lines += [f"  {k:8s} {v:6.2f}" for k, v in r["success"].items()]
        lines.append("  paired differences (pp, 95% CI over games)")
        lines += [f"  {k:14s} {v['diff']:+6.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for k, v in r["diff"].items()]
        lines.append("  behaviour: inspect / goto / immediate repeat share of steps; target seen; success | seen; timeouts")
        for k, b in r["behaviour"].items():
            lines.append(f"  {k:8s} {b['inspect_share']:.3f} {b['goto_share']:.3f} {b['repeat_share']:.3f} | seen {b['target_seen']:.3f} "
                         f"succ|seen {b['success_given_seen']:.3f} | timeouts {b['timeout_share_of_failures']:.2f}")
        for c, d in r["redraw"].items():
            lines.append(f"  redraw {c:3s} redrawn {d['redrawn_share']:.3f} of steps; kept-blocked fallback {d['fallback_blocked_share']:.3f}")
    open(os.path.join(C.HERE, "E4c_REPORT.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
