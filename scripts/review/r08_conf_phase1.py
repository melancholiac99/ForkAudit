"""Phase 2 stop/go (CLAUDE.md Phase 2): teacher-forced chosen conf vs the Phase-1 sampling-time log-probs.

The comparable quantity is `conf_realized` (chosen command tokens teacher-forced after the realized <think>...): it
conditions on the same text the sampler had produced. The bare arm-A conf conditions on no reasoning and is not
comparable by construction (reported for completeness). Phase-1 value = mean vLLM sampling log-prob of the generated
tokens whose character span overlaps the parsed action command (xml-format responses only, so the tag boundary is
clean). Reports Pearson / Spearman and median |difference|. No label is read. Output eval/phase2_conf_phase1.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL_PATH)
    F = pd.read_parquet(os.path.join(common.ROOT, "data", "features_B.parquet")).set_index(["traj_id", "turn"])
    rows = []
    for p in sorted(os.listdir(os.path.join(common.ROOT, "data", "rollout"))):
        tr = json.load(open(os.path.join(common.ROOT, "data", "rollout", p)))
        for s in tr["steps"]:
            if s["fmt"] != "xml" or not s["response_logprobs"]:
                continue
            text = s["response"]
            k = text.rfind("</think>")
            tail0 = k + len("</think>") if k != -1 else 0
            m = list(common._TAGGED.finditer(text[tail0:]))
            if not m:
                continue
            g = m[-1]
            a0 = tail0 + g.start(1) + (len(g.group(1)) - len(g.group(1).lstrip()))
            a1 = tail0 + g.end(1) - (len(g.group(1)) - len(g.group(1).rstrip()))
            ids, lps = s["response_token_ids"], s["response_logprobs"]
            pos, sel = 0, []
            for j, t in enumerate(ids):
                piece = tok.decode(ids[:j + 1])
                end = len(piece)
                if end > a0 and pos < a1:
                    sel.append(j)
                pos = end
                if pos >= a1:
                    break
            if not sel:
                continue
            lp = [lps[j] if not isinstance(lps[j], dict) else lps[j].get(str(ids[j]), np.nan) for j in sel]
            f = F.loc[(tr["traj_id"], s["turn"])]
            rows.append({"traj_id": tr["traj_id"], "turn": s["turn"], "phase1_mean": float(np.mean(lp)),
                         "conf_realized": f.conf_realized, "conf_bare": f.chosen_conf})
    D = pd.DataFrame(rows).dropna()
    out = {"n": int(len(D))}
    for c in ("conf_realized", "conf_bare"):
        out[c] = {"pearson": float(pearsonr(D.phase1_mean, D[c])[0]), "spearman": float(spearmanr(D.phase1_mean, D[c])[0]),
                  "median_abs_diff": float((D.phase1_mean - D[c]).abs().median()),
                  "mean_diff": float((D[c] - D.phase1_mean).mean())}
    out["provenance"] = common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}
    common.write_json(os.path.join(common.ROOT, "eval", "phase2_conf_phase1.json"), out)
    print({k: v for k, v in out.items() if k != "provenance"})


if __name__ == "__main__":
    main()
