"""FINAL RULING 2026-10-26 §3.2 (W1-model inputs): policy-side quantities at the five peek states of each point.

Peek state of (point t, action a) = the agent's next input if it had taken a: memory = rollout steps[:t] + (obs_t, a),
current observation = peek obs, admissible = peek admissible set (data/peek_alts, D-029). The prompt is built with the
same template code as the rollout (common.build_prompt); for the factual peek it must equal the stored prompt of
turn t+1 (checked, counted).
  --part hidden : HF bf16 forward (same pipeline as 23_hidden_export, position prompt_end), layers 14/21/26
                  -> hidden_peek/prompt_end_{L}.npy + hidden_peek/index.parquet          (W1-V: V probe applied here)
  --part score  : HF shared-prefix scorer (M0_HF_DTYPE=fp32, D-041) over the peek admissible set, bare context
                  -> data/scores_peek_fp32.parquet                                       (W1-P: entropy_q here)
Points: D (bank-1 cf_defined) only; factual + 4 alternatives = 5 states per point.
"""
import argparse
import importlib
import json
import os
import time

import numpy as np
import pandas as pd

import common
import scoring

LAYERS = (14, 21, 26)


def parse_range(s):
    a, b = s.split("-")
    return set(range(int(a), int(b) + 1))


def iter_states(seeds):
    for p in sorted(os.listdir(os.path.join(common.ROOT, "data", "rollout"))):
        tr = json.load(open(os.path.join(common.ROOT, "data", "rollout", p)))
        if tr["seed"] not in seeds:
            continue
        for i, stp in enumerate(tr["steps"]):
            b1 = json.load(open(os.path.join(common.ROOT, "data", "replay", tr["traj_id"], f"t{stp['turn']:02d}.json")))
            if not b1["cf_defined"]:
                continue
            pk = json.load(open(os.path.join(common.ROOT, "data", "peek_alts", tr["traj_id"], f"t{stp['turn']:02d}.json")))
            mem = [{"text_obs": s["obs_before"], "action": s["action"]} for s in tr["steps"][:i]]
            nxt = None
            if i + 1 < len(tr["steps"]):
                nxt = open(os.path.join(common.ROOT, tr["steps"][i + 1]["prompt_path"])).read()
            for q in pk["peeks"]:
                prompt = common.build_prompt(tr["task"], mem + [{"text_obs": stp["obs_before"], "action": q["action"]}],
                                             q["obs"], q["admissible_after"])
                yield {"traj_id": tr["traj_id"], "turn": stp["turn"], "role": q["role"],
                       "alt_index": -1 if q["alt_index"] is None else int(q["alt_index"]), "action": q["action"],
                       "prompt": prompt, "admissible": q["admissible_after"],
                       "prompt_match_next": None if (q["role"] != "factual" or nxt is None) else prompt == nxt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["hidden", "score"], required=True)
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--limit", type=int, default=None, help="first N states (tests)")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--hidden_dir", default=os.path.join(common.ROOT, "hidden_peek"))
    ap.add_argument("--score_out", default=os.path.join(common.ROOT, "data", "scores_peek_fp32.parquet"))
    ap.add_argument("--chunk_states", type=int, default=100)
    ap.add_argument("--cand_batch", type=int, default=64, help="candidates per replicated KV cache (memory)")
    args = ap.parse_args()
    st = list(iter_states(parse_range(args.seeds)))
    if args.limit:
        st = st[:args.limit]
    chk = [s["prompt_match_next"] for s in st if s["prompt_match_next"] is not None]
    print(f"{len(st)} peek states; factual-peek prompt == stored next prompt: {sum(chk)}/{len(chk)}", flush=True)
    prov = common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__)),
                                          "model": common.MODEL_PATH, "part": args.part,
                                          "factual_prompt_match": [int(sum(chk)), len(chk)]}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL_PATH)
    B = scoring.Builder(tok)
    t0 = time.time()
    if args.part == "hidden":
        import torch
        from transformers import AutoModelForCausalLM
        layers = [int(x) for x in args.layers.split(",")]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = AutoModelForCausalLM.from_pretrained(common.MODEL_PATH, dtype=torch.bfloat16).to(dev).eval()
        assert max(layers) <= m.config.num_hidden_layers
        buf = {L: np.zeros((len(st), m.config.hidden_size), dtype=np.float16) for L in layers}
        idx = []
        for r, s in enumerate(st):
            ids = B.enc(B.chat(s["prompt"]))
            with torch.no_grad():
                hs = m.model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
            for L in layers:
                buf[L][r] = hs[L][0, len(ids) - 1].float().cpu().numpy()
            idx.append({k: s[k] for k in ("traj_id", "turn", "role", "alt_index", "action", "prompt_match_next")}
                       | {"row": r, "n_tokens": len(ids)})
            if (r + 1) % 500 == 0:
                print(f"{r + 1}/{len(st)} {time.time() - t0:.0f}s", flush=True)
        os.makedirs(args.hidden_dir, exist_ok=True)
        for L, a in buf.items():
            assert np.isfinite(a.astype(np.float32)).all(), L
            np.save(os.path.join(args.hidden_dir, f"prompt_end_{L}.npy"), a)
        pd.DataFrame(idx).to_parquet(os.path.join(args.hidden_dir, "index.parquet"))
        common.write_json(os.path.join(args.hidden_dir, "provenance.json"),
                          {"provenance": prov, "layers": layers, "n_states": len(st), "seconds": time.time() - t0})
    else:
        sc = importlib.import_module("20_score_candidates")
        scorer = sc.HfPrefixScorer(common.MODEL_PATH)
        rows = []
        for c0 in range(0, len(st), args.chunk_states):
            chunk = st[c0:c0 + args.chunk_states]
            jobs = []
            for s in chunk:
                ctx = B.context("bare", s["prompt"])
                for a in s["admissible"]:
                    if a != "help":
                        jobs.append((s, a) + B.build(ctx, a))
            lps = scorer.score([j[2:] for j in jobs], cand_batch=args.cand_batch)
            for (s, a, *_), lp in zip(jobs, lps):
                rows.append({k: s[k] for k in ("traj_id", "turn", "role", "alt_index", "action")}
                            | {"cand": a, "logp_sum": float(lp.sum()), "logp_mean": float(lp.mean()), "n_tokens": len(lp)})
            print(f"{min(c0 + args.chunk_states, len(st))}/{len(st)} states, {len(jobs)} seqs, {time.time() - t0:.0f}s",
                  flush=True)
        df = pd.DataFrame(rows)
        assert np.isfinite(df.logp_sum).all()
        os.makedirs(os.path.dirname(args.score_out), exist_ok=True)
        df.to_parquet(args.score_out)
        common.write_json(args.score_out.replace(".parquet", ".provenance.json"),
                          {"provenance": prov, "dtype": os.environ.get("M0_HF_DTYPE", "bf16"), "n_states": len(st),
                           "n_rows": len(df), "seconds": time.time() - t0})
    print(f"done {args.part}: {len(st)} states, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
