"""Phase 2.2 (arm C, V baseline): residual-stream export at two positions, one HF forward per point.

Sequence = tok(chat(prompt)) ++ tok(realized response up to the opening tag of the parsed action).
  prompt_end : last token of tok(chat(prompt))  (generation prompt included)  -- primary, = arm-B information
  pre_action : last token of the sequence (after </think> ...,  just before the action tag) -- "commitment state"
Layers = hidden_states[L] for L in {14, 21, 26} (index 0 = embeddings; L = residual stream after block L). float16.
Points whose realized prefix cannot be located get pre_action = prompt_end and pre_action_ok = False.
Outputs hidden/{position}_{layer}.npy + hidden/index.parquet (traj_id, turn, row, pre_action_ok, n_tokens).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import common
import scoring

LAYERS = (14, 21, 26)
POSITIONS = ("prompt_end", "pre_action")


def parse_range(s):
    a, b = s.split("-")
    return set(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_dir", default=os.path.join(common.ROOT, "hidden"))
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL_PATH)
    B = scoring.Builder(tok)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(common.MODEL_PATH, dtype=torch.bfloat16).to(dev).eval()
    n_layers = m.config.num_hidden_layers
    assert max(layers) <= n_layers, (layers, n_layers)

    pts = list(scoring.iter_points(parse_range(args.seeds)))
    if args.limit:
        pts = pts[:args.limit]
    d = m.config.hidden_size
    buf = {(p, L): np.zeros((len(pts), d), dtype=np.float16) for p in POSITIONS for L in layers}
    idx = []
    t0 = time.time()
    for r, pt in enumerate(pts):
        pre = B.enc(B.chat(pt["prompt"]))
        rp = scoring.realized_prefix(pt["response"])
        ok = rp is not None
        ids = pre + (B.enc(rp) if ok else [])
        with torch.no_grad():
            hs = m.model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
        for L in layers:
            buf[("prompt_end", L)][r] = hs[L][0, len(pre) - 1].float().cpu().numpy()
            buf[("pre_action", L)][r] = hs[L][0, len(ids) - 1].float().cpu().numpy()
        idx.append(dict(traj_id=pt["traj_id"], turn=pt["turn"], row=r, pre_action_ok=ok, n_tokens=len(ids),
                        n_prompt_tokens=len(pre)))
        if (r + 1) % 200 == 0:
            print(f"{r + 1}/{len(pts)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    for (p, L), a in buf.items():
        assert np.isfinite(a.astype(np.float32)).all(), (p, L)
        np.save(os.path.join(args.out_dir, f"{p}_{L}.npy"), a)
    pd.DataFrame(idx).to_parquet(os.path.join(args.out_dir, "index.parquet"))
    common.write_json(os.path.join(args.out_dir, "provenance.json"),
                      {"provenance": common.provenance(__file__), "model": common.MODEL_PATH, "layers": layers,
                       "n_points": len(pts), "seconds": time.time() - t0})
    print("wrote", len(pts), "points,", len(buf), "arrays")


if __name__ == "__main__":
    main()
