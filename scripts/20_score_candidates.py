"""Phase 2.2: teacher-forced scoring of the candidate set (arm B), chosen-action variants (arm A, H1, sensitivity).

For every rollout step (all points; features use t-1 and t+1..t+5 neighbours):
  * every candidate a in 𝒜_t = admissible ∪ {chosen} \\ {help}:  variants bare + notask  -> scores.parquet
  * chosen only: emptythink, realized, hindsight                     -> scores_chosen.parquet
Construction: scoring.Builder (command tokens only, delimiters masked; raw log-probs).

Backends: --backend vllm (primary, CLAUDE.md §3) | hf (alignment check / CPU plumbing test).
Usage: python 20_score_candidates.py --backend vllm [--seeds 0-99] [--limit N] [--out_dir data]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import common
import scoring

MODEL_REV = {"Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28"}  # D-012


def model_hash(path):
    b = os.path.basename(path.rstrip("/"))
    return MODEL_REV.get(b, f"untracked:{b}")


def parse_range(s):
    a, b = s.split("-")
    return set(range(int(a), int(b) + 1))


def jobs_for(pt, B):
    """-> list of (key, ids, start, end). key = (kind, action, variant)."""
    out = []
    cands = scoring.candidate_set(pt["admissible"], pt["chosen"])
    ctx_b = B.context("bare", pt["prompt"])
    ctx_n = B.context("notask", pt["prompt"], task=pt["task"])
    for a in cands:
        out.append((("cand", a, "bare"),) + B.build(ctx_b, a))
        out.append((("cand", a, "notask"),) + B.build(ctx_n, a))
    for v in ("emptythink", "realized", "hindsight"):
        ctx = B.context(v, pt["prompt"], response=pt["response"], won=pt["won"], final_obs=pt["final_obs"])
        if ctx is not None:
            out.append((("chosen", pt["chosen"], v),) + B.build(ctx, pt["chosen"]))
    return out


class VllmScorer:
    def __init__(self, path):
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        self.TP = TokensPrompt
        self.sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
        self.llm = LLM(model=path, dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=6144,
                       enable_prefix_caching=True)

    def score(self, seqs):
        """seqs: list of (ids, start, end) -> list of np.array command-token logprobs."""
        outs = self.llm.generate([self.TP(prompt_token_ids=ids) for ids, _, _ in seqs], self.sp, use_tqdm=False)
        res = []
        for (ids, s, e), o in zip(seqs, outs):
            pl = o.prompt_logprobs
            res.append(np.array([pl[i][ids[i]].logprob for i in range(s, e)], dtype=np.float64))
        return res

    def close(self):
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception as ex:  # noqa: BLE001
            print("engine shutdown warning:", ex)


class MockScorer:
    """Plumbing test only: deterministic pseudo log-probs of the right length; never used for results."""

    def score(self, seqs):
        out = []
        for ids, s, e in seqs:
            rng = np.random.default_rng(hash(tuple(ids[-12:])) % (2 ** 32))
            out.append(-rng.exponential(1.0, e - s))
        return out

    def close(self):
        pass


class HfScorer:
    def __init__(self, path, device=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or torch.bfloat16
        self.m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(self.dev).eval()

    def score(self, seqs, batch=16):
        """lm_head is applied only at the positions predicting command tokens (memory: no [B, L, V] tensor)."""
        torch = self.torch
        res = []
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            L = max(len(ids) for ids, _, _ in chunk)
            x = torch.zeros((len(chunk), L), dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, (ids, _, _) in enumerate(chunk):
                x[j, :len(ids)] = torch.tensor(ids)
                att[j, :len(ids)] = 1
            with torch.no_grad():
                h = self.m.model(input_ids=x.to(self.dev), attention_mask=att.to(self.dev)).last_hidden_state
                for j, (ids, s, e) in enumerate(chunk):
                    pos = torch.arange(s - 1, e - 1, device=h.device)
                    lp = torch.log_softmax(self.m.lm_head(h[j, pos]).float(), -1)
                    tgt = torch.tensor(ids[s:e], device=h.device)
                    res.append(lp[torch.arange(len(tgt)), tgt].cpu().numpy().astype(np.float64))
        return res

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf", "mock"], default="vllm")
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N points (tests)")
    ap.add_argument("--only", default=None, help="comma list traj_id:turn (alignment check)")
    ap.add_argument("--out_dir", default=os.path.join(common.ROOT, "data"))
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--chunk_points", type=int, default=200)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL_PATH)
    assert args.backend != "mock" or args.out_dir != os.path.join(common.ROOT, "data"), "mock output must not go to data/"
    B = scoring.Builder(tok)
    pts = list(scoring.iter_points(parse_range(args.seeds)))
    if args.only:
        want = {(x.split(":")[0], int(x.split(":")[1])) for x in args.only.split(",")}
        pts = [p for p in pts if (p["traj_id"], p["turn"]) in want]
    if args.limit:
        pts = pts[:args.limit]
    scorer = {"vllm": VllmScorer, "hf": HfScorer, "mock": lambda p: MockScorer()}[args.backend](common.MODEL_PATH)
    mh = model_hash(common.MODEL_PATH)
    rows_c, rows_x = [], []
    t0 = time.time()
    for c0 in range(0, len(pts), args.chunk_points):
        chunk = pts[c0:c0 + args.chunk_points]
        jobs = [(pt, j) for pt in chunk for j in jobs_for(pt, B)]
        lps = scorer.score([j[1:] for _, j in jobs])
        acc = {}
        for (pt, (key, ids, s, e)), lp in zip(jobs, lps):
            acc[(pt["traj_id"], pt["turn"]) + key] = lp
        for pt in chunk:
            k0 = (pt["traj_id"], pt["turn"])
            adm = set(pt["admissible"])
            for a in scoring.candidate_set(pt["admissible"], pt["chosen"]):
                b, n = acc[k0 + ("cand", a, "bare")], acc[k0 + ("cand", a, "notask")]
                rows_c.append(dict(traj_id=pt["traj_id"], turn=pt["turn"], action=a, is_chosen=a == pt["chosen"],
                                   is_admissible=a in adm, logp_sum=b.sum(), logp_mean=b.mean(), n_tokens=len(b),
                                   logp_sum_notask=n.sum(), logp_mean_notask=n.mean(),
                                   scorer=args.backend, model_hash=mh))
            for v in ("emptythink", "realized", "hindsight"):
                lp = acc.get(k0 + ("chosen", pt["chosen"], v))
                rows_x.append(dict(traj_id=pt["traj_id"], turn=pt["turn"], variant=v, available=lp is not None,
                                   logp_sum=np.nan if lp is None else lp.sum(),
                                   logp_mean=np.nan if lp is None else lp.mean(),
                                   n_tokens=0 if lp is None else len(lp), scorer=args.backend, model_hash=mh))
        print(f"{min(c0 + args.chunk_points, len(pts))}/{len(pts)} points, {len(jobs)} seqs in chunk, "
              f"{time.time() - t0:.0f}s", flush=True)
    scorer.close()
    os.makedirs(args.out_dir, exist_ok=True)
    pc = os.path.join(args.out_dir, f"scores{args.tag}.parquet")
    px = os.path.join(args.out_dir, f"scores_chosen{args.tag}.parquet")
    pd.DataFrame(rows_c).to_parquet(pc)
    pd.DataFrame(rows_x).to_parquet(px)
    common.write_json(os.path.join(args.out_dir, f"scores{args.tag}.provenance.json"),
                      {"provenance": common.provenance(__file__), "backend": args.backend, "model_hash": mh,
                       "n_points": len(pts), "n_rows": len(rows_c), "seconds": time.time() - t0})
    print("wrote", pc, len(rows_c), "rows;", px, len(rows_x), "rows")


if __name__ == "__main__":
    main()
