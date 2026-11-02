"""E4 step 3: judge contexts and 32B scoring for arms A (condition F) and H (condition L).

prepare : build F and L requests exactly as P4 E2 (exploratory_P4/p4_e2_judge_prepare.py: same templates, same display
          order rule, same candidate list), from data/points/*.json. -> data/judge/requests.jsonl.gz
score   : score one condition with the P4 backend (BF16, FP32 vocabulary projection, shared prefix), with the same
          tokenisation and the same direct-scoring TV < 0.005 gate. Scoring seconds (model load excluded) are recorded
          per condition in data/judge/{name}_{cond}/run.json -> cost matching (PREREG_E4 §4).
Usage: python e4_03_judge.py prepare
       python e4_03_judge.py score --cond F --model PATH --name 32b --revision REV
"""
import argparse
import glob
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

import e4_common as E

P4 = os.path.join(E.ROOT, "exploratory_P4")
OUTD = os.path.join(E.DATA, "judge")
sha = lambda b: hashlib.sha256(b).hexdigest()


def build_requests(rec, templates):
    pid, t = rec["point_id"], int(rec["turn"])
    actions = [rec["factual"]] + rec["alternatives"]
    order = E.display_order(pid)
    base = (templates["base"] + f"\nMarked decision: turn {t}.\n<decision_time_prompt>\n" + rec["prompt"]
            + "\n</decision_time_prompt>\nCandidates:\n" + json.dumps([actions[j] for j in order]))
    aligned = [dict(action=actions[i], response=rec["peeks"][i]["obs"]) for i in order]
    episode = dict(task=rec["task"], initial_observation=rec["initial_obs"], steps=rec["episode"], final_reward=rec["final_reward"])
    texts = {"F": base + "\n" + templates["feedback"] + json.dumps(aligned, ensure_ascii=False),
             "L": base + "\n" + templates["log"] + json.dumps(episode, ensure_ascii=False)}
    rows = []
    for cond, text in texts.items():
        row = dict(request_id=f"{pid}__{cond}", point_id=pid, turn=t, condition=cond, text=text, actions=actions)
        row["request_sha256"] = sha(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        rows.append(row)
    return rows


def prepare():
    templates = {k: open(os.path.join(P4, "prompts", f"{k}.txt")).read() for k in ["base", "log", "feedback"]}
    rows = []
    for f in sorted(glob.glob(os.path.join(E.DATA, "points", "*.json"))):
        rec = json.load(open(f))
        if rec["cf_defined"]:
            rows += build_requests(rec, templates)
    body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()
    os.makedirs(OUTD, exist_ok=True)
    open(os.path.join(OUTD, "requests.jsonl.gz"), "wb").write(gzip.compress(body, mtime=0))
    E.write_json(os.path.join(OUTD, "preparation.json"), dict(
        contexts=len(rows), points=len(rows) // 2, requests_sha256=sha(body),
        templates_sha256={k: sha(v.encode()) for k, v in templates.items()}))
    print(len(rows), "contexts")


def score(args):
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU")
    prep = json.load(open(os.path.join(OUTD, "preparation.json")))
    raw = gzip.decompress(open(os.path.join(OUTD, "requests.jsonl.gz"), "rb").read())
    assert sha(raw) == prep["requests_sha256"]
    jobs = [j for j in map(json.loads, raw.decode().splitlines()) if j["condition"] == args.cond]
    target = os.path.join(OUTD, f"{args.name}_{args.cond}")
    os.makedirs(target, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def tokenize(job):  # identical to exploratory_P4/p4_e2_judge_score.py
        prefix = tok.apply_chat_template([dict(role="user", content=job["text"])], tokenize=False, add_generation_prompt=True) + "<action>"
        pre = tok.encode(prefix, add_special_tokens=False)
        full = [tok.encode(prefix + a + "</action>" + tok.eos_token, add_special_tokens=False) for a in job["actions"]]
        start = len(pre)
        for ids in full:
            i = 0
            while i < min(len(pre), len(ids)) and pre[i] == ids[i]:
                i += 1
            start = min(start, i)
        assert len(pre) - start <= 3 and max(map(len, full)) <= 32768
        assert all(ids[:start] == full[0][:start] and ids[-1] == tok.eos_token_id for ids in full)
        return dict(**job, prefix_ids=full[0][:start], candidates=[dict(suffix_ids=ids[start:]) for ids in full])

    original = json.load(open(os.path.join(P4, f"e2_{args.name}", "run.json")))   # same checkpoint and backend as P4 E2
    assert args.revision == original["revision"], "revision differs from P4 E2"
    mh = {}
    for p in sorted(glob.glob(os.path.join(args.model, "*.safetensors"))):
        mh[os.path.basename(p)] = E.file_sha256(p)
    assert mh == original["model_sha256"], "checkpoint differs from P4 E2"
    src = os.path.join(P4, "p4_e2_backend.py")
    assert E.file_sha256(src) == original["scorer_sha256"], "backend differs from P4 E2"
    spec = importlib.util.spec_from_file_location("p4_backend", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MODEL = args.model
    scorer = mod.Scorer()
    checks = []
    for raw_job in [jobs[0], max(jobs, key=lambda j: len(j["text"]))]:
        j = tokenize(raw_job)
        a, b = scorer.score(j), [scorer.direct(j, k) for k in range(len(j["candidates"]))]
        prob = lambda lp: (lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum())(np.array([sum(y) for y in lp]))
        tv = float(abs(prob(a) - prob(b)).sum() / 2)
        checks.append(dict(request_id=j["request_id"], tv=tv, passed=tv < .005))
    E.write_json(os.path.join(target, "numerical_check.json"), checks)
    assert all(c["passed"] for c in checks), "shared-prefix/direct mismatch"
    out = os.path.join(target, "scores.jsonl")
    done = {json.loads(l)["request_id"] for l in open(out)} if os.path.exists(out) else set()
    t0 = time.time()
    with open(out, "a", buffering=1) as f:
        for raw_job in jobs:
            if raw_job["request_id"] in done:
                continue
            j = tokenize(raw_job)
            s = np.array([sum(v) for v in scorer.score(j)])
            p = np.exp(s - s.max()); p /= p.sum()
            f.write(json.dumps(dict(request_id=j["request_id"], request_sha256=j["request_sha256"], point_id=j["point_id"],
                                    condition=j["condition"], candidate_prob=p.tolist())) + "\n")
            done.add(j["request_id"])
    run_path = os.path.join(target, "run.json")
    prev = json.load(open(run_path)) if os.path.exists(run_path) else {"scoring_seconds": 0.0}
    E.write_json(run_path, dict(model=args.model, revision=args.revision, condition=args.cond, contexts=len(done),
                                complete=len(done) == len(jobs), requests_sha256=prep["requests_sha256"],
                                scoring_seconds=prev["scoring_seconds"] + time.time() - t0,
                                gpu=torch.cuda.get_device_name(0), backend_sha256=E.file_sha256(src)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "score"])
    ap.add_argument("--cond", choices=["F", "L"])
    ap.add_argument("--model")
    ap.add_argument("--name", default="32b")
    ap.add_argument("--revision")
    a = ap.parse_args()
    prepare() if a.stage == "prepare" else score(a)
