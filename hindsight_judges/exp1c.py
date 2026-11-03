"""Experiment 1c (protocol v1): separate hindsight world evidence from factual-action provenance.

CPU: selftest / preflight / prepare / lengths / analyze. GPU: score (32B ONLY, chunked prefill under SDPA MATH).
No original P4 data, frozen hindsight experiment-one code, or model weights are modified.

Input: original P4 E2 L/S and E2-M M; P4 bank-1 replay action traces (rep=0),
original P4 task_list game map, and ALFWorld games. Rollout JSON is optional for cross-checks.
Replay observations are reconstructed
by CPU deterministic re-feed, with outcome, action, one-step-feedback and divergence gates.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONDITIONS = ("M", "L", "LN", "LT", "D0", "DE", "DX", "D1F", "D1B")
N_POINTS = 386
NBOOT = 4000
BOOT_SEED = 20261103
MAX_TOKENS = 32768
FROZEN_STATUS = "PROTOCOL 1c v1: frozen when the researcher approves this commit (--approved-commit)"
UNCHUNKED_CHECK_MAX = 4096  # contexts short enough for the frozen one-pass MATH scorer on an 85 GB H800
GAP = 1 / 3
BRANCH_SEED = "Hindsight-1c-20261103-v1"
# Same verb as the verified E1 execution label ("executed"); names the branch's first action, because a
# replay branch is not the historical trajectory itself.
LABEL = ("Execution record for the marked decision: the original agent executed the first action "
         "of branch {branch}.")
# Deliberately says nothing about who chose the displayed action (E1: an "evaluation marker" was itself
# a strong pointer). Showing one branch is the manipulation; no provenance is stated.
SINGLE_INTRO = (
    "The following is one replay continuation from the marked decision state. It starts with one of "
    "the candidate actions and then follows the same continuing Agent policy, with the same remaining "
    "50-step episode budget. The other four candidate actions have no displayed continuation. "
    "Judge all five original candidate actions."
)
DUAL_INTRO = (
    "The following two replay continuations start at the same marked decision state. "
    "Each uses one candidate action and then the same continuing Agent policy, with the same "
    "remaining 50-step episode budget. A/B are arbitrary names with randomized ordering. "
    "Both branches are equally valid counterfactual replay evidence, and three other candidate "
    "actions have no displayed continuation. Judge all five original candidate actions."
)
LT_INTRO = (
    "The following is the factual episode truncated just after the marked decision's "
    "action and environment observation. Subsequent actions and terminal reward are omitted.\n"
)  # trailing newline: same layout as the frozen log.txt / memory.txt templates
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.I | re.S)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text())


def save(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_rows(path: Path, expected: str):
    raw = gzip.decompress(path.read_bytes())
    assert sha(raw) == expected, f"decompressed source hash mismatch: {path}"
    return [json.loads(x) for x in raw.decode().splitlines()]


def original_sources(repo: Path):
    p4 = repo / "exploratory_P4"
    h1 = repo / "hindsight_judges" / "results" / "exp1_p4"
    p_e2 = load(p4 / "e2_preparation.json")
    p_m = load(p4 / "e2m_preparation.json")
    p_1 = load(h1 / "manifest.json")
    pmeta = load(h1 / "pointmeta.json")
    assert len(pmeta) == N_POINTS and len({z["point_id"] for z in pmeta}) == N_POINTS
    assert file_sha(h1 / "pointmeta.json") == p_1["pointmeta_sha256"]
    assert p_1["points"] == N_POINTS and p_1["contexts"] == N_POINTS*10
    files = {
        "P4-E2": p4 / "e2_requests.jsonl.gz",
        "P4-E2-PREP": p4 / "e2_preparation.json",
        "P4-E2-M": p4 / "e2m_requests.jsonl.gz",
        "P4-E2-M-PREP": p4 / "e2m_preparation.json",
        "P4-L-TEMPLATE": p4 / "prompts" / "log.txt",
        "P4-EXP1-POINTMETA": h1 / "pointmeta.json",
        "P4-EXP1-MANIFEST": h1 / "manifest.json",
        "P4-TASKLIST": repo / "data" / "task_list.json",
    }
    hashes = {key: file_sha(path) for key, path in files.items()}
    raw_e2 = source_rows(files["P4-E2"], p_e2["requests_sha256"])
    raw_m = source_rows(files["P4-E2-M"], p_m["requests_sha256"])
    pids = {z["point_id"] for z in pmeta}
    e2, e2m = defaultdict(dict), defaultdict(dict)
    for r in raw_e2:
        if r["point_id"] in pids and r["condition"] in ("S", "L", "F"):
            assert r["condition"] not in e2[r["point_id"]]
            e2[r["point_id"]][r["condition"]] = r
    for r in raw_m:
        if r["point_id"] in pids and r["condition"] == "M":
            assert "M" not in e2m[r["point_id"]]
            e2m[r["point_id"]]["M"] = r
    assert set(e2) == pids and set(e2m) == pids
    for z in pmeta:
        pid = z["point_id"]
        assert set(e2[pid]) == {"S", "L", "F"}
        assert set(e2m[pid]) == {"M"}
        ss, ll, ff, mm = e2[pid]["S"], e2[pid]["L"], e2[pid]["F"], e2m[pid]["M"]
        assert ss["actions"] == ll["actions"] == ff["actions"] == mm["actions"]
        assert z["factual_arm"] == 0 and 1 <= z["swap_arm"] <= 4
        assert ff["text"].startswith(ss["text"]) and mm["text"].startswith(ss["text"])
        assert ll["text"].startswith(ss["text"])
    template = files["P4-L-TEMPLATE"].read_text()
    return sorted(pmeta, key=lambda x: x["point_id"]), e2, e2m, template, hashes


def extract_episode(S: dict, L: dict, log_template: str):
    prefix = S["text"] + "\n" + log_template
    assert L["text"].startswith(prefix), "original P4 L template mismatch"
    episode = json.loads(L["text"][len(prefix):])
    t = int(S["turn"])
    assert len(episode["steps"]) >= t
    assert all(int(st["turn"]) == i + 1 for i, st in enumerate(episode["steps"]))
    assert episode["steps"][t - 1]["executed_action"] == S["actions"][0]
    return episode


def remove_visible_think(episode: dict):
    """Remove written <think> spans from rollout responses, NOT hidden model reasoning."""
    obj = json.loads(json.dumps(episode, ensure_ascii=False))
    n = 0
    for st in obj["steps"]:
        if "response" in st:
            text, hits = THINK_BLOCK.subn("", st["response"])
            st["response"] = text
            n += hits
    return obj, n


def truncate_after_t(episode: dict, t: int):
    obj = dict(task=episode["task"], initial_observation=episode["initial_observation"],
               steps=episode["steps"][:t])
    assert len(obj["steps"]) == t and "final_reward" not in obj
    return obj


def arm_feedback(S: dict, F: dict, repo: Path):
    """The original F condition already includes a checked 5-arm one-step peek.
    Reconstruct its action->response association WITHOUT looking at alternative outcomes.
    """
    base = S["text"] + "\n" + (repo / "exploratory_P4" / "prompts" / "feedback.txt").read_text()
    assert F["text"].startswith(base)
    fb = json.loads(F["text"][len(base):])
    assert len(fb) == 5
    assert len({z["action"] for z in fb}) == 5, "F has duplicate candidates"
    return {z["action"]: z["response"] for z in fb}


def replay_continuation(rec: dict, arm: int):
    """Never select by reward. Strictly bank-1 replica zero for both arms."""
    role = 1 if arm == 0 else arm + 1
    hits = [c for c in rec["continuations"] if int(c["role"]) == role and int(c["rep"]) == 0]
    assert len(hits) == 1, f"expected exactly one bank-1 rep-0 for role={role}"
    return hits[0]


def validate_replay_header(rec: dict, traj: dict, S: dict, alt: int):
    t = int(S["turn"])
    assert rec["traj_id"] == traj["traj_id"] == S["traj_id"]
    assert int(rec["turn"]) == t
    assert rec["cf_defined"] is True
    assert rec["factual_action"] == S["actions"][0]
    assert rec["alternatives"] == S["actions"][1:]
    assert traj["steps"][t-1]["action"] == S["actions"][0]
    c_f = replay_continuation(rec, 0)
    c_b = replay_continuation(rec, alt)
    assert c_f["first_action"] == S["actions"][0]
    assert c_b["first_action"] == S["actions"][alt]
    assert int(rec["factual_outcomes"][0]) == int(c_f["won"])
    assert int(rec["alt_outcomes"][alt-1][0]) == int(c_b["won"])
    assert int(c_f["refeed_div"]) == int(c_b["refeed_div"]) == 0
    return c_f, c_b


def reconstruct(game, traj: dict, t: int, continuation: dict, first_obs_expected: str):
    """Game instance implementing AlfGame.refeed and step. Deterministic CPU replay only."""
    prefix = traj["steps"][:t-1]
    obs, adm, won, div = game.refeed([s["action"] for s in prefix],
                                     [s["obs_after"] for s in prefix])
    assert div == 0, f"divergent prefix on {traj['traj_id']} t{t:02d}"
    assert obs == traj["steps"][t-1]["obs_before"], "marked-state observation mismatch"
    out = []
    actions = continuation["actions"]
    assert actions and actions[0] == continuation["first_action"]
    done = False
    for k, action in enumerate(actions):
        assert not done and not won, "continuation illegally continues after termination"
        obs, adm, won, done = game.step(action)
        out.append({"turn":t+k,"action":action,"observation_after":obs})
        if k == 0:
            assert obs == first_obs_expected, "first feedback differs from P4 aligned F"
    assert int(won) == int(continuation["won"]), "replay outcome changed"
    assert int(continuation["len"]) == t - 1 + len(actions), "replay total length mismatch"
    assert won or done or t - 1 + len(actions) == 50, "replay ended before episode budget"
    return {"steps":out,"terminal_success":int(won)}


def branch_order(pid: str):
    coin = int(sha(f"{BRANCH_SEED}:{pid}".encode())[:16],16) & 1
    return ("A", "B") if coin == 0 else ("B", "A")


def dual_text(M_text: str, pid: str, fac: dict, alt: dict):
    fact_label, alt_label = branch_order(pid)
    parts = {fact_label:fac, alt_label:alt}
    # The dict itself has no keys such as factual, original, contrast, q12 or arm index.
    doc = {"branch_A":parts["A"],"branch_B":parts["B"]}
    shared = M_text + "\n" + DUAL_INTRO + "\n" + json.dumps(doc, ensure_ascii=False)
    d_true = shared + "\n" + LABEL.format(branch=fact_label)
    d_swap = shared + "\n" + LABEL.format(branch=alt_label)
    assert d_true.replace(LABEL.format(branch=fact_label), "") == d_swap.replace(LABEL.format(branch=alt_label), "")
    return {"D0":shared,"DE":d_true,"DX":d_swap}, dict(factual_branch=fact_label,contrast_branch=alt_label)


def single_text(M_text: str, branch: dict):
    """One anonymous replay branch (f's for D1F, b's for D1B), same JSON layout as one dual branch."""
    return M_text + "\n" + SINGLE_INTRO + "\n" + json.dumps({"replay": branch}, ensure_ascii=False)


def make_request(S: dict, condition: str, text: str):
    row=dict(request_id=f"{S['point_id']}__{condition}",point_id=S["point_id"],
             traj_id=S["traj_id"],turn=int(S["turn"]),condition=condition,
             text=text,actions=S["actions"])
    row["request_sha256"] = sha(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
    return row


def task_map(repo:Path):
    task_file=repo/"data"/"task_list.json"
    obj=load(task_file)
    tasks={int(z["seed"]):z for z in obj["tasks"]}
    assert len(tasks)==len(obj["tasks"]) and len(tasks)>=100
    for task in tasks.values():
        assert task["game_sha256"] and task["game_file"].endswith("/game.tw-pddl")
    return tasks


def task_for(traj_id: str, tasks: dict):
    match=re.fullmatch(r"s(\d{3})",traj_id)
    assert match, f"P4 cohort trajectory ID incompatible with task_list seed: {traj_id}"
    return tasks[int(match.group(1))]


def factual_trace(episode:dict, traj_id:str, task:dict):
    """Rebuild *factual prefix* directly from frozen E2 L (not absent rollout JSON).
    P4 E2 L includes each prior action and post-step observation; task_list has the game path.
    """
    before=episode["initial_observation"]
    steps=[]
    for index,st in enumerate(episode["steps"],1):
        assert int(st["turn"])==index
        assert isinstance(st["observation_after"],str)
        steps.append(dict(turn=index,action=st["executed_action"],obs_before=before,
                          obs_after=st["observation_after"]))
        before=st["observation_after"]
    return dict(traj_id=traj_id,game_file=task["game_file"],task=episode["task"],steps=steps)


def verify_optional_original_rollout(repo, trace, episode):
    """If source P4 rollout JSON is also present, require exact consistency. Not a prerequisite."""
    path=repo/"data"/"rollout"/(trace["traj_id"]+".json")
    if not path.exists():return
    r=load(path)
    assert r["traj_id"]==trace["traj_id"] and r["game_file"]==trace["game_file"]
    assert r["task"]==episode["task"] and r["initial_obs"]==episode["initial_observation"]
    assert len(r["steps"])==len(trace["steps"])
    for a,b in zip(r["steps"],trace["steps"]):
        assert int(a["turn"])==b["turn"] and a["action"]==b["action"]
        assert a["obs_before"]==b["obs_before"] and a["obs_after"]==b["obs_after"]


def prerequisite_paths(repo, replay_root, pointmeta):
    tasks=task_map(repo)
    replays={replay_root/z["traj"]/f"t{int(z['point_id'].rsplit('_t',1)[1]):02d}.json" for z in pointmeta}
    games_root=Path(os.environ.get("ALFWORLD_DATA", "/root/autodl-tmp/alfworld_data"))
    games={games_root/task_for(z["traj"],tasks)["game_file"] for z in pointmeta}
    missing_replays=[str(p) for p in sorted(replays) if not p.exists()]
    missing_games=[str(p) for p in sorted(games) if not p.exists()]
    return sorted(replays), sorted(games), missing_replays, missing_games


def preflight(args):
    repo=Path(args.repo).resolve()
    replay_root=Path(args.replay_root).resolve()
    meta, e2, e2m, template, hashes=original_sources(repo)
    replays,games,missing_replays,missing_games=prerequisite_paths(repo,replay_root,meta)
    print(f"P4 I/U Fork points: {len(meta)}. P4 bank-1 replay records: {len(replays)}. "
          f"ALFWorld game files: {len(games)}. Missing: {len(missing_replays)} replay / "
          f"{len(missing_games)} games")
    if missing_replays:
        print("Missing P4 bank-1 files (first 8):",*missing_replays[:8],sep="\n  ")
        raise SystemExit("PRE-FLIGHT BLOCKED: original P4 bank-1 action sequences are unavailable. "
                         "They are not in the public git tree; supply --replay-root from the P4 server archive. "
                         "Do NOT substitute E4 replay or regenerate trajectories.")
    if missing_games:
        print("Missing ALFWorld game paths (first 5):",*missing_games[:5],sep="\n  ")
        raise SystemExit("PRE-FLIGHT BLOCKED: set ALFWORLD_DATA to the original P4 game files")
    for p in replays:assert p.stat().st_size>100, f"incomplete replay: {p}"
    print("PRE-FLIGHT PASS: original P4 bank-1 files and game paths available. "
          "P4 E2 L and task_list supply factual prefixes; rollout JSON is optional for cross-checks.")


def prepare(args):
    repo=Path(args.repo).resolve()
    out=Path(args.out).resolve()
    replay_root=Path(args.replay_root).resolve()
    meta,e2,e2m,template,hashes=original_sources(repo)
    replays,games,missing_replays,missing_games=prerequisite_paths(repo,replay_root,meta)
    assert not missing_replays and not missing_games, "Missing original P4 bank-1 or games; run preflight"
    tasks=task_map(repo)
    game_hash_cache={}
    sys.path.insert(0,str(repo/"scripts"))
    import alfenv  # noqa: import only once input sources have been verified; no GPU imports
    env=None
    current_game=None
    records=[]; audit=[]; n_think=0
    try:
        for index,z in enumerate(meta):
            pid=z["point_id"];t=int(pid.rsplit("_t",1)[1]);b=int(z["swap_arm"])
            S=e2[pid]["S"];L=e2[pid]["L"];F=e2[pid]["F"];M=e2m[pid]["M"]
            ep=extract_episode(S,L,template)
            logged_nocot,hits=remove_visible_think(ep)
            n_think+=hits
            ln=S["text"]+"\n"+template+json.dumps(logged_nocot,ensure_ascii=False)
            lt=S["text"]+"\n"+LT_INTRO+json.dumps(truncate_after_t(ep,t),ensure_ascii=False)
            rp=replay_root/z["traj"]/f"t{t:02d}.json"
            replay=load(rp)
            task=task_for(z["traj"],tasks)
            tr=factual_trace(ep,z["traj"],task)
            verify_optional_original_rollout(repo,tr,ep)
            c0,cb=validate_replay_header(replay,tr,S,b)
            game_path=Path(alfenv.ALFWORLD_DATA)/task["game_file"]
            assert game_path.exists(),f"Missing ALFWorld game {game_path}; set ALFWORLD_DATA"
            if str(game_path) not in game_hash_cache:
                game_hash_cache[str(game_path)]=file_sha(game_path)
            assert game_hash_cache[str(game_path)]==task["game_sha256"],f"ALFWorld game hash mismatch {game_path}"
            if current_game!=str(game_path):
                if env is None:env=alfenv.AlfGame(str(game_path))
                else:env.load(str(game_path))
                current_game=str(game_path)
            feedback=arm_feedback(S,F,repo)
            f_branch=reconstruct(env,tr,t,c0,feedback[S["actions"][0]])
            b_branch=reconstruct(env,tr,t,cb,feedback[S["actions"][b]])
            dual,order=dual_text(M["text"],pid,f_branch,b_branch)
            texts={"M":M["text"],"L":L["text"],"LN":ln,"LT":lt,**dual,
                   "D1F":single_text(M["text"],f_branch),"D1B":single_text(M["text"],b_branch)}
            assert set(texts)==set(CONDITIONS)
            records.extend(make_request(S,c,texts[c]) for c in CONDITIONS)
            audit.append(dict(point_id=pid,traj=z["traj"],turn=t,layer=z["layer"],
                swap_arm=b,q12_gap=float(z["q12_gap"]),think_blocks=hits,
                factual_branch=order["factual_branch"],contrast_branch=order["contrast_branch"],
                bank=1,rep=0,factual_outcome=f_branch["terminal_success"],
                contrast_outcome=b_branch["terminal_success"],
                factual_steps=len(f_branch["steps"]),contrast_steps=len(b_branch["steps"]),
                replay_sha256=file_sha(rp),
                factual_episode_sha256=sha(json.dumps(ep,sort_keys=True,ensure_ascii=False).encode()),
                game_sha256=game_hash_cache[str(game_path)]))
            if (index+1)%50==0:print(f"CPU replay {index+1}/{len(meta)}",flush=True)
    finally:
        if env is not None:env.close()
    assert len(records)==N_POINTS*len(CONDITIONS) and len(audit)==N_POINTS
    # P4 rollouts may include text <think>; if none exist, mark the LN contrast non-operative.
    # We never assume hidden CoT is available.
    out.mkdir(parents=True,exist_ok=True)
    body=("\n".join(json.dumps(x,ensure_ascii=False) for x in records)+"\n").encode()
    arms_path=Path(args.arms).resolve()
    p4_e2=load(repo/"exploratory_P4"/"e2.json")
    expected_arms=p4_e2["inputs"]["exploratory_deep_discovery/exploratory_arms.parquet"]
    assert file_sha(arms_path)==expected_arms, "P4 arms source mismatch"
    hashes["P4-ARMS"]=expected_arms
    man=dict(status="PREPARED / NOT SCORED; "+FROZEN_STATUS,points=N_POINTS,requests=len(records),
             conditions=list(CONDITIONS),requests_sha256=sha(body),
             audit_sha256=None,source_sha256=hashes,
             replay_source_root=str(replay_root),
             think_blocks_removed=n_think,
             prepare_script_sha256=file_sha(Path(__file__)),
             git_commit=subprocess.run(["git","rev-parse","HEAD"],cwd=repo,text=True,capture_output=True).stdout.strip())
    if (out/"manifest.json").exists():
        old=load(out/"manifest.json")
        assert old["requests_sha256"]==man["requests_sha256"], "prepared requests changed"
        assert old["source_sha256"]==hashes, "source files changed"
        assert file_sha(out/"requests.jsonl.gz") and gzip.decompress((out/"requests.jsonl.gz").read_bytes())==body
        assert load(out/"audit.json")==audit
        print("Existing 1c preparation matches; no rewrite.")
        return
    (out/"requests.jsonl.gz").write_bytes(gzip.compress(body,mtime=0))
    save(out/"audit.json",audit)
    man["audit_sha256"]=file_sha(out/"audit.json")
    save(out/"manifest.json",man)
    print("PREPARED",len(records),"requests;",n_think,"visible think blocks removed.")


def prepared(out):
    man=load(out/"manifest.json")
    raw=gzip.decompress((out/"requests.jsonl.gz").read_bytes())
    assert sha(raw)==man["requests_sha256"]
    rows=[json.loads(x) for x in raw.decode().splitlines()]
    assert len(rows)==man["requests"]==N_POINTS*len(CONDITIONS)
    assert len({r["request_id"] for r in rows})==len(rows)
    for r in rows:
        without={k:v for k,v in r.items() if k!="request_sha256"}
        assert r["request_sha256"]==sha(json.dumps(without,sort_keys=True,ensure_ascii=False).encode())
    audit=load(out/"audit.json")
    assert file_sha(out/"audit.json")==man["audit_sha256"]
    assert len(audit)==N_POINTS
    for i,z in enumerate(audit):
        actual=rows[i*len(CONDITIONS):(i+1)*len(CONDITIONS)]
        assert [r["condition"] for r in actual]==list(CONDITIONS)
        assert all(r["point_id"]==z["point_id"] for r in actual)
        for c in ("D0", "DE", "DX", "D1F", "D1B"):
            assert actual[CONDITIONS.index(c)]["text"].startswith(actual[0]["text"]), f"{c} lacks history"
    return rows,audit,man


def get_tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model,local_files_only=True)


def tokenize(tok,job):
    prefix=tok.apply_chat_template([dict(role="user",content=job["text"])],tokenize=False,add_generation_prompt=True)+"<action>"
    pre=tok.encode(prefix,add_special_tokens=False)
    full=[tok.encode(prefix+a+"</action>"+tok.eos_token,add_special_tokens=False) for a in job["actions"]]
    start=len(pre)
    for ids in full:
        i=0
        while i<min(len(pre),len(ids)) and ids[i]==pre[i]:i+=1
        start=min(start,i)
    assert len(pre)-start<=3
    assert all(ids[:start]==full[0][:start] and ids[-1]==tok.eos_token_id for ids in full)
    return dict(**job,prefix_ids=full[0][:start],candidates=[dict(suffix_ids=ids[start:]) for ids in full],
                max_tokens=max(map(len,full)))


def lengths(args):
    out=Path(args.out).resolve()
    jobs,audit,man=prepared(out)
    tok=get_tokenizer(Path(args.model).resolve())
    lens=[(tokenize(tok,j)["max_tokens"],j["point_id"],j["condition"]) for j in jobs]
    top=sorted(lens,reverse=True)[:10]
    print("Request length gate: max",max(x[0] for x in lens),"of",MAX_TOKENS,"tokens")
    print("Longest 10:",*top,sep="\n  ")
    if top[0][0]>MAX_TOKENS:
        raise SystemExit("BLOCKED: dual/full contexts exceed frozen 32,768 token cap. "
                         "No silent truncation or selective exclusions. Revise before GPU scoring.")
    print("LENGTH GATE PASS for all",len(lens),"requests")


def run_fingerprint():
    import torch
    return {"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,
            "cuda":torch.version.cuda,"sdpa_math":bool(torch.backends.cuda.math_sdp_enabled()),
            "sdpa_flash":bool(torch.backends.cuda.flash_sdp_enabled()),
            "sdpa_mem_efficient":bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "sdpa_cudnn":bool(torch.backends.cuda.cudnn_sdp_enabled())}


def score(args):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    if not torch.cuda.is_available():raise SystemExit("CUDA GPU required. No scoring started.")
    # Entire model load and scoring run use same MATH kernel as verified 32B experiment 1.
    with sdpa_kernel(SDPBackend.MATH):
        _score_math(args)


def _score_math(args):
    import torch
    repo=Path(args.repo).resolve();out=Path(args.out).resolve()
    jobs,audit,man=prepared(out)
    model=Path(args.model).resolve()
    original=load(repo/"exploratory_P4"/"e2_32b"/"run.json")
    assert original["complete"] and args.revision==original["revision"]
    weights={p.name:file_sha(p) for p in sorted(model.glob("*.safetensors"))}
    assert weights and weights==original["model_sha256"],"32B weights differ from frozen P4"
    backend=repo/"exploratory_P4"/"p4_e2_backend.py"
    assert file_sha(backend)==original["scorer_sha256"],"P4 scorer changed"
    env_meta=load(repo/"exploratory_P4"/"e2_environment.json")
    for name,digest in env_meta["model_metadata_hashes"]["32B"].items():
        assert file_sha(model/name)==digest,"Qwen metadata changed: "+name
    tok=get_tokenizer(model)
    # All token lengths checked BEFORE model load, and no point can be silently removed.
    lens={j["request_id"]:tokenize(tok,j)["max_tokens"] for j in jobs}
    maxlen=max(lens.values())
    assert maxlen<=MAX_TOKENS, f"context length {maxlen} exceeds {MAX_TOKENS}"
    spec=importlib.util.spec_from_file_location("frozen_p4_scorer",backend)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.MODEL=str(model)
    chunk_src=HERE/"exp1c_backend.py"
    cspec=importlib.util.spec_from_file_location("exp1c_chunked_scorer",chunk_src)
    cmod=importlib.util.module_from_spec(cspec);cspec.loader.exec_module(cmod)
    fingerprint=run_fingerprint()
    assert fingerprint["sdpa_math"] and not any(fingerprint[k] for k in ("sdpa_flash","sdpa_mem_efficient","sdpa_cudnn"))
    run_meta=dict(model="32b",revision=args.revision,model_sha256=weights,
        backend_sha256=file_sha(backend),chunked_backend_sha256=file_sha(chunk_src),chunk=cmod.CHUNK,
        script_sha256=file_sha(Path(__file__)),
        protocol_sha256=file_sha(HERE/"EXPERIMENT1C_DRAFT.md"),
        requests_sha256=man["requests_sha256"],audit_sha256=man["audit_sha256"],
        fingerprint=fingerprint, max_tokens=maxlen)
    target=out/"32b";target.mkdir(parents=True,exist_ok=True)
    run_path=target/"run.json";score_path=target/"scores.jsonl"
    if run_path.exists():
        prev=load(run_path)
        assert all(prev.get(k)==v for k,v in run_meta.items()),"resume would mix environments or code"
    else:
        assert not score_path.exists(),"scores present without a matching run manifest"
    done={}
    if score_path.exists():
        for line in score_path.read_text().splitlines():
            r=json.loads(line);assert r["request_id"] not in done
            done[r["request_id"]]=r
    src={j["request_id"]:j for j in jobs}
    assert set(done)<=set(src)
    assert all(done[k]["request_sha256"]==src[k]["request_sha256"] for k in done)
    if run_path.exists() and prev.get("complete"):
        assert len(done)==len(jobs) and file_sha(score_path)==prev["scores_sha256"]
        print("Already scored and verified; stopping without overwrite");return
    save(run_path,{**run_meta,"complete":False,"contexts":len(done)})
    model_start=time.time();frozen=module.Scorer();scorer=cmod.ChunkedScorer(frozen)
    assert run_fingerprint()==fingerprint
    def prob(lps):
        x=np.array([sum(x) for x in lps],float);y=np.exp(x-x.max());return y/y.sum()
    # Longest context first: an out-of-memory failure must surface before anything else runs.
    longest=max(jobs,key=lambda j:lens[j["request_id"]])
    check_jobs=[longest]+[next(j for j in jobs if j["condition"]==c) for c in CONDITIONS]
    # Chunking equivalence: shortest context of each condition, if short enough for one-pass MATH.
    short=[min((j for j in jobs if j["condition"]==c),key=lambda j:lens[j["request_id"]]) for c in CONDITIONS]
    short=[j for j in short if lens[j["request_id"]]<=UNCHUNKED_CHECK_MAX]
    checks=[]
    try:
        for item in check_jobs:
            j=tokenize(tok,item)
            a=prob(scorer.score(j));b=prob([scorer.direct(j,k) for k in range(5)])
            tv=float(np.abs(a-b).sum()/2)
            checks.append(dict(kind="chunked_shared_prefix_vs_chunked_direct",request_id=item["request_id"],
                               tokens=lens[item["request_id"]],tv=tv,passed=tv<0.005))
        for item in short:
            j=tokenize(tok,item)
            a=prob(scorer.score(j));b=prob(frozen.score(j))
            tv=float(np.abs(a-b).sum()/2)
            checks.append(dict(kind="chunked_vs_frozen_one_pass",request_id=item["request_id"],
                               tokens=lens[item["request_id"]],tv=tv,passed=tv<0.005))
    except torch.cuda.OutOfMemoryError as exc:
        save(target/"numerical_check.json",checks+[dict(kind="OOM",error=str(exc)[:500],
             peak_gb=float(torch.cuda.max_memory_allocated()/1e9))])
        raise SystemExit("OUT OF MEMORY in the numerical gate; NO score file written. "
                         "Do not change kernels or chunk size without a recorded amendment.")
    save(target/"numerical_check.json",checks)
    assert sum(x["kind"]=="chunked_vs_frozen_one_pass" for x in checks)>=3, \
        "fewer than 3 chunking-equivalence checks possible; review before scoring"
    assert all(x["passed"] for x in checks),"Numerical gate failed; NO score file written"
    start=time.time()
    with score_path.open("a",buffering=1) as f:
        for idx,j0 in enumerate(jobs):
            if j0["request_id"] in done:continue
            j=tokenize(tok,j0)
            ll=np.array([sum(x) for x in scorer.score(j)],float)
            assert np.isfinite(ll).all()
            y=np.exp(ll-ll.max());p=y/y.sum()
            row=dict(request_id=j0["request_id"],point_id=j0["point_id"],
                condition=j0["condition"],request_sha256=j0["request_sha256"],
                candidate_prob=p.tolist(),sequence_logp=ll.tolist(),argmax=int(p.argmax()))
            f.write(json.dumps(row,allow_nan=False)+"\n");done[row["request_id"]]=row
            if (idx+1)%len(CONDITIONS)==0:
                f.flush();os.fsync(f.fileno())
                save(target/"progress.json",{"contexts":len(done),"total":len(jobs),"seconds":time.time()-start})
    assert len(done)==len(jobs)
    save(run_path,{**run_meta,"complete":True,"contexts":len(done),
                   "scores_sha256":file_sha(score_path),"model_load_seconds":time.time()-model_start,
                   "score_seconds":time.time()-start,
                   "peak_gb":float(torch.cuda.max_memory_allocated()/1e9)})
    print("SCORING DONE",len(done))


def mean_ci(values,trajectories,seed=BOOT_SEED,nboot=NBOOT):
    v=np.array(values,float);tt=np.array(trajectories)
    good=np.isfinite(v);v=v[good];tt=tt[good]
    if not len(v):return dict(n=0,estimate=None,ci95=None)
    tids=sorted(set(tt));idx={t:i for i,t in enumerate(tids)}
    gi=np.array([idx[t] for t in tt]);sums=np.bincount(gi,weights=v,minlength=len(tids))
    counts=np.bincount(gi,minlength=len(tids))
    # Same seed and cluster resampling as preliminary design; each metric sees identical
    # resamples within an identical trajectory support.
    W=np.random.default_rng(seed).multinomial(len(tids),np.full(len(tids),1/len(tids)),size=nboot)
    denom=W@counts;dist=(W@sums)/denom
    return dict(n=len(v),n_trajectories=len(tids),estimate=float(v.mean()),
                ci95=[float(x) for x in np.percentile(dist,[2.5,97.5])])


def read_arms(repo:Path,arms_path:Path,expected_sha:str):
    assert file_sha(arms_path)==expected_sha,"held-out q bank changed"
    sys.path.insert(0,str(repo/"exploratory_P4"/"post_e2"))
    import minipq
    a=minipq.read(str(arms_path));out={}
    for i in range(len(a["traj_id"])):
        pid=f"{a['traj_id'][i]}_t{int(a['turn'][i]):02d}"
        j=int(a["j"][i])
        if pid not in out:out[pid]={}
        if j not in out[pid]:
            def number(x):return float(x) if x is not None else float("nan")
            out[pid][j]=dict(q12=number(a["q_12"][i]),
                q3456=number(a["q_3456"][i]),qgold=number(a["q_gold"][i]))
    return out


def analyze(args):
    repo=Path(args.repo).resolve();out=Path(args.out).resolve()
    jobs,audit,man=prepared(out)
    target=out/"32b";run=load(target/"run.json")
    assert run["complete"] and run["contexts"]==len(jobs)
    assert run["requests_sha256"]==man["requests_sha256"] and run["audit_sha256"]==man["audit_sha256"]
    assert file_sha(target/"scores.jsonl")==run["scores_sha256"]
    expected={j["request_id"]:j for j in jobs};scores={}
    for line in (target/"scores.jsonl").read_text().splitlines():
        r=json.loads(line);assert r["request_id"] not in scores
        assert r["request_id"] in expected and r["request_sha256"]==expected[r["request_id"]]["request_sha256"]
        assert len(r["candidate_prob"])==len(r["sequence_logp"])==5
        p=np.array(r["candidate_prob"],float);ll=np.array(r["sequence_logp"],float)
        assert np.isfinite(p).all() and np.isfinite(ll).all() and np.all(p>=0) and abs(p.sum()-1)<1e-6
        scores[r["request_id"]]=r
    assert len(scores)==len(jobs)
    arms=read_arms(repo,Path(args.arms).resolve(),man["source_sha256"]["P4-ARMS"])
    groups={"all":lambda a:True,"I":lambda a:a["layer"]=="I","U":lambda a:a["layer"]=="U",
        "bad_q12_ge_third":lambda a:a["q12_gap"]>=GAP-1e-8,
        "gold_all":lambda a:True,
        "gold_bad_q12_ge_third":lambda a:a["q12_gap"]>=GAP-1e-8}
    metrics={}
    for name,selector in groups.items():
        d=defaultdict(list);traj=defaultdict(list)
        def put(key,value,tr):
            d[key].append(value);traj[key].append(tr)
        for a in audit:
            if not selector(a):continue
            pid=a["point_id"];b=a["swap_arm"];tr=a["traj"]
            qkey="qgold" if name.startswith("gold") else "q3456"
            vals=np.array([arms[pid][j][qkey] for j in range(5)],float)
            if not np.isfinite(vals).all():
                assert name.startswith("gold"),"primary heldout q3456 is missing"
                continue
            s={k:scores[pid+"__"+k] for k in CONDITIONS}
            p={k:np.array(s[k]["candidate_prob"],float) for k in CONDITIONS}
            ll={k:np.array(s[k]["sequence_logp"],float) for k in CONDITIONS}
            lo={k:ll[k][0]-ll[k][b] for k in CONDITIONS}
            value={k:float(p[k]@vals) for k in CONDITIONS}
            others=[j for j in range(5) if j not in (0,b)]
            def pull(cond,arm,ref="M"):
                dd=ll[cond]-ll[ref];return float(dd[arm]-dd[others].mean())
            fo,bo=int(a["factual_outcome"]),int(a["contrast_outcome"])
            measurements={
                # H1: identity label with both full branches displayed and fixed
                "dual_pointer_swap_nat":(lo["DE"]-lo["DX"])/2,
                "dual_pointer_true_minus_anon_nat":lo["DE"]-lo["D0"],
                # H3: decision value of adding the true label to symmetric evidence
                "dual_true_label_minus_anon_value":value["DE"]-value["D0"],
                "dual_true_minus_swap_label_value":value["DE"]-value["DX"],
                # descriptive only: D0 displays branches for f and the q12-selected b (oracle competitor)
                "dual_no_pointer_minus_L_value":value["D0"]-value["L"],
                "dual_no_pointer_minus_M_value":value["D0"]-value["M"],
                "dual_no_pointer_minus_true_value":value["D0"]-value["DE"],
                "dual_no_pointer_minus_swap_value":value["D0"]-value["DX"],
                "single_f_minus_M_value":value["D1F"]-value["M"],
                "L_minus_M_value":value["L"]-value["M"],
                "LN_minus_L_value":value["LN"]-value["L"],
                "LT_minus_L_value":value["LT"]-value["L"],
                "L_minus_M_factual_logodds_nat":lo["L"]-lo["M"],
                "LN_minus_L_factual_logodds_nat":lo["LN"]-lo["L"],
                "LT_minus_L_factual_logodds_nat":lo["LT"]-lo["L"],
                "D0_minus_M_factual_logodds_nat":lo["D0"]-lo["M"],
                "D1F_minus_M_factual_logodds_nat":lo["D1F"]-lo["M"],
                "M_argmax_factual":float(p["M"].argmax()==0),
                "L_argmax_factual":float(p["L"].argmax()==0),
                "D0_argmax_factual":float(p["D0"].argmax()==0),
                "DE_argmax_factual":float(p["DE"].argmax()==0),
                "DX_argmax_factual":float(p["DX"].argmax()==0),
                "D1F_argmax_factual":float(p["D1F"].argmax()==0),
                "q3456_f_minus_b":vals[0]-vals[b],
                "dual_branch_length_diff":a["factual_steps"]-a["contrast_steps"],
            }
            for k,v in measurements.items():put(k,v,tr)
            # H1 secondary: both arms below the log-prob ceiling in D0 (E1 lesson: sequence log-prob <= 0)
            if ll["D0"][0]<-1 and ll["D0"][b]<-1:put("dual_pointer_swap_nat_headroom",(lo["DE"]-lo["DX"])/2,tr)
            put(f"dual_pointer_swap_nat_f{fo}b{bo}",(lo["DE"]-lo["DX"])/2,tr)
            # H2: pull toward the only displayed action, relative to M and to the three never-displayed arms,
            # pooled over displayed branches that FAILED (an information-only judge should not move toward it)
            for cond,arm,outcome in (("D1F",0,fo),("D1B",b,bo)):
                tag="failed" if outcome==0 else "succeeded"
                put(f"single_branch_pull_{tag}_nat",pull(cond,arm),tr)
                if ll["M"][arm]<-1:put(f"single_branch_pull_{tag}_nat_headroom",pull(cond,arm),tr)
            if fo==0:put("single_f_failed_minus_M_factual_logodds_nat",lo["D1F"]-lo["M"],tr)
        metrics[name]={key:mean_ci(vals,traj[key]) for key,vals in d.items()}
        q=np.array(d["q3456_f_minus_b"]);tq=traj["q3456_f_minus_b"]
        for target_name in ("L_minus_M_factual_logodds_nat","D0_minus_M_factual_logodds_nat",
                            "D1F_minus_M_factual_logodds_nat"):
            metrics[name][target_name]["slope_vs_heldout_q"]=slope_ci(q,np.array(d[target_name]),tq)
    main=metrics["all"];bad=metrics["bad_q12_ge_third"]
    lower=lambda z:z["ci95"] is not None and z["ci95"][0]>0
    upper=lambda z:z["ci95"] is not None and z["ci95"][1]<0
    none={"ci95":None}
    decisions={
        "H1_identity_label_under_symmetric_two_branch_evidence":lower(main.get("dual_pointer_swap_nat",none)),
        "H2_pull_toward_displayed_failed_branch":lower(main.get("single_branch_pull_failed_nat",none)),
        "H3_true_label_value_harm_on_q12_worse":upper(bad.get("dual_true_label_minus_anon_value",none)),
        "bootstrap_seed":BOOT_SEED,"nboot":NBOOT}
    findings=dict(status=FROZEN_STATUS,scope="P4 I/U Fork 386 points; bank1 rep0 branches for f and q12-best b",
                  analysis_seed=BOOT_SEED,nboot=NBOOT,conditions=list(CONDITIONS),
                  manifest_sha256=file_sha(out/"manifest.json"),
                  run=run,decisions=decisions,metrics=metrics)
    save(target/"analysis.json",findings)
    lines=["Experiment 1c: full hindsight logs, anonymous replay branches and execution labels (32B, one run).",
           "Values in pp, log odds in nats; 95% trajectory-cluster bootstrap.",
           "Predefined decisions: "+json.dumps(decisions),
           "CAUTION: D0/D1 display branches only for f and the q12-selected b; any D-vs-M/L value contrast",
           "includes that oracle selection and is descriptive, not evidence that removing the pointer helps.",
           "Causal identity manipulation is DE vs DX (identical two-branch evidence); H2 is the display-only pointer."]
    for g,data in metrics.items():
        lines.append("\n"+g)
        for k,z in data.items():
            if k=="q3456_f_minus_b" or z["n"]==0:continue
            unit="nat" if "nat" in k else ("turns" if "branch_length" in k else ("%" if "argmax" in k else "pp"))
            scale=1 if unit in ("nat","turns") else 100
            sl=z.get("slope_vs_heldout_q")
            extra=f" slope={sl['estimate']:+.3f} [{sl['ci95'][0]:+.3f},{sl['ci95'][1]:+.3f}]" if sl else ""
            lines.append(f"  {k:48s} {z['estimate']*scale:+.3f} [{z['ci95'][0]*scale:+.3f},{z['ci95'][1]*scale:+.3f}] {unit} n={z['n']} traj={z['n_trajectories']}{extra}")
    (target/"REPORT.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


def slope_ci(x,y,trajectories,seed=BOOT_SEED,nboot=1000):
    """OLS slope of y on x with a trajectory-cluster bootstrap CI (descriptive)."""
    x=np.asarray(x,float);y=np.asarray(y,float);tt=np.asarray(trajectories)
    if len(x)<3 or np.var(x)<1e-12:return None
    ids=sorted(set(tt));rows={t:np.where(tt==t)[0] for t in ids}
    def fit(ix):
        xx,yy=x[ix],y[ix];v=np.var(xx)
        return float(np.cov(xx,yy,bias=True)[0,1]/v) if v>1e-12 else np.nan
    rng=np.random.default_rng(seed);bs=[]
    for _ in range(nboot):
        pick=rng.integers(0,len(ids),len(ids));bs.append(fit(np.concatenate([rows[ids[i]] for i in pick])))
    bs=np.array(bs);bs=bs[np.isfinite(bs)]
    return dict(estimate=fit(np.arange(len(x))),ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))])


def selftest(_args):
    ep=dict(task="find key",initial_observation="start",steps=[
        {"turn":1,"executed_action":"open drawer","observation_after":"empty","response":"<think>I think.</think><action>open drawer</action>"},
        {"turn":2,"executed_action":"open cabinet","observation_after":"key","response":"<think>Now.</think><action>open cabinet</action>"}],final_reward=1)
    en,n=remove_visible_think(ep)
    assert n==2 and "<think>" not in en["steps"][0]["response"]
    assert "<action>open drawer</action>" in en["steps"][0]["response"]
    assert "<think>" in ep["steps"][0]["response"]
    cut=truncate_after_t(ep,1)
    assert len(cut["steps"])==1 and "final_reward" not in cut
    assert "observation_after" not in json.dumps(cut) or cut["steps"][-1]["observation_after"] == "empty"
    assert cut["steps"][-1]["observation_after"] == "empty"
    cont={"role":1,"rep":0,"first_action":"open drawer","actions":["open drawer","open cabinet"],"won":True,"len":2,"refeed_div":0}
    class FakeGame:
        def __init__(self):self.n=0
        def refeed(self,aa,bb):assert aa==bb==[];self.n=0;return "start",[],False,0
        def step(self,act):
            self.n+=1
            return ("empty" if self.n==1 else "key"),[],self.n==2,self.n==2
    tr=dict(traj_id="s000",steps=[{"obs_before":"start","action":"open drawer","obs_after":"empty"}])
    tr["steps"].append(dict(obs_before="empty",action="open cabinet",obs_after="key"))
    z=reconstruct(FakeGame(),tr,1,cont,"empty")
    assert z["terminal_success"]==1 and len(z["steps"])==2
    txt,roles=dual_text("M prompt", "s000_t01",z,{"steps":[{"turn":1,"action":"look","observation_after":"nothing"}],"terminal_success":0})
    assert txt["DE"].replace(LABEL.format(branch=roles["factual_branch"]),"")==txt["D0"]+"\n"
    assert txt["DX"].replace(LABEL.format(branch=roles["contrast_branch"]),"")==txt["D0"]+"\n"
    assert "execution record" not in txt["D0"].lower() and "original agent" not in txt["D0"].lower()
    assert LABEL.format(branch="A").replace("A.","B.")==LABEL.format(branch="B")
    one=single_text("M prompt",z)
    assert one.startswith("M prompt\n") and "branch_" not in one and "original agent" not in one.lower()
    assert "executed" not in SINGLE_INTRO.lower() and "marker" not in SINGLE_INTRO.lower()
    assert LT_INTRO.endswith("\n")
    assert replay_continuation({"continuations":[dict(cont,rep=1,won=False),cont]},0)==cont
    assert mean_ci([1,0,0],["A","A","B"],nboot=100)["estimate"]==1/3
    print("SELFTEST PASSED (written CoT, truncation, CPU replay, replica selection, two-branch label isolation, single-branch text, bootstrap)")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("selftest","preflight","prepare","lengths","score","analyze"))
    p.add_argument("--repo",default=str(ROOT))
    p.add_argument("--replay-root",default=str(ROOT/"data"/"replay"),
                   help="original P4 bank-1 records, NOT E4 replay")
    p.add_argument("--out",default=str(HERE/"results"/"exp1c_p4"))
    p.add_argument("--arms",default=str(ROOT/"exploratory_deep_discovery"/"exploratory_arms.parquet"))
    p.add_argument("--model",default=None,help="local frozen Qwen2.5-32B-Instruct checkpoint")
    p.add_argument("--revision",default="5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd")
    p.add_argument("--approved-commit",default=None,help="mandatory for GPU score: researcher-approved HEAD SHA")
    args=p.parse_args()
    if args.command in ("lengths","score") and not args.model:p.error("--model is required")
    if args.command == "score":
        if not args.approved_commit:p.error("GPU score requires researcher approval: --approved-commit <reviewed HEAD SHA>")
        current=subprocess.run(["git","rev-parse","HEAD"],cwd=args.repo,text=True,capture_output=True,check=True).stdout.strip()
        assert current == args.approved_commit, f"HEAD {current} differs from approved {args.approved_commit}"
        tracked=subprocess.run(["git","status","--porcelain","--","hindsight_judges/exp1c.py",
                "hindsight_judges/EXPERIMENT1C_DRAFT.md"],cwd=args.repo,text=True,capture_output=True,check=True).stdout.strip()
        assert not tracked, "approved script or protocol has uncommitted changes"
    globals()[args.command](args)


if __name__=="__main__":main()
