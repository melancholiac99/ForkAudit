"""User-authorized end-of-run driver (D-049): finish Phase 1D/2/3, save, sync, then /usr/bin/shutdown.

Runs detached (setsid nohup), independent of any SSH session:
  1. wait for the GPU queue (run_phase1d_gpu2.sh) to exit; a queue whose logs stop changing for STALE_MIN minutes is
     treated as hung and killed; a global deadline bounds the wait
  2. check each GPU product and re-run a missing step once (gold is resumable)
  3. CPU pipeline, each step with a timeout and one retry: G features, W1-model features, vLLM-bf16 B features, gold
     audit/replication (sets the B3 switch), main evaluation (fp32 and vLLM), extras, RESULTS.md
  4. append D-045 / D-049 summaries to DECISIONS.md from the JSON outputs; git commit code + results
  5. archive code, results, logs and new data products with a read-back sha256 check, fsync, sync
  6. write logs/phase1d_shutdown_status.json and run /usr/bin/shutdown -- also on any failure (errors are saved first)
"""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finish_phase1c import ROOT, save_archive, sha  # noqa: E402
from watch_phase1 import identity  # noqa: E402

PY = "/root/miniconda3/envs/m0/bin/python"
STALE_MIN = 90
DEADLINE = time.mktime(time.strptime("2026-10-27 09:00", "%Y-%m-%d %H:%M"))
STATUS = ROOT / "logs" / "phase1d_shutdown_status.json"
status = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": {}, "errors": []}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def save_status():
    STATUS.write_text(json.dumps(status, indent=2, default=str))
    with STATUS.open("rb") as f:
        os.fsync(f.fileno())


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def newest_log_mtime():
    ts = [p.stat().st_mtime for p in (ROOT / "logs").glob("1d_*.log")] + [(ROOT / "logs/phase1d_gpu2.log").stat().st_mtime]
    return max(ts)


def run(name, cmd, timeout_h, env=None, retries=1):
    """Run a step via scripts/run.sh; one retry; never raises."""
    for attempt in range(retries + 1):
        lp = ROOT / "logs" / f"1e_{name}{'' if attempt == 0 else f'_retry{attempt}'}.log"
        log(f"START {name} (attempt {attempt + 1}) -> {lp.name}")
        t0 = time.time()
        try:
            with lp.open("w") as f:
                r = subprocess.run(["scripts/run.sh"] + cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                   timeout=timeout_h * 3600, env={**os.environ, **(env or {})})
            ok = r.returncode == 0
            rc = r.returncode
        except subprocess.TimeoutExpired:
            ok, rc = False, "timeout"
        status["steps"][name] = {"ok": ok, "rc": rc, "attempt": attempt + 1, "seconds": round(time.time() - t0), "log": lp.name}
        save_status()
        log(f"END {name}: ok={ok} rc={rc}")
        if ok:
            return True
    status["errors"].append(f"{name} failed after {retries + 1} attempts")
    return False


def gpu_products():
    gold_log = (ROOT / "logs/1d_gold.log")
    gold_ok = gold_log.exists() and any(l.startswith("DONE 300 points") for l in gold_log.read_text().splitlines()) \
        and len(list((ROOT / "data/replay_gold").glob("*/*.json"))) == 300
    sib = ROOT / "data/siblings"
    return {"gold": gold_ok,
            "siblings": sib.exists() and len(list(sib.glob("*.json"))) == 700,
            "peek_hidden": (ROOT / "hidden_peek/index.parquet").exists() and (ROOT / "hidden_peek/prompt_end_26.npy").exists(),
            "peek_score": (ROOT / "data/scores_peek_fp32.parquet").exists(),
            "w1_judge": (ROOT / "data/w1_judge.parquet").exists()}


GPU_STEPS = {
    "gold": (["11_replay.py", "--engine", "async", "--bank", "6", "--points", "live", "--seeds", "0-99", "--subset_file",
              str(ROOT / "eval/gold_points.json"), "--n_rep", "20", "--out_root", str(ROOT / "data/replay_gold")], 8, None),
    "siblings": (["10_rollout.py", "--seeds", "0-99", "--siblings", "1-7"], 2, None),
    "peek_hidden": (["24_peek_states.py", "--part", "hidden"], 2, None),
    "peek_score": (["24_peek_states.py", "--part", "score"], 4, {"M0_HF_DTYPE": "fp32"}),
    "w1_judge": (["25_w1_judge.py"], 2, None),
}


def append_decisions():
    gold = json.loads((ROOT / "eval/gold.json").read_text()) if (ROOT / "eval/gold.json").exists() else None
    main = json.loads((ROOT / "eval/main.json").read_text()) if (ROOT / "eval/main.json").exists() else None
    ex = json.loads((ROOT / "eval/extras.json").read_text()) if (ROOT / "eval/extras.json").exists() else None
    L = ["", "## D-045 — Gold set: audit and independent-20 replication; B3 switch — INFO (auto, finish_phase1d.py) "
         + time.strftime("%Y-%m-%d"), ""]
    if gold:
        L.append(f"`scripts/42_gold.py` → `eval/gold.json`. Audit: `{json.dumps(gold['audit'])}`.")
        for r, v in gold["replication_candidate_layer"].items():
            rep = v["replication_indep20"]
            L.append(f"- {r}: flagged by the 12-run rule {v['n_flagged_12']}; independent-20 replication "
                     f"{rep['frac'] if rep else None} (CI {rep['ci95_cluster'] if rep else None}); null replication "
                     f"{v['null_replication_indep20']}; merged-32 (selection-biased) {v['replication_merged32_selection_biased']}.")
        L.append(f"- **B3 switch (cap replication < 0.50 → B3 report-only): {gold['B3_report_only_switch_fires']}.** "
                 "Applied mechanically per CLAUDE.md §6.3a.")
    else:
        L.append("Gold outputs missing; see logs/phase1d_shutdown_status.json.")
    L += ["", "## D-049 — Phase 3 run and end-of-run shutdown — INFO (auto, finish_phase1d.py) " + time.strftime("%Y-%m-%d"), "",
          "The user asked for saving code, results and logs, a verified sync, and then `/usr/bin/shutdown`, also on "
          "unrecoverable failure. The run was driven by the detached `scripts/finish_phase1d.py`.",
          f"- Steps: `{json.dumps({k: v['ok'] for k, v in status['steps'].items()})}`; errors: `{status['errors']}`."]
    if main:
        L.append(f"- Verdict (eval/main.json): `{json.dumps(main['verdict'], default=str)[:1500]}`.")
        L.append(f"- Predictions: `{json.dumps(main['predictions'], default=str)[:1500]}`.")
    if ex:
        L.append(f"- Gates: `{json.dumps(ex['gates'])}`.")
    L.append("- RESULTS.md was rendered by `scripts/43_results.py`; this entry has not been reviewed by a person.")
    with (ROOT / "DECISIONS.md").open("a") as f:
        f.write("\n".join(L) + "\n")


def git_commit():
    files = ["DECISIONS.md", "RESULTS.md", "scripts", "eval"]
    subprocess.run(["git", "add"] + [x for x in files if (ROOT / x).exists()], cwd=ROOT)
    r = subprocess.run(["git", "commit", "-q", "-m", "Phase 1D/2/3 end-of-run: gold, siblings, W1/G features, evaluation, "
                        "RESULTS.md, D-045, D-049 (finish_phase1d.py)\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
                       cwd=ROOT, capture_output=True, text=True)
    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"rc": r.returncode, "stdout": r.stdout[-500:], "stderr": r.stderr[-500:], "head": h}


def archive():
    """Code, results, logs and the new data products only (the earlier archives hold banks 1-6)."""
    import tarfile
    dest = ROOT / "artifacts" / ("phase1d_" + time.strftime("%Y%m%d_%H%M%S"))
    dest.mkdir(parents=True)
    files = []
    for folder in ("scripts", "eval", "logs", "proposals", "prompts/siblings", "data/replay_gold", "data/siblings",
                   "data/probes", "data/peek_alts", "hidden_peek"):
        if (ROOT / folder).exists():
            files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                      and not p.name.endswith((".tmp", ".lock"))]
    files += list(ROOT.glob("*.md")) + [p for p in (ROOT / "data").glob("*") if p.is_file()]
    manifest = {}
    with tarfile.open(dest / "saved_files.tar", "w") as tf:
        for p in sorted(set(files)):
            name = str(p.relative_to(ROOT))
            with p.open("rb") as f:
                manifest[name] = sha(f)
            tf.add(p, arcname=name, recursive=False)
    with tarfile.open(dest / "saved_files.tar", "r") as tf:
        mem = tf.getmembers()
        assert len(mem) == len(manifest)
        for m in mem:
            with tf.extractfile(m) as f:
                assert sha(f) == manifest[m.name], m.name
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for p in dest.iterdir():
        with p.open("rb") as f:
            os.fsync(f.fileno())
    os.sync()
    return str(dest), len(manifest)


def main():
    queue_pid, queue_start = int(sys.argv[1]), sys.argv[2]
    lock = (ROOT / "logs/phase1d_finish.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        log(f"armed; waiting for GPU queue pid {queue_pid}")
        while identity(queue_pid) == queue_start:
            if time.time() > DEADLINE:
                status["errors"].append("global deadline reached while the GPU queue was running; killed")
                subprocess.run(["pkill", "-f", "run_phase1d_gpu2.sh"])
                subprocess.run(["pkill", "-f", "11_replay.py|10_rollout.py|24_peek_states.py|25_w1_judge.py"])
                break
            if time.time() - newest_log_mtime() > STALE_MIN * 60:
                status["errors"].append(f"GPU queue logs stale for {STALE_MIN} min; killed as hung")
                subprocess.run(["pkill", "-f", "run_phase1d_gpu2.sh"])
                subprocess.run(["pkill", "-f", "11_replay.py|10_rollout.py|24_peek_states.py|25_w1_judge.py"])
                time.sleep(30)
                break
            time.sleep(60)
        status["queue_exit"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        prod = gpu_products()
        status["gpu_products_initial"] = prod
        save_status()
        for name in ("gold", "siblings", "peek_hidden", "peek_score", "w1_judge"):
            if not gpu_products()[name] and time.time() < DEADLINE:
                cmd, th, env = GPU_STEPS[name]
                run(f"gpu_{name}", cmd, th, env=env, retries=0)
        prod = gpu_products()
        status["gpu_products_final"] = prod
        save_status()
        if prod["siblings"]:
            run("features_G", ["28_features_G.py"], 1)
        if prod["peek_hidden"] and prod["peek_score"] and prod["w1_judge"]:
            run("features_W1_model", ["27_features_W1.py", "--part", "model"], 1)
        run("features_B_vllm", ["21_features_B.py", "--tag", "_vllm"], 1)
        b3 = []
        if prod["gold"] and run("gold", ["42_gold.py"], 2):
            g = json.loads((ROOT / "eval/gold.json").read_text())
            if g.get("B3_report_only_switch_fires"):
                b3 = ["--b3_report_only"]
        status["b3_report_only"] = bool(b3)
        env1 = {"OMP_NUM_THREADS": "1"}
        run("eval_main", ["40_eval.py", "--n_jobs", "16"] + b3, 6, env=env1)
        run("eval_main_vllm", ["40_eval.py", "--tag", "_vllm", "--n_jobs", "16"] + b3, 6, env=env1)
        run("eval_extras", ["41_eval_extras.py"], 3, env=env1)
        run("results", ["43_results.py"], 0.5)
        try:
            append_decisions()
        except Exception:
            status["errors"].append("append_decisions: " + traceback.format_exc()[-800:])
        try:
            status["git"] = git_commit()
        except Exception:
            status["errors"].append("git: " + traceback.format_exc()[-800:])
    except Exception:
        status["errors"].append("driver: " + traceback.format_exc()[-2000:])
    finally:
        try:
            status["archive"] = archive()
        except Exception:
            status["errors"].append("archive: " + traceback.format_exc()[-1500:])
        status["result"] = "complete" if not status["errors"] else "completed_with_errors_logs_saved"
        status["shutdown_requested_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save_status()
        os.sync()
        log(f"result={status['result']}; files synced; executing /usr/bin/shutdown")
        r = subprocess.run(["/bin/bash", "/usr/bin/shutdown"])
        log(f"shutdown returned {r.returncode}")


if __name__ == "__main__":
    main()
