"""Persist the shopping audit's preflight evidence; never starts GPU inference.

Exit 2 means required environment assets or model provenance are not ready.
Even exit 0 would only admit implementing/running the seven real pilot gates.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exploratory_shop_audit.data import (
    canonical_bytes, digest_file, load_native_data, sample_tasks,
    summarize_data, verify_tags, write_once,
)


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"argv": args, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": args, "error": str(error)}


def source_manifest(root):
    return {str(path.relative_to(root)): digest_file(path) for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".html", ".sh", ".txt", ".md"}}


def model_manifest(model, tree_path, revision_path, expected_revision):
    """Hash actual bytes and compare to saved upstream LFS/git-blob metadata."""
    import hashlib

    tree = json.loads(tree_path.read_text())
    revision = json.loads(revision_path.read_text())
    errors = []
    if revision.get("sha") != expected_revision:
        errors.append("Saved model revision metadata mismatch")
    files = {}
    for row in tree:
        name = row["path"]
        if name == ".gitattributes":
            continue
        path = model / name
        if not path.is_file():
            errors.append(f"Missing model file: {name}")
            continue
        sha = digest_file(path)
        if "lfs" in row:
            match = sha == row["lfs"]["oid"] and path.stat().st_size == row["size"]
        else:
            blob = path.read_bytes()
            git_sha = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
            match = git_sha == row["oid"] and len(blob) == row["size"]
        files[name] = {"sha256": sha, "bytes": path.stat().st_size,
                       "matches_saved_upstream_metadata": match}
        if not match:
            errors.append(f"Model bytes mismatch: {name}")
    return {"model_path": str(model.resolve()), "revision": revision.get("sha"),
            "revision_metadata_sha256": digest_file(revision_path),
            "tree_metadata_sha256": digest_file(tree_path), "files": files,
            "errors": errors, "loaded_for_inference": False}


def cross_process_snapshot_check(out, hashes):
    """Two fresh Python interpreters read the same saved tables (not engines)."""
    code = """
import json,sys
from pathlib import Path
from exploratory_shop_audit.data import load_verified,digest_value
root=Path(sys.argv[1]); hashes=json.loads(sys.argv[2])
tables={name:load_verified(root/name,checksum) for name,checksum in hashes.items()}
print(json.dumps({name:{'count':len(value),'canonical_sha256':digest_value(value)}
                  for name,value in tables.items()},sort_keys=True))
"""
    outputs = [command([sys.executable, "-c", code, str(out), json.dumps(hashes)])
               for _ in range(2)]
    passed = all(item.get("returncode") == 0 for item in outputs)
    passed = passed and outputs[0]["stdout"] == outputs[1]["stdout"]
    return {"passed": passed, "processes": outputs,
            "scope": "JSON snapshot readers only; independent engine restoration NOT_RUN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-tree", type=Path, required=True)
    parser.add_argument("--model-revision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text())
    out = args.out.resolve()
    if out.exists():
        parser.error("Output directory already exists; preserve it and choose a new preflight run")
    out.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    progress_path = out / "progress.jsonl"

    def progress(stage, **extra):
        with progress_path.open("ab") as stream:
            stream.write(canonical_bytes({"utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage, "gpu_hours": 0.0, "valid_episodes": 0, **extra}) + b"\n")

    progress("preflight_start", errors=0)
    # Archive and source hashes are saved even if later data/schema checks fail.
    provenance = {"started_utc": started, "generator": str(Path(__file__).relative_to(root)),
        "forkaudit_commit": command(["git", "rev-parse", "HEAD"]),
        "worktree_status": command(["git", "status", "--short"]),
        "generator_sources": source_manifest(Path(__file__).parent),
        "config_sha256": digest_file(config_path),
        "upstream_commit": config["upstream_commit"],
        "source_archive_sha256": digest_file(args.source_archive),
        "source_files": source_manifest(args.source_root),
        "data_path": str(args.data.resolve()), "data_sha256": digest_file(args.data),
        "gpu_query": command(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"]),
        "python": sys.version}
    packages = sorted((dist.metadata.get("Name", "unknown"), dist.version)
                      for dist in importlib.metadata.distributions())
    write_once(out / "dependencies.json", packages)
    provenance["dependencies_sha256"] = digest_file(out / "dependencies.json")
    write_once(out / "provenance.json", provenance)
    write_once(out / "config.json", config)
    blockers = []
    try:
        if provenance["data_sha256"] != config["data_sha256"]:
            raise ValueError("Data does not match the pinned raw gzip checksum")
        raw_bytes = gzip.decompress(args.data.read_bytes())
        raw = json.loads(raw_bytes)
        verify_tags(raw)
        product_json = out / "products_input.json"
        with product_json.open("xb") as stream:
            stream.write(raw_bytes)
        products, prices, goals, native_sources = load_native_data(
            args.source_root, product_json, config["master_seed"])
        summary = summarize_data(raw, products, goals)
        tasks = sample_tasks(products, goals, config)
        tasks["sources"] = {"raw_sha256": provenance["data_sha256"],
                            "native_initialization": native_sources}
        write_once(out / "data_audit.json", summary)
        write_once(out / "native_initialization.json", native_sources)
        write_once(out / "product_prices_frozen.json", prices)
        write_once(out / "goals_frozen.json", goals)
        write_once(out / "tasks.json", tasks)
        snapshots = {name: digest_file(out / name) for name in [
            "product_prices_frozen.json", "goals_frozen.json", "tasks.json"]}
        write_once(out / "snapshot_manifest.json", snapshots)
        cross = cross_process_snapshot_check(out, {key: value for key, value in snapshots.items()
                                                  if key != "tasks.json"})
        write_once(out / "snapshot_process_check.json", cross)
        if not cross["passed"]:
            blockers.append("Snapshot readers disagreed or failed")
        progress("native_data_initialization_end", errors=len(blockers),
                 goals=len(goals), pilot_tasks=len(tasks["pilot"]), formal_tasks=len(tasks["formal"]))
    except Exception as error:
        import traceback
        blockers.append(f"Native data precheck failed: {type(error).__name__}: {error}")
        write_once(out / "data_failure.json", {"error": blockers[-1], "traceback": traceback.format_exc()})
        summary = None

    required_search_paths = ["shop_env/search_engine/convert_product_file_format.py",
                             "shop_env/search_engine/run_indexing.sh", "shop_env/search_engine/indexes"]
    missing = [name for name in required_search_paths if not (args.source_root / name).exists()]
    if missing:
        blockers.append("Pinned source lacks native search assets: " + ", ".join(missing))
    try:
        model = model_manifest(args.model, args.model_tree, args.model_revision,
                               config["model"]["expected_revision"])
        write_once(out / "model_manifest.json", model)
        blockers.extend(model["errors"])
    except Exception as error:
        blockers.append(f"Model provenance failed: {type(error).__name__}: {error}")
    if provenance["gpu_query"].get("returncode") != 0:
        blockers.append("GPU unavailable to this process")
    status = "BLOCKED" if blockers else "PREFLIGHT_READY_PILOT_NOT_RUN"
    preflight = {"status": status, "blockers": blockers, "selected_tier": None,
        "data_summary": summary, "pilot_episodes": 0, "formal_episodes": 0,
        "fork_states": 0, "gpu_hours": 0.0,
        "runtime_gates": {key: "NOT_RUN" for key in [
            "20_stop_scores", "20_prefixes_two_engines", "state_isolation",
            "no_backend_leak", "termination_and_errors", "throughput_3_states", "truncation"]},
        "scientific_metrics": "NOT_RUN", "stop_rules": "EVIDENCE_INSUFFICIENT"}
    write_once(out / "preflight.json", preflight)
    write_once(out / "costs.json", {"allocated_gpu_hours": 0.0, "model_requests": 0,
        "max_gpu_hours": 16, "selected_tier": None,
        "note": "CPU provenance preparation only; no GPU model allocation."})
    progress("preflight_end", errors=len(blockers), status=status)
    report = ["# 本轮前检报告", "", f"状态：{status}。未运行任何策略轨迹或正式测量。", "",
              "本轮 GPU 占用 0 小时；试点／正式／分叉／S-F 均 NOT_RUN，档位尚未选择。", "",
              "## 阻塞项", "", *[f"- {value}" for value in blockers], "",
              "## 数据核查", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2),
              "```", "", "原始 eval 仅用于全库来源／划分检查，未送入策略或预测探针。",
              "两张表已保存且经过独立 Python 读取校验；这不等于两个真实引擎恢复通过。",
              "规格组合为笛卡尔计数，尚未验证点击可达性或计算 r_star。", "",
              "## 一页结论", "", "环境稳定性：原生检索资产缺失，未启动环境，不能判定。",
              "价格项：已核查数据生成路径并冻结两张表；策略可见性、收益影响尚未测量。",
              "历史截断、CTHA 缺口、停手覆盖／机会、回放信度与实际推理成本：均无数据。",
              "三条止损线：证据不足，均不作触发／未触发判定；不启动训练。", "",
              "## 下一步", "", "取得与固定版本匹配的原生索引和生成配方（含字段、中文分析器、排序配置与哈希），",
              "接通隔离适配后执行七项真实试点门，再根据吞吐冻结档位。",
              "不能替换为未登记检索器后声称原生环境门通过。"]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out": str(out), "blockers": blockers}, ensure_ascii=False))
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
