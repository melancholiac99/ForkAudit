"""Real, unmodified community environment smoke; never formal data collection.

Three fixed TRAIN tasks, visible first-result/first-option scripted purchases,
then independent-process prefix restoration and actual terminal execution.
No policy training, no legacy reward patch, no synthesized query metadata.
"""

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import canonical_bytes, digest_file, digest_value, write_once


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((plain(item) for item in value), key=canonical_bytes)
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def state(env):
    session = env.server.user_sessions[env.session]
    tracker = session.get("progress_tracker")
    return {
        "observation": env.observation,
        "available": env.get_available_actions(),
        "public_structured_observation": env.structured_observation(),
        "current_url": env.browser.current_url,
        "session_backend": plain(session),
        "remaining_steps": env.server.max_steps - tracker.steps,
        "previous_actions": list(env.prev_actions),
    }


def instantiate(args):
    os.environ["SHOP_SEARCH_INDEX"] = str(args.index.resolve())
    sys.path.insert(0, str(args.shop_env.resolve()))
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env = WebAgentTextEnv(observation_mode="text", file_path=str(args.products.resolve()),
                          split="train", if_persona=False, shuffle_goals=False,
                          shift_goals=False)
    # Guard the actual executed path, not a patched legacy score implementation.
    import inspect
    done_source = inspect.getsource(type(env.server).done)
    if "evaluate_purchase(" not in done_source or "get_reward(" in done_source:
        raise RuntimeError("Unexpected reward call path; refuse legacy empty-query scoring")
    return env


def reset(env, index):
    # Reset must clear terminal bookkeeping left by a previous same-ID session.
    env.server.user_sessions.pop(index, None)
    env.reset(idx=index)
    return state(env)


def execute(env, action):
    before = state(env)
    observation, status, info = env.step(action)
    after = state(env)
    if observation != after["observation"]:
        raise RuntimeError("Unexpected built-in history composition")
    return {"action": action, "before": before, "after": after,
            "status": plain(status), "info": plain(info)}


def first_visible_option(env, axis):
    # Read exactly the radio controls offered by the environment.
    for value, element in env.text_to_clickable.items():
        if element.get("name") == axis:
            return value
    raise RuntimeError(f"No clickable option on axis {axis!r}")


def collect(args):
    env = instantiate(args)
    prices_hash = digest_value(env.server.product_prices)
    goals_hash = digest_value(env.server.goals)
    write_once(args.out / "community_prices.json", env.server.product_prices)
    write_once(args.out / "community_goals.json", env.server.goals)
    write_once(args.out / "tables.json", {"prices": prices_hash, "goals": goals_hash})
    tasks = json.loads(args.tasks.read_text())["pilot"][:3]
    episodes = []
    for task in tasks:
        index = task["goal_index"]
        goal = env.server.goals[index]
        product = env.server.product_item_dict[goal["asin"]]
        if goal["asin"] != task["target_asin"] or product["tag"] != "train":
            raise RuntimeError("Task mapping or TRAIN tag changed")
        episode = {"task_id": task["task_id"], "goal_index": index,
                   "controller": "scripted_first_visible_product_first_option",
                   "is_policy_rollout": False, "steps": []}
        initial = reset(env, index)
        episode["initial"] = initial
        # Only use the public task instruction as the search string.
        search = execute(env, f"search[{env.instruction_text}]")
        episode["steps"].append(search)
        products = search["after"]["public_structured_observation"].get("products", [])
        if not products:
            raise RuntimeError(f"No visible search result for {task['task_id']}")
        first = products[0]["asin"]
        episode["steps"].append(execute(env, f"click[{first}]"))
        axes = list(env.structured_observation().get("available_options", {}))
        if not axes:
            raise RuntimeError("First visible item has no options; retain failure, do not cherry-pick")
        for axis in axes:
            option = first_visible_option(env, axis)
            episode["steps"].append(execute(env, f"click[{option}]"))
            if episode["steps"][-1]["status"]["done"]:
                raise RuntimeError("Terminated before purchase")
        episode["purchase_state"] = state(env)
        item_hash = digest_value(env.server.product_item_dict[first])
        episode["steps"].append(execute(env, "click[buy now]"))
        terminal = episode["steps"][-1]
        if not terminal["status"]["done"] or not terminal["status"].get("purchase"):
            raise RuntimeError("Buy did not produce a completed purchase")
        if digest_value(env.server.product_item_dict[first]) != item_hash:
            raise RuntimeError("Purchase mutated shared product object")
        episode["reward"] = terminal["status"]["reward"]
        episode["reward_type"] = terminal["status"].get("reward_detail", {}).get("reward_type")
        episode["purchase_product_unchanged"] = True
        # Test reset after an actual purchase (separate from independent replay).
        episode["reset_after_purchase_equal"] = reset(env, index) == initial
        if not episode["reset_after_purchase_equal"]:
            raise RuntimeError("Post-purchase reset diverged")
        write_once(args.out / f"episode_{index}.json", episode)
        episodes.append({"goal_index": index, "file": f"episode_{index}.json",
                         "actions": [step["action"] for step in episode["steps"]],
                         "reward": episode["reward"], "reward_type": episode["reward_type"]})
    if digest_value(env.server.product_prices) != prices_hash or digest_value(env.server.goals) != goals_hash:
        raise RuntimeError("Price/goal tables mutated during smoke")
    write_once(args.out / "episodes_manifest.json", episodes)
    env.server.search_engine.close()


def replay(args):
    env = instantiate(args)
    tables = json.loads((args.out / "tables.json").read_text())
    if digest_value(env.server.product_prices) != tables["prices"] or digest_value(env.server.goals) != tables["goals"]:
        raise RuntimeError("Independent engine price/goal tables differ")
    results = []
    for item in json.loads((args.out / "episodes_manifest.json").read_text()):
        original = json.loads((args.out / item["file"]).read_text())
        got_initial = reset(env, item["goal_index"])
        if got_initial != original["initial"]:
            raise RuntimeError("Independent reset observation/state mismatch")
        comparisons = []
        # Each decision prefix is rebuilt from reset, preserving original budget.
        for offset, expected in enumerate(original["steps"]):
            reset(env, item["goal_index"])
            for prefix in original["steps"][:offset]:
                execute(env, prefix["action"])
            actual = execute(env, expected["action"])
            equal = actual == expected
            comparisons.append({"prefix_actions": offset, "state_equal": actual["before"] == expected["before"],
                                "after_equal": actual["after"] == expected["after"],
                                "status_equal": actual["status"] == expected["status"], "equal": equal})
            if not equal:
                write_once(args.out / f"divergence_{item['goal_index']}_{offset}.json",
                           {"expected": expected, "actual": actual})
                raise RuntimeError(f"Prefix divergence at {item['goal_index']}:{offset}")
        results.append({"goal_index": item["goal_index"], "prefix_checks": comparisons,
                        "actual_purchase_reward": actual["status"]["reward"],
                        "purchase_equal": actual["status"] == original["steps"][-1]["status"]})
    write_once(args.out / "independent_replay.json", {
        "process_id": os.getpid(), "tables_equal": True, "episodes": results,
        "all_equal": True, "scope": "three scripted TRAIN episodes; not 20-task pilot acceptance"})
    env.server.search_engine.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shop-env", type=Path, required=True)
    p.add_argument("--products", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", choices=["collect", "replay"])
    args = p.parse_args()
    if args.worker:
        try:
            (collect if args.worker == "collect" else replay)(args)
            return 0
        except Exception:
            write_once(args.out / f"{args.worker}_failure.json", {"traceback": traceback.format_exc()})
            raise
    if args.out.exists():
        p.error("Use a new output directory; failed evidence must remain intact")
    args.out.mkdir(parents=True)
    config = args.shop_env / "configs/environment.json"
    write_once(args.out / "registration.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "community_commit": "13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5",
        "formal_collection": False, "policy_model_requests": 0, "gpu_hours": 0,
        "script_sha256": digest_file(__file__), "tasks_sha256": digest_file(args.tasks),
        "products_sha256": digest_file(args.products), "index_sha256": digest_file(args.index),
        "config_sha256": digest_file(config), "config": json.loads(config.read_text()),
        "source_hashes": {str(path.relative_to(args.shop_env)): digest_file(path)
            for path in sorted(args.shop_env.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".html", ".json"}},
        "python": sys.version, "dependencies": sorted((dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions())})
    base = [sys.executable, str(Path(__file__).resolve()), "--shop-env", str(args.shop_env.resolve()),
            "--products", str(args.products.resolve()), "--index", str(args.index.resolve()),
            "--tasks", str(args.tasks.resolve()), "--out", str(args.out.resolve())]
    phases = []
    for phase in ("collect", "replay"):
        start = time.monotonic()
        with (args.out / f"{phase}.log").open("w") as log:
            result = subprocess.run([*base, "--worker", phase], stdout=log, stderr=subprocess.STDOUT)
        phases.append({"phase": phase, "returncode": result.returncode,
                       "wall_seconds": time.monotonic() - start})
        write_once(args.out / f"{phase}_exit.json", phases[-1])
        if result.returncode:
            print(json.dumps({"status": "FAILED", "phase": phase, "out": str(args.out)}))
            return result.returncode
    write_once(args.out / "result.json", {"status": "PASS", "phases": phases,
        "formal_collection": False, "gpu_hours": 0, "policy_model_requests": 0})
    print(json.dumps({"status": "PASS", "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
