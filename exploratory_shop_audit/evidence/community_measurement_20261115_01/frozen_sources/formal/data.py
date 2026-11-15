"""CPU-only provenance, native data initialization, and outcome-blind sampling.

This module does not implement or emulate the shopping environment or its reward.
"""

import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stable_seed(*parts):
    # Domain-separated, unambiguous JSON; never Python's process-randomized hash.
    return int.from_bytes(hashlib.sha256(canonical_bytes(
        ["shop-audit-v1", *parts])).digest()[:8], "big") % (2**32)


def write_once(path, value):
    """Never overwrite or silently accept different frozen evidence."""
    path = Path(path)
    encoded = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(encoded)


def load_verified(path, expected_sha256):
    path = Path(path)
    if digest_file(path) != expected_sha256:
        raise ValueError(f"Snapshot checksum mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_tags(products):
    if not isinstance(products, list) or not products:
        raise ValueError("Product data must be a nonempty list")
    counts = Counter()
    group_tags = defaultdict(set)
    for product in products:
        tag = product.get("tag")
        if tag not in {"train", "eval"}:
            raise ValueError(f"Missing or unexpected tag for {product.get('asin')}")
        asin = product.get("asin")
        if not isinstance(asin, str) or not asin:
            raise ValueError("Product ID must be a nonempty string")
        if not isinstance(product.get("instructions"), list):
            raise ValueError(f"Missing instructions: {asin}")
        counts[tag] += 1
        group_tags[asin].add(tag)
    overlap = sorted(asin for asin, tags in group_tags.items() if len(tags) > 1)
    if overlap:
        raise ValueError(f"Target products overlap train/eval: {overlap[:10]}")
    return dict(counts)


def load_native_data(source_root, product_json, seed):
    """Execute only unmodified, hashed native data-loader function ASTs.

    No mocked get_reward, NLP, search, browser, or environment is supplied.
    This is explicitly NOT a full environment initialization acceptance test.
    """
    source_root = Path(source_root)
    selections = {
        "shop_env/web_agent_site/engine/engine.py": (
            "clean_product_keys", "generate_product_prices", "load_products"),
        "shop_env/web_agent_site/engine/goal.py": (
            "get_price_range_above", "get_existed_goals", "get_goals"),
    }
    namespace = {
        "random": random.Random(seed), "json": json, "math": math,
        "defaultdict": defaultdict,
        "print": lambda *args, **kwargs: None,
        "tqdm": lambda iterable, **kwargs: iterable,
    }
    source_evidence = {}
    for relative, names in selections.items():
        path = source_root / relative
        code = path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(path))
        functions = {node.name: node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names}
        if set(functions) != set(names):
            raise ValueError(f"Native loader API mismatch in {path}")
        selected = ast.Module(body=[functions[name] for name in names], type_ignores=[])
        exec(compile(selected, str(path), "exec"), namespace)
        source_evidence[relative] = {
            "sha256": digest_file(path),
            "functions": {name: hashlib.sha256(ast.get_source_segment(
                code, functions[name]).encode()).hexdigest() for name in names},
        }
    products, by_asin, prices, _ = namespace["load_products"](str(product_json))
    goals = namespace["get_goals"](products, prices, if_persona=False)
    if set(by_asin) != set(prices):
        raise ValueError("Product/price IDs differ")
    return products, prices, goals, source_evidence


def first_category(category):
    if not isinstance(category, str) or not category.strip():
        return "unknown"
    return category.split("›")[0].strip() or "unknown"


def allocate_strata(capacities, n):
    """Proportional largest-remainder quotas with stable category tie breaks."""
    total = sum(capacities.values())
    if not 0 <= n <= total:
        raise ValueError(f"Requested {n} tasks from {total} product groups")
    if n == 0:
        return {key: 0 for key in capacities}
    exact = {key: n * count / total for key, count in capacities.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    order = sorted(capacities, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:n - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def sample_tasks(products, goals, config):
    verify_tags(products)
    by_asin = {product["asin"]: product for product in products}
    if len(by_asin) != len(products):
        raise ValueError("Native products are not unique")
    seed = config["master_seed"]
    settings = config["sampling"]
    grouped = defaultdict(list)
    for index, goal in enumerate(goals):
        product = by_asin[goal["asin"]]
        if product["tag"] != "train":
            continue
        grouped[goal["asin"]].append({
            "task_id": f"goal:{index}", "goal_index": index,
            "target_asin": goal["asin"], "tag": "train",
            "category_original": first_category(goal.get("category")),
        })
    representatives = []
    for asin, tasks in grouped.items():
        representatives.append(min(tasks, key=lambda task: (
            stable_seed(seed, "within_product", task["task_id"]), task["task_id"])))
    category_counts = Counter(task["category_original"] for task in representatives)
    category_map = {
        category: (category if category == "unknown" or count >=
                   settings["small_category_min_products"] else "other_small")
        for category, count in category_counts.items()
    }
    pools = defaultdict(list)
    for task in representatives:
        task["category"] = category_map[task["category_original"]]
        pools[task["category"]].append(task)
    for rows in pools.values():
        rows.sort(key=lambda task: (
            stable_seed(seed, "product_order", task["target_asin"]), task["target_asin"]))

    def draw(n, phase, repetitions):
        quotas = allocate_strata({key: len(value) for key, value in pools.items()}, n)
        selected = []
        for category, quota in quotas.items():
            selected.extend(pools[category][:quota])
            del pools[category][:quota]
        selected.sort(key=lambda task: (
            stable_seed(seed, "task_order", phase, task["task_id"]), task["task_id"]))
        for task in selected:
            task["phase"] = phase
            task["trajectory_seeds"] = [stable_seed(
                seed, task["task_id"], repeat, phase) for repeat in range(repetitions)]
        return selected

    pilot = draw(settings["pilot_tasks"], "pilot", settings["pilot_repetitions"])
    formal = draw(settings["formal_tasks"], "formal", settings["formal_repetitions"])
    reserve = [task for rows in pools.values() for task in rows]
    reserve.sort(key=lambda task: (
        stable_seed(seed, "reserve_order", task["task_id"]), task["task_id"]))
    return {"master_seed": seed, "category_map": category_map,
            "train_product_groups": len(grouped), "pilot": pilot, "formal": formal,
            "reserve": reserve, "formal_low_budget_task_ids": [
                task["task_id"] for task in formal[:100]],
            "selected_tier": None,
            "note": "Task identities only; no fork states or model outcomes selected."}


def summarize_data(raw, products, goals):
    verify_tags(raw)
    tags = {product["asin"]: product["tag"] for product in products}
    raw_option_counts = [math.prod(len(values) for values in
        (product.get("customization_options") or {}).values() if values is not None)
        for product in raw]
    loaded_option_counts = [math.prod(len(values) for values in product["options"].values())
                            for product in products]
    empty_queries = sum(not product["query"] for product in products)
    native_price_weights = [1 / (len(goal["attributes"]) + len(goal["goal_options"]) + 1)
                            for goal in goals if tags[goal["asin"]] == "train"]
    return {
        "raw_product_count": len(raw), "raw_product_tags": dict(Counter(p["tag"] for p in raw)),
        "raw_instruction_tags": dict(Counter({tag: sum(len(p["instructions"]) for p in raw
            if p["tag"] == tag) for tag in ("train", "eval")})),
        "loaded_product_count": len(products),
        "loaded_product_tags": dict(Counter(p["tag"] for p in products)),
        "generated_goal_count": len(goals),
        "generated_goal_tags": dict(Counter(tags[g["asin"]] for g in goals)),
        "unique_target_products_by_tag": dict(Counter(tags[a] for a in {g["asin"] for g in goals})),
        "raw_missing_query_count": sum("query" not in p for p in raw),
        "loaded_empty_query_count": empty_queries,
        "all_query_match_true_by_native_construction": empty_queries == len(products),
        "raw_price_interval_count": sum(len(p.get("pricing") or []) >= 2 for p in raw),
        "raw_price_interval_fraction": sum(len(p.get("pricing") or []) >= 2 for p in raw) / len(raw),
        "max_raw_cartesian_option_count": max(raw_option_counts),
        "max_loaded_cartesian_option_count": max(loaded_option_counts),
        "option_count_caveat": "Cartesian count only; reachable click-map constraints and exact r_star NOT_RUN.",
        "train_price_term_denominator_weight_mean": sum(native_price_weights) / len(native_price_weights),
        "price_weight_caveat": "1/(m+o+1), not realized reward contribution or visibility.",
        "policy_eval_tasks_used": 0,
    }
