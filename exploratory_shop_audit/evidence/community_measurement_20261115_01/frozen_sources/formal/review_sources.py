"""Recompute the environment source/data comparison from saved downloads only."""

import argparse
import ast
from collections import Counter
import difflib
import gzip
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import digest_file, write_once


def parser_from(path):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "parse_action")
    scope = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["parse_action"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--community-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh report directory")
    args.out.mkdir(parents=True)
    original = args.original_root / "shop_env"
    community = args.community_root / "environments/ShopSimulator/shop_env"
    gzip_name = "data/fine_items_eval_train_all.json.gz"
    original_data = json.loads(gzip.decompress((original / gzip_name).read_bytes()))
    community_data = json.loads(gzip.decompress((community / gzip_name).read_bytes()))
    by_asin = {product["asin"]: product for product in original_data}
    supplementary = Counter()
    for line in (args.review_root / "official_train_standard.jsonl").open():
        product = json.loads(line)
        supplementary["rows"] += 1
        supplementary[f"tag_{product.get('tag')}"] += 1
        supplementary["query_present"] += "query" in product
        supplementary["identical_to_original_bundle"] += by_asin.get(product["asin"]) == product

    engine = "web_agent_site/engine/engine.py"
    native_parse, community_parse = parser_from(original / engine), parser_from(community / engine)
    counts = Counter()
    failures = []
    for product in original_data:
        for axis, contents in (product.get("customization_options") or {}).items():
            for option in contents or []:
                value = option["value"].strip().replace("/", " | ").lower()
                expected = ("click", value)
                action = f"click[{value}]"
                native, other = native_parse(action), community_parse(action)
                counts["option_values"] += 1
                counts["native_failures"] += native != expected
                counts["community_failures"] += other != expected
                counts["parsers_disagree"] += native != other
                if other != expected:
                    counts["failures_with_bracket"] += "[" in value
                    failures.append({"asin": product["asin"], "tag": product["tag"],
                        "axis": axis, "value": value, "action": action,
                        "native_parsed": native, "community_parsed": other})
    write_once(args.out / "action_parser_failures.json", failures)

    files = [engine, "web_agent_site/engine/goal.py", "web_agent_site/envs/web_agent_text_env.py",
             "web_agent_site/templates/item_page.html", "shop_env/shop_agent.py", "shop_env/pack_api.py"]
    diff_manifest = {}
    for name in files:
        left, right = original / name, community / name
        if left.exists() and right.exists():
            target = args.out / (name.replace("/", "__") + ".diff")
            target.write_text("".join(difflib.unified_diff(
                left.read_text().splitlines(keepends=True), right.read_text().splitlines(keepends=True),
                fromfile=f"original/{name}", tofile=f"community/{name}")))
            diff_manifest[name] = {"original_sha256": digest_file(left),
                                   "community_sha256": digest_file(right), "diff": target.name}
    download_files = ["community.tar.gz", "community_commit.json", "official_train_standard.jsonl",
                      "official_dataset_revision.json", "official_dataset_tree_pinned.json",
                      "official_issues.json", "products.sqlite3", "products.manifest.json"]
    provenance = {name: digest_file(args.review_root / name) for name in download_files
                  if (args.review_root / name).is_file()}
    pinned = args.review_root / "official_dataset_tree_pinned.json"
    supplemental_pin_verified = False
    if pinned.exists():
        row = next(row for row in json.loads(pinned.read_text())
                   if row["path"] == "fine_items_train_standard.jsonl")
        supplemental_pin_verified = row.get("lfs", {}).get("oid") == provenance["official_train_standard.jsonl"]
    result = {
        "original_commit": "475837e7a51737c44c0a9f79f9459be1e79060fa",
        "community_commit": "13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5",
        "official_dataset_revision": "ab34b2898059b21e09794de17e61684f1baeb50d",
        "official_supplement_matches_pinned_lfs": supplemental_pin_verified,
        "official_supplement": dict(supplementary), "source_diff_manifest": diff_manifest,
        "bundled_data_identical": original_data == community_data,
        "original_bundle_sha256": digest_file(original / gzip_name),
        "community_bundle_sha256": digest_file(community / gzip_name),
        "products": len(original_data), "query_present": sum("query" in p for p in original_data),
        "original_query_reconstruction_available": False,
        "query_finding": "WebShop loader expects pre-existing product query metadata; checked ShopSimulator bundles and official training supplement lack it. No verified mapping from domain/category/instruction exists in checked sources.",
        "action_parser_audit": dict(counts), "downloads": provenance,
        "script_sha256": digest_file(__file__), "formal_collection": False,
    }
    write_once(args.out / "source_review.json", result)
    print(json.dumps({key: result[key] for key in ["official_supplement_matches_pinned_lfs",
        "official_supplement", "bundled_data_identical", "action_parser_audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
