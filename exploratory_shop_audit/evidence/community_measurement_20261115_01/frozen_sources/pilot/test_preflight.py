"""CPU preparation tests. These do not certify real ShopSimulator runtime gates."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from exploratory_shop_audit.data import (
    allocate_strata, digest_file, digest_value, first_category, load_verified,
    sample_tasks, stable_seed, verify_tags, write_once,
)


class PreflightTests(unittest.TestCase):
    def test_snapshot_cannot_be_overwritten_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prices.json"
            prices = {"商品A": 10.25, "商品B": 50.0}
            write_once(path, prices)
            checksum = digest_file(path)
            write_once(path, prices)
            self.assertEqual(load_verified(path, checksum), prices)
            with self.assertRaises(ValueError):
                write_once(path, {"商品A": 10.26})
            path.write_text('{"商品A":10.26}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_verified(path, checksum)

    def test_seed_stable_in_fresh_processes_with_different_python_hash_seeds(self):
        import os
        code = ("from exploratory_shop_audit.data import stable_seed; "
                "print(stable_seed(20261115,'中文任务',0,'formal'))")
        outputs = [subprocess.check_output([sys.executable, "-c", code],
                   env={**os.environ, "PYTHONHASHSEED": value}) for value in ("1", "999")]
        self.assertEqual(outputs[0], outputs[1])
        self.assertNotEqual(stable_seed(20261115, "point", "A", 0),
                            stable_seed(20261115, "point", "B", 0))
        self.assertNotEqual(stable_seed("ab", "c"), stable_seed("a", "bc"))

    def test_tags_fail_closed(self):
        product = {"asin": "id", "tag": "train", "instructions": []}
        self.assertEqual(verify_tags([product]), {"train": 1})
        for bad in (None, "test", "validation", "TRAIN"):
            with self.assertRaises(ValueError):
                verify_tags([{**product, "tag": bad}])
        with self.assertRaises(ValueError):
            verify_tags([product, {**product, "tag": "eval"}])

    def test_sampling_is_outcome_blind_group_disjoint_and_train_only(self):
        products = [{"asin": str(i), "tag": "train" if i < 30 else "eval",
                     "instructions": [{}, {}]} for i in range(35)]
        goals = [{"asin": str(i), "category": ("家居›床" if i % 2 else "服饰›衣")}
                 for i in range(35) for _ in range(2)]
        config = {"master_seed": 20261115, "sampling": {
            "pilot_tasks": 4, "formal_tasks": 10, "pilot_repetitions": 2,
            "formal_repetitions": 4, "small_category_min_products": 2}}
        before = copy.deepcopy((products, goals, config))
        result = sample_tasks(products, goals, config)
        self.assertEqual(result, sample_tasks(products, goals, config))
        self.assertEqual((products, goals, config), before)
        sets = [{task["target_asin"] for task in result[phase]}
                for phase in ("pilot", "formal", "reserve")]
        self.assertEqual([len(s) for s in sets], [4, 10, 16])
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertTrue(sets[0].isdisjoint(sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))
        self.assertEqual(set.union(*sets), {str(i) for i in range(30)})
        for task in result["formal"]:
            self.assertEqual(len(set(task["trajectory_seeds"])), 4)
        self.assertIsNone(result["selected_tier"])

    def test_strata_quota_and_unknown(self):
        self.assertEqual(allocate_strata({"A": 80, "B": 20}, 20), {"A": 16, "B": 4})
        self.assertEqual(allocate_strata({"B": 1, "A": 1}, 1), {"B": 0, "A": 1})
        self.assertEqual(first_category(None), "unknown")
        self.assertEqual(first_category("家居›床›枕"), "家居")
        with self.assertRaises(ValueError):
            allocate_strata({"A": 1}, 2)

    def test_native_functions_are_extracted_without_import_side_effects(self):
        from exploratory_shop_audit.data import load_native_data
        # A fixture verifies execution mechanics only, not real native scoring.
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            engine = source / "shop_env/web_agent_site/engine"
            engine.mkdir(parents=True)
            (engine / "engine.py").write_text('''
raise RuntimeError("module imports must not execute")
def clean_product_keys(products):
    return products
def generate_product_prices(products):
    return {p['asin']: random.uniform(10,20) for p in products}
def load_products(filepath):
    with open(filepath) as f:
        p=json.load(f)
    return p, {x['asin']:x for x in p}, generate_product_prices(p), {}
''')
            (engine / "goal.py").write_text('''
raise RuntimeError("NLP module imports must not execute")
def get_price_range_above(price):
    return [price+1,price+2]
def get_existed_goals(products,prices,if_persona=False):
    return [{'asin':p['asin'],'price_upper':prices[p['asin']]+random.random()} for p in products]
def get_goals(products,prices,if_persona=False):
    return get_existed_goals(products,prices,if_persona)
''')
            data = source / "products.json"
            data.write_text('[{"asin":"a"}]')
            first = load_native_data(source, data, 20261115)
            second = load_native_data(source, data, 20261115)
            self.assertEqual(first, second)
            self.assertGreater(first[2][0]["price_upper"], first[1]["a"])
            self.assertEqual(len(first[3]), 2)

    def test_cross_process_reads_are_explicitly_not_engine_validation(self):
        from exploratory_shop_audit.preflight import cross_process_snapshot_check
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tables = {"product_prices_frozen.json": {"x": 10.5},
                      "goals_frozen.json": [{"asin": "x", "price_upper": 20}]}
            hashes = {}
            for name, table in tables.items():
                write_once(root / name, table)
                hashes[name] = digest_file(root / name)
            check = cross_process_snapshot_check(root, hashes)
            self.assertTrue(check["passed"])
            self.assertIn("NOT_RUN", check["scope"])
            parsed = json.loads(check["processes"][0]["stdout"])
            self.assertEqual(parsed["goals_frozen.json"]["canonical_sha256"],
                             digest_value(tables["goals_frozen.json"]))


if __name__ == "__main__":
    unittest.main()
