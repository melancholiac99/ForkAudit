import ast
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class BF16ProtocolTests(unittest.TestCase):
    def test_optimizer_loop_and_loss_are_unchanged(self):
        def loop(path):
            tree = ast.parse(path.read_text())
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
            return ast.dump(next(n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch'))
        self.assertEqual(loop(HERE / 'trainer.py'), loop(HERE.parent / 'a100_full_response/trainer.py'))

    def test_original_probability_and_stability_limits_remain(self):
        p = json.loads((HERE / 'PLAN.json').read_text())
        self.assertEqual(p['gates'], dict(mean_abs=.01, p99_abs=.05, max_abs=.25,
            sampled_behavior_k3=.02, clip_fraction=.30))
        self.assertFalse(p['adoption'])


if __name__ == '__main__':
    unittest.main()
