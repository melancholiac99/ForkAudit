import ast
import unittest
from small_common import HERE, OLD, PILOT, load


class SmallScopeTests(unittest.TestCase):
    def test_actor_training_branch_is_unchanged(self):
        def branch(path):
            tree = ast.parse(path.read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            collect = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "a.action == 'collect'")
            return [ast.dump(n) for n in collect.orelse[0].orelse]
        self.assertEqual(branch(HERE / 'worker.py'), branch(OLD / 'pilot_worker.py'))

    def test_four_tasks_and_four_repetitions(self):
        games = load(PILOT / 'train_games.json')
        self.assertEqual({g['seed'] for g in games}, {1004, 1028, 1150, 1065})
        for folder in ['base_train_eval', 'shared_initial_cache']:
            paths = list((PILOT / folder).glob('s*_r*.json'))
            self.assertEqual(len(paths), 16)
            expected = {(g['seed'], r) for g in games for r in range(4)}
            self.assertEqual({(load(p)['seed'], load(p)['rep']) for p in paths}, expected)

    def test_original_optimizer_and_thresholds_are_retained(self):
        old, new = load(OLD / 'PLAN.json'), load(HERE / 'PLAN.json')
        for key in ['optimizer', 'parity_gate', 'stability']:
            self.assertEqual(old[key], new[key])
        self.assertEqual(new['pilot']['rounds'], 4)
        self.assertEqual(new['pilot']['reps'], 4)
        self.assertEqual(new['pilot']['tasks'], 4)


if __name__ == '__main__':
    unittest.main()
