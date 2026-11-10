"""Fast CPU-only checks: python -m unittest discover -s exploratory_E4e -p 'test_e4e.py' -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import stage_logic as S
import funnel as F
import four_arm_eval as R


class FakePool:
    def __init__(self, initially_holding=False):
        self.moves = {}
        self.initially_holding = initially_holding

    def open(self, jobs):
        out = {}
        for key, *_ in jobs:
            self.moves[key] = 0
            initial = ("Your task is to: put a apple in fridge." +
                       (" You are carrying book 1." if self.initially_holding else ""))
            out[key] = (initial, ["put book 1 in drawer 1", "go to fridge 1"], False, 0)
        return out

    def step(self, actions):
        out = {}
        for key, a in actions.items():
            self.moves[key] += 1
            if a == "put book 1 in drawer 1":
                obs = "You move the book 1 to the drawer 1."
            else:
                obs = "Nothing happens."
            out[key] = (obs, ["go to fridge 1"], False, self.moves[key] >= 2)
        return out

    def close(self, keys):
        pass


class FakePolicy:
    def sample(self, prompts, seeds, logprobs=False):
        return [[{"text": "<think>go</think><action>go to fridge 1</action>"}] for _ in prompts]


class StageTests(unittest.TestCase):
    def test_completed_episode_survives_later_batch_failure(self):
        class FailingPool(FakePool):
            def step(self, actions):
                out = super().step(actions)
                if "3000:0" in actions:
                    obs, adm, _, _ = out["3000:0"]
                    out["3000:0"] = (obs, adm, True, True)
                else:
                    raise RuntimeError("simulated later failure")
                return out
        with tempfile.TemporaryDirectory() as root:
            games = [dict(seed=s, game_file="g", task_type="pick_and_place_simple") for s in [3000, 3001]]
            with self.assertRaisesRegex(RuntimeError, "simulated later failure"):
                R.run(games, 1, FakePolicy(), FailingPool(), root, "base", SimpleNamespace(ALFWORLD_DATA=""), {"plan": "a"})
            completed = Path(root, "s3000_r0.json")
            self.assertTrue(json.loads(completed.read_text())["won"])
            self.assertFalse(Path(root, "s3001_r0.json").exists())
            with self.assertRaisesRegex(RuntimeError, "different configuration"):
                R.run(games[:1], 1, FakePolicy(), FakePool(), root, "base", SimpleNamespace(ALFWORLD_DATA=""), {"plan": "b"})

    def test_tracking_and_legal_recovery(self):
        c = S.CarryState()
        self.assertFalse(c.known)
        c.update("You pick up the book 1 from the shelf 1.")
        self.assertEqual(c.item, "book 1")
        self.assertEqual(S.forced_release_action(c, "put a apple in fridge.", "pick_and_place_simple",
                         ["take apple 1 from table 1", "put book 1 in drawer 1"]), "put book 1 in drawer 1")
        self.assertIsNone(S.forced_release_action(c, "put a apple in fridge.", "pick_and_place_simple", ["look"]))
        c.update("Nothing happens.")
        self.assertEqual(c.item, "book 1")
        c.update("You move the book 1 to the drawer 1.")
        self.assertIsNone(c.item)
        self.assertTrue(c.known)

    def test_task_specific_milestones(self):
        steps = [dict(action="go to table 1", obs_after="You see apple 1 on table 1."),
                 dict(action="take apple 1 from table 1", obs_after="You pick up the apple 1 from the table 1."),
                 dict(action="heat apple 1 with microwave 1", obs_after="You heat the apple 1 using the microwave 1.")]
        ep = dict(task="heat a apple and put it in fridge.", task_type="pick_heat_then_place_in_recep",
                  initial_obs="nothing", steps=steps, won=False)
        m = S.milestones(ep)
        self.assertEqual(m["stages"], {"found": True, "picked": True, "processed": True, "completed": False})
        look = dict(task="look at a book under desklamp", task_type="look_at_obj_in_light", initial_obs="book 1", steps=[], won=False)
        self.assertEqual(list(S.milestones(look)["stages"]), ["found", "picked", "completed"])
        two = dict(task="put two apple in bowl", task_type="pick_two_obj_and_place", initial_obs="apple 1", steps=[], won=False)
        self.assertFalse(S.milestones(two)["stages"]["found"])
        two["steps"] = [dict(obs_after="You pick up the apple 1 from the table 1.")]
        self.assertFalse(S.milestones(two)["stages"]["found"])
        self.assertFalse(S.milestones(two)["stages"]["picked"])
        two["steps"].append(dict(obs_after="You see apple 2 on the table 2."))
        self.assertTrue(S.milestones(two)["stages"]["found"])
        self.assertFalse(S.milestones(two)["stages"]["picked"])
        two["won"] = True
        self.assertTrue(all(S.milestones(two)["stages"].values()))

    def test_funnel(self):
        ep = dict(task="put a apple in fridge.", task_type="pick_and_place_simple", initial_obs="apple 1", steps=[], won=False)
        ep2 = {**ep, "won": True, "seed": 1}
        out = F.funnel_for([ep, ep2])
        row = out["by_task_type"]["pick_and_place_simple"]
        self.assertEqual([s["passed"] for s in row["stages"]], [2, 1, 1])
        self.assertEqual(F.paired_ci([{"seed": 1, "won": False}], [ep2])["difference_pp"], 100)

    def test_prompt_and_run(self):
        c = S.CarryState(item="book 1", known=True)
        p = R.prompt_for("put a apple in fridge", [], "room", ["go to fridge 1"], c, "both")
        self.assertIn("currently holding book 1", p)
        self.assertNotIn("Carrying status:", R.prompt_for("put a apple in fridge", [], "room", ["look"], c, "base"))
        with tempfile.TemporaryDirectory() as root:
            games = [dict(seed=3000, game_file="g", task_type="pick_and_place_simple")]
            pool, pol = FakePool(), FakePolicy()
            R.run(games, 1, pol, pool, root, "both", SimpleNamespace(ALFWORLD_DATA=""), {})
            e = json.loads(Path(root, "s3000_r0.json").read_text())
            self.assertFalse(any(s["rule_triggered"] for s in e["steps"]))
            self.assertEqual(e["n_turns"], 2)
        with tempfile.TemporaryDirectory() as root:
            pool, pol = FakePool(initially_holding=True), FakePolicy()
            R.run(games, 1, pol, pool, root, "hard", SimpleNamespace(ALFWORLD_DATA=""), {})
            e = json.loads(Path(root, "s3000_r0.json").read_text())
            self.assertTrue(e["steps"][0]["rule_triggered"])
            self.assertEqual(e["steps"][0]["action"], "put book 1 in drawer 1")
            self.assertTrue(e["steps"][0]["overridden"])
            self.assertIsNone(e["steps"][0]["held_after"])


if __name__ == "__main__":
    unittest.main()
