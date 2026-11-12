"""CTHA v2 CPU tests: synthesized examples, never claims of real training gains."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_ctha import fake_episode, cohort
import features_v2 as F
import habits_v2 as H
import prepare
import p0_audit


def dataset(n_types=4, n_games_per_type=6, steps=3):
    episodes = []
    for kind in range(n_types):
        for game_id in range(n_games_per_type):
            index = kind * n_games_per_type + game_id
            for rep in range(4):
                win = rep in (0, 1)
                e = fake_episode(index, rep, win, iteration=0)
                e["task_type"] = f"task-type-{kind}"
                e["steps"] = []
                for turn in range(steps):
                    # Clearly predictive, student-observable action. No hidden-state input.
                    a = "go to drawer 1" if win else "take cup 1 from table 1"
                    e["steps"].append(dict(
                        turn=turn + 1, action=a,
                        response=f"<think>ok</think><action>{a}</action>",
                        response_token_ids=[20, 33, 17], obs_before="You are at kitchen.",
                        obs_after="Only the outcome of the action.", admissible=[
                            "go to drawer 1", "take cup 1 from table 1",
                            "open cabinet 1", "examine sink 1"],
                        prompt="Your task is to: find an apple and put it in a fridge. "
                               "Action 1: 'go to table 1'"))
                episodes.append(e)
    return episodes


class TestFeatures(unittest.TestCase):
    def test_no_future_outcome_in_generic_or_progress_features(self):
        e = dataset(3, 1, 2)[0]
        step = e["steps"][0]
        for family in ("habit", "progress", "both", "action", "manual"):
            before = F.step_features(step, e, family)
            changed = copy.deepcopy(e)
            changed["won"] = not e["won"]
            changed["steps"][0]["obs_after"] = "You successfully reach the goal"
            changed["steps"][0]["response"] = "Different verbal reasoning"
            after = F.step_features(changed["steps"][0], changed, family)
            np.testing.assert_array_equal(before, after)

    def test_progress_is_separable_and_dimensions_stable(self):
        e = dataset(3, 1)[0]
        st = e["steps"][0]
        habit = F.step_features(st, e, "habit")
        prog = F.step_features(st, e, "progress")
        both = F.step_features(st, e, "both")
        self.assertEqual(len(habit), len(F.feature_names("habit")))
        self.assertEqual(len(prog), len(F.feature_names("progress")))
        np.testing.assert_array_equal(both, np.concatenate([habit, prog]))
        # Candidate action changes but cannot change current observation in the input.
        alternate = F.with_action(st, "take cup 1 from table 1")
        self.assertEqual(alternate["obs_before"], st["obs_before"])
        self.assertNotEqual(float(F.step_features(alternate, e, "habit") @ habit),
                            float(habit @ habit))

    def test_prefix_is_mean_rate_and_not_episode_length(self):
        e = dataset(3, 1, 3)[0]
        a = F.prefix_feature_rate(e, 8)
        e["steps"] = e["steps"][:1]
        b = F.prefix_feature_rate(e, 8)
        np.testing.assert_allclose(a, b)


class TestStatisticalGate(unittest.TestCase):
    def test_task_type_is_excluded_from_its_own_beta(self):
        ep = dataset()
        b1, _ = H.type_heldout_betas(ep, family="habit", window=2, alpha=2.,
                                     min_mixed=2, stable_sign=0)
        copy_ep = copy.deepcopy(ep)
        for e in copy_ep:
            if e["task_type"] == "task-type-0":
                e["won"] = not e["won"]
        b2, _ = H.type_heldout_betas(copy_ep, family="habit", window=2, alpha=2.,
                                     min_mixed=2, stable_sign=0)
        np.testing.assert_array_equal(b1["task-type-0"], b2["task-type-0"])

    def test_permutation_control_and_holdout_training_sanity(self):
        ep = dataset()
        stats = H.oof_audit(ep, family="habit", window=2, alpha=2.,
                            min_mixed=8, min_pairs=25,
                            permutations=3, bootstraps=60, seed=12)
        self.assertTrue(stats["accuracy"] > 0.60)
        self.assertTrue(stats["total_pairs"] >= 25)
        self.assertEqual(stats["heldout_games"], 24)
        self.assertEqual(len(stats["folds"]), 4)
        self.assertTrue(np.isfinite(stats["null_accuracy"]))

    def test_gate_rejects_no_mixed_or_no_task_types(self):
        ep = dataset(2, 2)
        stats = H.oof_audit(ep, family="habit", window=1, min_mixed=8,
                            min_pairs=20, permutations=1, bootstraps=10)
        self.assertFalse(stats["passed"])
        self.assertEqual(stats["reason"], "warmup_or_insufficient_task_types")

    def test_progress_comparison_reports_pair_accuracy(self):
        ep = dataset()
        out = p0_audit.accuracy_gap(ep, window=2, alpha=2., bootstraps=40)
        self.assertEqual(out["games"], 24)
        self.assertTrue(np.isfinite(out["delta"]))


class TestOnlineAdvantage(unittest.TestCase):
    def test_gigpo_anchor_groups_are_task_local(self):
        ep = dataset(3, 2, 2)
        gig, failed, diag = H.action_base_advantages(ep, base="gigpo", local_weight=1)
        grpo, _, d2 = H.action_base_advantages(ep, base="grpo")
        self.assertGreater(diag["n_repeated_anchors"], 0)
        first = ep[0]
        second = ep[2]
        self.assertGreater(gig[(first["episode_id"], 0)], gig[(second["episode_id"], 0)])
        self.assertTrue(np.isfinite(np.asarray(list(gig.values()))).all())
        self.assertTrue(np.isfinite(np.asarray(list(grpo.values()))).all())

    def test_all_failed_task_has_no_base_advantage(self):
        ep = dataset(3, 2, 2)
        game = ep[0]["game_file"]
        for e in ep:
            if e["game_file"] == game:
                e["won"] = False
        ga, failed, _ = H.action_base_advantages(ep, base="gigpo")
        self.assertTrue(failed[game])
        self.assertTrue(all(ga[(e["episode_id"], 0)] == 0 for e in ep
                            if e["game_file"] == game))

    def test_full_rollout_updates_and_prefix_only_aux(self):
        ep = dataset(4, 2, 3)
        rows, info = H.make_batch(
            ep, [], mode="fixed", family="habit", base="gigpo",
            feature_window=1, train_horizon=0, aux_horizon=1)
        self.assertEqual(len(rows), len(ep)*3)
        self.assertTrue(info["full_episode_policy_updates"])
        self.assertEqual(info["base"], "gigpo")
        self.assertTrue(all(r["advantage_habit"] == 0 for r in rows if r["turn"] > 1))
        # No training token (including the third step) gets dropped.
        self.assertEqual(sum(r["turn"] == 3 for r in rows), len(ep))

    def test_annealing_is_nonnegative_and_monotone(self):
        vals = [prepare.effective_omega(.3, k, 4, 12) for k in range(14)]
        self.assertAlmostEqual(vals[0], .3)
        self.assertAlmostEqual(vals[4], .3)
        self.assertAlmostEqual(vals[12], 0)
        self.assertTrue(all(x >= y for x, y in zip(vals, vals[1:])))


if __name__ == "__main__":
    unittest.main()
