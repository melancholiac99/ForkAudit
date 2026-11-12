"""CPU-only regression tests. Synthetic episodes are NOT Agent benchmark results."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from habits import (FEATURES, observable_features, task_target, episode_vector,
                    centered_design, ridge, group_returns, make_batch, load_episodes)
import prepare
import run
import rollout
import train


def fake_episode(game: int, rep: int, won: bool, iteration=0, action=None) -> dict:
    if action is None:
        action = "go to drawer 1" if won else "take cup 1 from table 1"
    actions = ["go to drawer 1", "take cup 1 from table 1", "examine sink 1"]
    prompt = ("Your task is to: find an apple and put it in a fridge. "
              "Action 1: 'go to table 1'")
    step = dict(turn=1, prompt=prompt, action=action, admissible=actions,
                obs_before="At a kitchen", obs_after="Nothing happens.",
                response="<think>test</think><action>" + action + "</action>",
                response_token_ids=[25, 14, 31], valid=True)
    return dict(game_file=f"game-{game}", task_type=f"type-{game%3}", seed=1000+game, rep=rep,
                task="find an apple and put it in a fridge", won=bool(won),
                steps=[step], iteration=iteration,
                episode_id=f"{iteration}:{1000+game}:{rep}",
                provenance=dict(split="train", iteration=iteration, dry=False))


def cohort(n=12, iteration=0, all_fail_game=0):
    episodes = []
    for game in range(n):
        for rep in range(4):
            success = (rep < 2) if game != all_fail_game else False
            episodes.append(fake_episode(game, rep, success, iteration))
    return episodes


class HabitTests(unittest.TestCase):
    def test_student_visible_features_and_goal_matching(self):
        self.assertEqual(task_target("find an apple and put it in a fridge"), "apple")
        actions = ["take apple 1 from shelf 1", "take cup 1 from table 1", "go to table 1"]
        good = observable_features("Action 1: 'go to table 1'", actions[0], actions, "find an apple and put it in a fridge")
        bad = observable_features("Action 1: 'go to table 1'", actions[1], actions, "find an apple and put it in a fridge")
        goto = observable_features("Action 1: 'go to table 1'", actions[2], actions, "find an apple and put it in a fridge")
        self.assertEqual(good[FEATURES.index("take_target")], 1)
        self.assertEqual(bad[FEATURES.index("take_non_target")], 1)
        self.assertEqual(goto[FEATURES.index("goto_recent")], 1)
        self.assertEqual(goto[FEATURES.index("goto_outside_recent")], 0)

    def test_centered_ridge_learns_negative_wrong_take(self):
        episodes = cohort(16)
        beta, diag = ridge(episodes, horizon=1, alpha=1, min_mixed=4)
        self.assertTrue(diag["active"])
        self.assertLess(beta[FEATURES.index("take_non_target")], 0)
        self.assertGreater(beta[FEATURES.index("goto_outside_recent")], 0)

    def test_grpo_zero_for_all_failed(self):
        eps = cohort()
        a = group_returns(eps)
        self.assertTrue(all(a[e["episode_id"]] == 0 for e in eps if e["game_file"] == "game-0"))
        self.assertAlmostEqual(sum(a[e["episode_id"]] for e in eps if e["game_file"] == "game-1"), 0, places=7)

    def test_no_focal_game_leakage_to_habit_coefficients(self):
        eps = cohort(12)
        rows, report = make_batch(eps, [], mode="ctha_nogate", min_mixed=3, horizon=1)
        self.assertGreater(report["nonzero_aux_samples"], 0)
        self.assertTrue(any(abs(x["advantage_habit"]) > 0 and x["group_all_failed"] for x in rows))
        # Change labels of one game but keep its prompts/actions identical.
        modified = copy.deepcopy(eps)
        for e in modified:
            if e["game_file"] == "game-0":
                e["won"] = not e["won"]
        _, changed = make_batch(modified, [], mode="ctha_nogate", min_mixed=3, horizon=1)
        self.assertEqual(report["beta_by_game"]["game-0"], changed["beta_by_game"]["game-0"])

    def test_habit_gate_and_nonfail_groups(self):
        eps = cohort(12)
        rows, report = make_batch(eps, [], mode="ctha", min_mixed=2, horizon=1)
        self.assertGreater(report["mixed_groups_current"], 0)
        self.assertTrue(all(x["advantage_habit"] == 0
                            for x in rows if not x["group_all_failed"]))
        grpo_rows, _ = make_batch(eps, [], mode="grpo", min_mixed=2, horizon=1)
        self.assertTrue(all(x["advantage_habit"] == 0 for x in grpo_rows))
        self.assertTrue(all(np.isfinite(x["advantage"]) for x in rows))

    def test_all_failed_without_success_signal_learns_nothing(self):
        eps = [fake_episode(game, rep, False) for game in range(6) for rep in range(4)]
        rows, diag = make_batch(eps, [], mode="ctha", horizon=1, min_mixed=2)
        self.assertEqual(diag["nonzero_aux_samples"], 0)
        self.assertTrue(all(x["advantage"] == 0 for x in rows))

    def test_grouping_by_iteration_avoids_mixed_policies(self):
        ep = cohort(5)
        ep2 = [fake_episode(i, j, True, iteration=1) for i in range(5) for j in range(4)]
        X, y, n = centered_design(ep + ep2, horizon=1)
        self.assertEqual(n, 4)
        self.assertEqual(len(X), 16)

    def test_games_selection_has_no_duplicates_across_epoch_boundary(self):
        games = [dict(game_file=str(j)) for j in range(13)]
        for iteration in range(30):
            selected = run.select_games(games, 3, iteration, 5)
            self.assertEqual(len(set(v["game_file"] for v in selected)), 5)
            self.assertEqual(selected, run.select_games(games, 3, iteration, 5))

    def test_sampling_seeds_are_disjoint(self):
        self.assertNotEqual(rollout.sample_seed(1000, 1, 0, 2),
                            rollout.sample_seed(1000, 2, 0, 2))
        self.assertNotEqual(rollout.sample_seed(1000, 1, 0, 2),
                            rollout.sample_seed(1000, 1, 1, 2))

    def test_prepare_rejects_validation_as_training(self):
        ep = fake_episode(0, 0, True)
        prepare.validate_train([ep, fake_episode(0, 1, False)], 0)
        ep["provenance"]["split"] = "valid_unseen"
        with self.assertRaisesRegex(ValueError, "non-train"):
            prepare.validate_train([ep], 0)

    def test_disk_prepare_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rollouts = folder / "round_000" / "rollouts"
            rollouts.mkdir(parents=True)
            for e in cohort(10):
                filename = f"s{e['seed']}_r{e['rep']}.json"
                (rollouts / filename).write_text(json.dumps(e))
            args = SimpleNamespace(run_dir=folder, rollouts=rollouts, iteration=0,
                                   out=folder / "round_000" / "prepared", mode="ctha",
                                   family="habit", base_advantage="grpo",
                                   feature_window=1, train_horizon=1, aux_horizon=1,
                                   ridge_alpha=5., min_mixed=2, min_pairs=2,
                                   window=4, permutations=2, bootstraps=20,
                                   min_delta=0., omega=.3, anneal_start=0,
                                   anneal_end=4, aux_clip=2., sign_fraction=.75,
                                   baseline="legal", gamma=1., local_weight=1.)
            first = prepare.prepare(args)
            second = prepare.prepare(args)
            self.assertEqual(first, second)
            self.assertEqual(first["n_samples"], 40)
            self.assertEqual(len(load_episodes(rollouts, 0)), 40)
            samples = args.out / "samples.jsonl"
            self.assertEqual(hashlib.sha256(samples.read_bytes()).hexdigest(),
                             first["samples_sha256"])

    def test_train_refuses_offpolicy_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "samples.jsonl"
            payload = json.dumps(dict(prompt="hello", response="<action>go</action>",
                                      response_token_ids=[15], advantage=1)) + "\n"
            d.write_text(payload)
            report = root / "advantage_report.json"
            report.write_text(json.dumps(dict(samples_sha256=hashlib.sha256(payload.encode()).hexdigest())))
            summary = root / "rollout_summary.json"
            summary.write_text(json.dumps(dict(provenance=dict(
                split="train", dry=False, model=str(root.resolve()),
                adapter_sha256=None, temperature=.7, max_model_len=6144))))
            args = SimpleNamespace(data=d, advantage_report=report, rollout_summary=summary,
                                   model=root, adapter=None, temperature=.7,
                                   max_model_len=6144, out=root / "adapter",
                                   lr=1e-5, epsilon=.2, epochs=1, accum=8,
                                   grad_clip=1., action_only=False,
                                   no_grad_checkpoint=False, seed=1)
            rows, config, completed = train.verify_inputs(args)
            self.assertFalse(completed)
            self.assertEqual(len(rows), 1)
            args.temperature = 1.0
            with self.assertRaisesRegex(ValueError, "temperature"):
                train.verify_inputs(args)


if __name__ == "__main__":
    unittest.main()
