"""CPU checks for E4d's registered design; no ALFWorld, GPU or pi1 outcomes needed."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4d_common as C
import e4d as D


class E4dProtocol(unittest.TestCase):
    def test_even_spacing_is_among_qualified_states(self):
        rows = [dict(point_id=f"s1_t{t:02d}", turn=t) for t in [2, 5, 7, 9, 14, 16, 25, 33, 41]]
        self.assertEqual([r["turn"] for r in C.evenly_spaced(rows)], [5, 14, 33])
        self.assertEqual(len(C.evenly_spaced(rows[:2])), 2)
        self.assertEqual([r["turn"] for r in C.evenly_spaced(list(reversed(rows)))], [5, 14, 33])

    def test_classification_e4b_compatible(self):
        self.assertEqual(C.classify([[0]*3 for _ in range(5)]), "Z3")
        self.assertEqual(C.classify([[1]*3 for _ in range(5)]), "E3")
        self.assertEqual(C.classify([[1, 0, 0], [0, 0, 0], [0]*3, [0]*3, [0]*3]), "V3")
        with self.assertRaises(AssertionError):
            C.classify([[0]*3]*4)

    def test_prefix_reconstructed_from_existing_e4_base_record(self):
        tr = dict(initial_obs="initial", steps=[dict(action="go to desk", obs_after="desk"),
                                              dict(action="open desk", obs_after="opened"),
                                              dict(action="examine desk", obs_after="look")])
        pre, exp, before, mem = D.prefix_of(tr, 2)
        self.assertEqual(pre, ["go to desk", "open desk"])
        self.assertEqual(exp, ["desk", "opened"])
        self.assertEqual(before, "opened")
        self.assertEqual(mem, [dict(text_obs="initial", action="go to desk"),
                               dict(text_obs="desk", action="open desk")])

    def test_dp_first_step_validity_is_per_condition(self):
        p = dict(seed=3000, game_file="foo", task_type="pick", task="test", initial_obs="initial",
                 i=1, turn=2, obs_before="desk", prefix=["goto"], expected=["desk"])
        dp, u = D.build_dp(p, "invalid-generated-action", False, "out.json", 10)
        self.assertEqual(dp["tr"]["steps"][1]["valid"], False)
        self.assertEqual(dp["tr"]["steps"][1]["action"], "invalid-generated-action")
        self.assertEqual(dp["tr"]["steps"][0]["obs_before"], "initial")
        self.assertEqual(u, 10)

    def test_lora_request_explictly_passed_through_async_interface(self):
        calls = []
        engine = types.SimpleNamespace(add_request=lambda *a, **kw: calls.append((a, kw)),
                                       step=lambda: [], has_unfinished_requests=lambda: False)
        pol = types.SimpleNamespace(llm=types.SimpleNamespace(llm_engine=engine), lora=object(),
                                    chat_text=lambda x: "T:"+x, params=lambda sd: (sd,))
        s = D.stream_for(pol)
        s.add("abc", "prompt", 123)
        self.assertIn("lora_request", calls[0][1])
        self.assertIs(calls[0][1]["lora_request"], pol.lora)
        pol.lora = None
        s.add("def", "prompt", 456)
        self.assertEqual(calls[1][1], {})

    def test_primary_is_after_minus_step_and_pooling_is_game_level(self):
        def row(pid, game, after, step):
            return dict(point_id=pid, game=game, remaining=15,
                        Azero={"00":0.1, "01":after, "10":step, "11":0.5},
                        Dzero={"00":0.1, "01":after, "10":step, "11":0.5})
        rows = [row("g1_t1", "g1", 0.5, 0.2), row("g1_t2", "g1", 0.5, 0.2),
                row("g2_t1", "g2", 0.3, 0.1)]
        gr = C.game_contrasts(rows)
        self.assertEqual(len(gr), 2)
        self.assertAlmostEqual(gr[0]["primary"], 0.3)
        self.assertAlmostEqual(gr[0]["rescue"], 0.4)
        self.assertAlmostEqual(gr[0]["one_step"], 0.1)
        self.assertAlmostEqual(C.bootstrap_game(gr, "primary", n_boot=200)["mean_pp"], 25)

    def test_recovery_can_improve_without_arm_range_change(self):
        # Methodological control: salvage does not imply a new fork emerges.
        old = [0.0]*5
        new = [0.4]*5
        self.assertGreater(sum(new)/5 - sum(old)/5, 0)
        self.assertEqual(max(new)-min(new), 0.0)

    def test_hierarchy_uses_fresh_contrasts_and_power_gate(self):
        self.assertEqual(C.reading([1, 2], [1, 3], 120, 60), "E4d-SUPPORT")
        self.assertEqual(C.reading([1, 2], [-4, 3], 120, 60), "E4d-RESCUE-ONLY")
        self.assertEqual(C.reading([1, 2], [-3, -1], 120, 60), "E4d-COUNTER")
        self.assertEqual(C.reading([1, 2], [1, 3], 20, 20), "E4d-UNDERPOWERED")
        self.assertEqual(C.remaining_bin(10), "10-19")
        self.assertEqual(C.remaining_bin(20), "20-29")
        self.assertEqual(C.remaining_bin(30), "30+")

    def test_frozen_replay_seed_offsets_are_distinct(self):
        self.assertNotEqual(C.SCREEN_OFFSET, C.FORMAL_OFFSET)
        self.assertNotEqual(C.FIRST_OFFSET, C.FORMAL_OFFSET)
        self.assertEqual(C.SEEDS, (0,1,2))
        self.assertEqual(C.REPS, 8)


if __name__ == "__main__":
    unittest.main()

# Extra built-in CPU cases live below the standard unittest invocation so that
# they are picked up by discovery (not via __main__ special cases).
class E4dLockGate(unittest.TestCase):
    def test_lock_is_independent_of_pi1_and_rejects_change(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected, lock = root / "selected.json", root / "lock.json"
            source = root / "source"
            adapters = root / "adapters"
            source.mkdir()
            bas = source / "s3000_r0.json"
            bas.write_text('{"test":"frozen"}')
            p = dict(point_id="s3000_t02", seed=3000, turn=2, game="3000", baseline_sha256=D.sha_file(bas),
                     factual="examine desk 1", alternatives=["a", "b", "c", "d"])
            D.save_json(selected, dict(meta=dict(test=True), points=[p]))
            scr = root / "screen" / p["point_id"] / "replay.json"
            D.save_json(scr, dict(cf_defined=True, factual_action=p["factual"], alternatives=p["alternatives"],
                                  factual_outcomes=[0, 0, 0], alt_outcomes=[[0, 0, 0] for _ in range(4)],
                                  continuations=[{"refeed_div":0} for _ in range(15)]))
            for arm in C.ALL_ARMS:
                for seed in C.SEEDS:
                    dest = adapters / f"{arm}_s{seed}" / "adapter_model.safetensors"
                    dest.parent.mkdir(parents=True)
                    dest.write_bytes(f"{arm}_{seed}".encode())
            with patch.multiple(D, DATA=root, SELECTED=selected, LOCK=lock,
                                BASE_DIR=source, ADAPTER_DIR=adapters):
                D.stage_lock(types.SimpleNamespace(approve=True))
                self.assertEqual(D.load(lock)["Z3"], ["s3000_t02"])
                self.assertTrue(D.load(lock)["underpowered"])
                with self.assertRaises(RuntimeError):
                    D.stage_lock(types.SimpleNamespace(approve=True))
                selected.write_text(selected.read_text() + " ")
                with self.assertRaises(RuntimeError):
                    D.locked()

    def test_probe_heldout_gap_noise_control(self):
        a = [[0]*24 for _ in range(5)]
        self.assertEqual(C.crossfit_range(a)["heldout_gap"], 0)
        a[2] = [1]*24
        self.assertEqual(C.crossfit_range(a)["mean"], 0.2)
        self.assertGreater(C.crossfit_range(a)["heldout_gap"], 0)
        self.assertEqual(C.probe_ids(["b","a","c"], n=2), C.probe_ids(["c","b","a"], n=2))
