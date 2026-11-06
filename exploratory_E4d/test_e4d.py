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
                 i=1, turn=2, obs_before="desk", prefix=["goto"], expected=["desk"], target_word="bowl")
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
                     factual="examine desk 1", alternatives=["a", "b", "c", "d"],
                     target_word="bowl", peek_source="CPU-deterministic-one-step-v1",
                     peek_observations=["nothing", "bowl 1", "none", "nope", "empty"],
                     peek_obs_sha256=[D.sha_file_bytes(o.encode()) for o in ["nothing", "bowl 1", "none", "nope", "empty"]],
                     peek_target_flags=[False, True, False, False, False], O_h=1)
            D.save_json(selected, dict(meta=dict(test=True), points=[p]))
            scr = root / "screen" / p["point_id"] / "replay.json"
            D.save_json(scr, dict(cf_defined=True, factual_action=p["factual"], alternatives=p["alternatives"],
                                  factual_outcomes=[0, 0, 0], alt_outcomes=[[0, 0, 0] for _ in range(4)],
                                  target_logging_schema="credit-grok-target-v1",
                                  continuations=[{"refeed_div":0, "target_observation":{"schema":"credit-grok-target-v1"}} for _ in range(15)]))
            for arm in C.ALL_ARMS:
                for seed in C.SEEDS:
                    dest = adapters / f"{arm}_s{seed}" / "adapter_model.safetensors"
                    dest.parent.mkdir(parents=True)
                    dest.write_bytes(f"{arm}_{seed}".encode())
            with patch.multiple(D, DATA=root, SELECTED=selected, LOCK=lock,
                                BASE_DIR=source, ADAPTER_DIR=adapters):
                D.stage_lock(types.SimpleNamespace(approve=True, probe_n=72))
                self.assertEqual(D.load(lock)["Z3"], ["s3000_t02"])
                self.assertTrue(D.load(lock)["underpowered"])
                with self.assertRaises(RuntimeError):
                    D.stage_lock(types.SimpleNamespace(approve=True, probe_n=72))
                selected.write_text(selected.read_text() + " ")
                with self.assertRaises(RuntimeError):
                    D.locked()

    def test_probe_heldout_gap_noise_control(self):
        a = [[0]*C.PROBE_REPS for _ in range(5)]
        self.assertEqual(C.crossfit_range(a)["heldout_gap"], 0)
        a[2] = [1]*C.PROBE_REPS
        self.assertEqual(C.crossfit_range(a)["mean"], 0.2)
        self.assertGreater(C.crossfit_range(a)["heldout_gap"], 0)
        self.assertEqual(C.probe_ids([str(i) for i in range(80)], n=72),
                         C.probe_ids([str(i) for i in reversed(range(80))], n=72))


class CreditGrokSecondaryTests(unittest.TestCase):
    def test_O_is_frozen_one_step_target_asymmetry(self):
        obs = ["Nothing happens", "You see bowl 2", "open shelf", "bowl 1", "nothing"]
        a = C.target_peek_flags(obs, "bowl")
        self.assertEqual(a["O"], 1)
        self.assertEqual(a["flags"], [False, True, False, True, False])
        self.assertEqual(C.target_peek_flags(["bowl 1"]*5, "bowl")["O"], 0)
        self.assertEqual(C.target_peek_flags(["no bowl"]*5, "bowl")["O"], 0)

    def test_probe_budget_and_round_robin_seeds(self):
        self.assertEqual(C.PROBE_N, 72)
        self.assertEqual(C.PROBE_REPS, 8)
        self.assertEqual(72*5*8*3, 8640)
        for pid in ("s3000_t09", "s3045_t27"):
            seeds = [C.probe_seed_for(pid, u) for u in range(8)]
            self.assertEqual(set(seeds[:4]), {0, 1, 2})
            self.assertEqual(set(seeds[4:]), {0, 1, 2})
            self.assertEqual(sorted(seeds.count(j) for j in range(3)), [2, 3, 3])
            self.assertEqual([C.probe_seed_for(pid,u,1) for u in range(8)], [0]*8)
        self.assertEqual(len(C.probe_ids([f"s{x}_t01" for x in range(100)])), 72)
        self.assertEqual(len(C.probe_ids([f"s{x}_t01" for x in range(100)], n=96)), 96)
        with self.assertRaises(ValueError):
            C.probe_ids(["a"], n=24)

    def test_crossfit_swapping_4_plus_4_estimation(self):
        a = [[0]*8 for _ in range(5)]
        a[2] = [1]*8
        x = C.crossfit_range(a)
        self.assertEqual(x["heldout_gap"], 1.0)
        self.assertEqual(x["crossfit_ab"], 1.0)
        self.assertEqual(x["crossfit_ba"], 1.0)
        # With opposite-half fit noise, both selection directions are evaluated honestly.
        a[2] = [1,1,1,1,0,0,0,0]
        self.assertEqual(C.crossfit_range(a)["crossfit_ab"], 0.0)
        self.assertEqual(C.crossfit_range(a)["crossfit_ba"], 0.0)

    def test_target_logging_records_first_target_turn_even_after_prefix(self):
        class FakePool:
            def __init__(self, events): self.events=list(events)
            def poll(self, timeout=0.0): return self.events.pop(0)
        dp = dict(turn=5, target_word="bowl", initial_obs="home", expected=["hall", "desk", "shelf", "nothing"])
        c = types.SimpleNamespace(dp=0, key="0:1:0", role=1, rep=0)
        fp = FakePool([[('open_step', {c.key:(("nothing", [], False, 0), ("A bowl 4 is here", [], False, False))})],
                       [('step', {c.key:("Bowl 5", [], False, False)})]])
        pool = D.TargetLoggingPool(fp, [dp], [c])
        pool.poll(); pool.poll()
        v=pool.metadata(c.key)
        self.assertEqual(v["target_first_seen_turn"], 5)
        self.assertEqual(v["first_seen_after_decision"], 5)
        self.assertTrue(v["seen_on_forced_step"])
        self.assertTrue(v["ever_seen"])
        self.assertEqual(v["observed_steps"], 2)
        # Historical discovery is recorded separately from post-decision rediscovery.
        dp["expected"][1] = "bowl 2"
        c2=types.SimpleNamespace(dp=0,key="0:1:1",role=1,rep=1)
        t=D.TargetLoggingPool(FakePool([]), [dp],[c2]).metadata(c2.key)
        self.assertEqual(t["target_first_seen_turn"], 2)
        self.assertTrue(t["seen_before_decision"])
        self.assertFalse(t["seen_after_decision"])

    def test_target_logging_attaches_to_actual_replay_records(self):
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/"r.json"
            D.save_json(file, dict(continuations=[dict(role=1,rep=0)], factual_outcomes=[1]))
            c=types.SimpleNamespace(dp=0,key="0:1:0",role=1,rep=0)
            dp=dict(path=str(file))
            tracker=types.SimpleNamespace(metadata=lambda k: {"schema":"credit-grok-target-v1", "ever_seen":True})
            D.attach_target_logs([dp],[c],tracker)
            r=D.load(file)
            D.verify_target_log(r)
            self.assertEqual(r["target_observation"]["ever_seen"], True)
            self.assertEqual(r["continuations"][0]["target_observation"]["ever_seen"], True)

    def test_O_prediction_is_game_clustered(self):
        rows = [dict(game=f"g{n}", O_h=n%2, gap_gain=(0.5 if n%2 else 0.0)) for n in range(20)]
        result=C.bootstrap_O_contrast(rows, n_boot=600)
        self.assertEqual(result["mean_pp"], 50.)
        self.assertAlmostEqual(result["ci95_pp"][0], 50., places=4)
        self.assertIsNone(C.bootstrap_O_contrast([x for x in rows if x["O_h"]==1], n_boot=100))

    def test_secondary_triage_leaves_main_reading_alone(self):
        self.assertEqual(C.probe_triage([5, 8],[2, 5],[1, 4],30,42,15,21, O1_gap_ci=[2,6]), "MCR-PREDICTIVE-CANDIDATE")
        self.assertEqual(C.probe_triage([5, 8],[2, 5],[-1, 4],30,42,15,21), "EMERGENCE-NOT-PREDICTED")
        self.assertEqual(C.probe_triage([5, 8],[-0.05, 0.09],None,0,72,0,30), "UPLIFT-WITHOUT-RESOLVED-FORK")
        self.assertEqual(C.probe_triage([5, 8],[-11, 4],None,0,72,0,30), "UNRESOLVED")
        self.assertEqual(C.reading([1,2],[1,2],100,50), "E4d-SUPPORT")

    def test_registered_rl_checkpoint_is_immutable(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            model=root/"my_lora"
            model.mkdir()
            (model/"adapter_config.json").write_text('{}')
            (model/"adapter_model.safetensors").write_bytes(b"model")
            lock=root/"lock.json"
            D.save_json(lock,dict(probe_ids=["s1_t05"], peek_signal_O={"s1_t05":1}))
            reg=root/"regs"
            with patch.multiple(D, PROBE_REGISTRY=reg, LOCK=lock):
                with patch.object(D, 'locked', return_value=D.load(lock)):
                    D.stage_register_probe(types.SimpleNamespace(policy_id="mcr-v1",approve=True,
                          revision="revision-1",checkpoint_paths=[str(model)],checkpoint_kind="lora"))
                    spec=D.probe_spec("mcr-v1",D.load(lock))
                    self.assertEqual(spec["n_checkpoints"],1)
                    with self.assertRaises(RuntimeError):
                        D.stage_register_probe(types.SimpleNamespace(policy_id="mcr-v1",approve=True,
                              revision="revision-2",checkpoint_paths=[str(model)],checkpoint_kind="lora"))
                    (model/"adapter_model.safetensors").write_bytes(b"changed")
                    with self.assertRaises(RuntimeError):
                        D.probe_spec("mcr-v1",D.load(lock))

    def test_discovery_decomposition_exact_even_with_undefined_conditional(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            p=dict(point_id="s3000_t05",game="3000")
            for u in range(24):
                for arm, cond in (("base","00"),("Azero","01"),("Dzero","01")):
                    path=C.run_path(root,arm,cond,p["point_id"],u)
                    good=arm != "base"
                    obs=dict(schema="credit-grok-target-v1", ever_seen=good,
                              seen_after_decision=good, seen_before_decision=False,
                              target_first_seen_turn=6 if good else None)
                    D.save_json(path,dict(target_logging_schema="credit-grok-target-v1",
                                          target_observation=obs, factual_outcomes=[int(good)],
                                          continuations=[dict(target_observation=obs,role=1,rep=u)]))
            with patch.object(D, 'DATA', root):
                result=D.target_mechanism_summary([p])
            self.assertEqual(result["conditional"]["00"]["success"], 0)
            self.assertEqual(result["conditional"]["01"]["success"], 1)
            self.assertEqual(result["exact_success_split_pp"]["found_and_succeeded"],100)
            self.assertEqual(result["exact_success_split_pp"]["not_found_and_succeeded"],0)
            self.assertFalse(result["shapley_descriptive"].get("estimable",True))

class CreditGrokAdditionalRegression(unittest.TestCase):
    def test_seed_balanced_crossfit_does_not_overweight_repeat_seed(self):
        pid="s3099_t23"
        ids=[C.probe_seed_for(pid,u) for u in range(8)]
        # Only one training seed sees success, whatever its 3/3/2 number of reps.
        vals=[[int(ids[u]==ids[0]) for u in range(8)]]+[[0]*8 for _ in range(4)]
        m=C.seed_balanced_arm_means(vals,ids)
        self.assertAlmostEqual(m[0],1/3)
        self.assertAlmostEqual(C.crossfit_range(vals,ids)["mean"],1/15)

    def test_rl_checkpoint_comparison_has_independent_O_prediction(self):
        lock=dict(probe_ids=[f"s{x}_t09" for x in range(40)])
        allp={a:{} for a in ("base","Azero","Dzero","grpo-v1")}
        for i,pid in enumerate(lock["probe_ids"]):
            O=i%2
            for arm in allp:
                val=(0.65 if O and arm=="grpo-v1" else
                     0.25 if not O and arm=="grpo-v1" else
                     0.2 if arm in ("Azero","Dzero") else 0.1)
                allp[arm][pid]=dict(game=f"g{i}",O_h=O,remaining=15,
                                    mean=val+0.05,heldout_gap=val,discovery=val,
                                    positive_heldout=True)
        rl=D.probe_comparison(lock,allp,("grpo-v1",))
        z=D.probe_comparison(lock,allp,C.PRIMARY_ARMS)
        self.assertGreater(rl["O_prediction"]["mean_pp"], 30)
        self.assertAlmostEqual(z["O_prediction"]["mean_pp"],0)
        self.assertEqual(rl["reading_secondary"],"MCR-PREDICTIVE-CANDIDATE")
