import unittest
from unittest.mock import patch
from copy import deepcopy
from exploratory_shop_audit.grpo_forced_audit.audit import candidates,semantics,reliability
from exploratory_shop_audit.grpo_forced_audit.cpu import complete_configurations
from exploratory_shop_audit.grpo_forced_audit.analysis import interval
from exploratory_shop_audit.env_worker import parse_action_fixed


class ExperimentTests(unittest.TestCase):
    def test_route_does_not_read_session_or_query_as_page(self):
        from .interface import route_page_name
        for route in ['item_page','item_sub_page','search_results','done']:
            self.assertEqual(route_page_name(f'http://localhost/{route}/search_results:item_page:done/search_results'),route)
        self.assertEqual(route_page_name('http://localhost/audit:search_results:0'), '')
        self.assertIsNone(route_page_name(None))

    def test_dp_preserves_microgroups_global_minis_and_tail(self):
        from .parallel import partitions
        for world in [1,2,4]:
            size=144 if world==4 else 136
            lengths=[(i*17)%83+1 for i in range(size)]
            groups=partitions(lengths,world)
            self.assertEqual(sorted(i for g in groups for i in g),list(range(size)))
            self.assertEqual(len({len(g) for g in groups}),1)
            for g in groups:
                for start in range(0,len(g),4):
                    micro=g[start:start+4]
                    self.assertEqual(micro,list(range(micro[0],micro[0]+4)))
                    self.assertEqual(micro[0]%4,0)
            for start in range(0,size,64):
                local_start=start//world;local_end=min(start+64,size)//world
                observed=[i for g in groups for i in g[local_start:local_end]]
                self.assertEqual(sorted(observed),list(range(start,min(start+64,size))))

    def test_dp_checkpoint_migration_is_idempotent_and_shares_storage(self):
        import tempfile,os
        from pathlib import Path
        from .parallel import prepare_restore
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);actor=root/'actor';actor.mkdir()
            for kind in ['model','optim','extra_state']:(actor/f'{kind}_world_size_1_rank_0.pt').write_bytes(kind.encode())
            (actor/'live_fingerprint_rank_0.json').write_text('{}')
            prepare_restore(root,4);prepare_restore(root,4)
            for rank in range(4):
                for kind in ['model','optim','extra_state']:
                    self.assertTrue(os.path.samefile(actor/f'{kind}_world_size_1_rank_0.pt',actor/f'{kind}_world_size_4_rank_{rank}.pt'))

    def test_uncapped_requires_explicit_researcher_record(self):
        import tempfile
        from pathlib import Path
        from .budget import uncapped
        from .common import save
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.assertFalse(uncapped(root))
            save(root/'resource_authorization.json',{'gpu_hour_limit':None});self.assertFalse(uncapped(root))
            save(root/'resource_authorization.json',{'gpu_hour_limit':None,'researcher_authorized':True});self.assertTrue(uncapped(root))

    def test_full_checkpoint_retention_keeps_adapters_and_latest_two(self):
        import tempfile
        from pathlib import Path
        from .parallel import prune_full_checkpoints
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for n in [1,2,10]:
                actor=root/f'training/checkpoints/global_step_{n}/actor';(actor/'lora_adapter').mkdir(parents=True)
                for name in ['live_fingerprint_rank_0.json','model_world_size_4_rank_0.pt','optim_world_size_4_rank_0.pt','extra_state_world_size_4_rank_0.pt','lora_adapter/adapter_model.safetensors']:(actor/name).write_bytes(b'data')
            prune_full_checkpoints(root)
            self.assertFalse((root/'training/checkpoints/global_step_1/actor/model_world_size_4_rank_0.pt').exists())
            self.assertTrue((root/'training/checkpoints/global_step_1/actor/lora_adapter/adapter_model.safetensors').exists())
            for n in [2,10]:self.assertTrue((root/f'training/checkpoints/global_step_{n}/actor/model_world_size_4_rank_0.pt').exists())

    def test_exact_action_whitespace_and_brackets(self):
        value='a  [b]  c'
        self.assertEqual(parse_action_fixed(f'click[{value}]'),('click',value))
        self.assertNotEqual(semantics('click[a  b]'),semantics('click[a b]'))
    def test_complete_native_configurations_exclude_missing_axis(self):
        p={'options':{'color':['red','blue'],'size':['s','m']}}
        values=list(complete_configurations(p));self.assertEqual(len(values),4)
        self.assertTrue(all(len(v)==2 for v in values))
        self.assertEqual(len(list(complete_configurations(p,{'color':'red'}))),2)
    def test_collision_does_not_create_unreachable_complete_configuration(self):
        self.assertEqual(list(complete_configurations({'options':{'a':['x'],'b':['x']}})),[])
    def test_forced_results_keep_original_illegal_and_visible_top5(self):
        state={'public':{'page_type':'search_results','products':[{'asin':str(i)} for i in range(8)]},'available':{'clickables':[str(i) for i in range(8)]}}
        actions,_=candidates(state,'illegal[action]',1)
        self.assertEqual(actions,['illegal[action]']+[f'click[{i}]' for i in range(5)])
    def test_product_reserves_buy_and_round_robin(self):
        state={'public':{'page_type':'product_detail','available_options':{'a':['x','y'],'b':['u','v']},'selected_options':{}},'available':{'clickables':['x','y','u','v','buy now']},'stop_available':True}
        actions,_=candidates(state,'click[x]',42)
        self.assertIn('click[buy now]',actions);self.assertEqual(len(actions),5)
        self.assertEqual(candidates(state,'click[x]',42)[0],actions)
    def test_no_signal_reliability_is_na(self):
        r=reliability([(0.,1.),(-.1,1.)],'negative')
        self.assertIsNone(r['rho3']);self.assertFalse(r['identifiable'])
    def test_concurrent_gpu_budget_counts_active_devices(self):
        import tempfile
        from pathlib import Path
        from .common import append,save
        from .budget import occupancy
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'active_gpu_sessions').mkdir()
            append(run/'cost_ledger.jsonl',{'event':'end','gpu_hours':2})
            save(run/'active_gpu_sessions/a.json',{'started_unix':100,'gpu_count':1})
            save(run/'active_gpu_sessions/b.json',{'started_unix':1900,'gpu_count':2})
            self.assertAlmostEqual(occupancy(run,3700),4)

    def test_native_resume_identity_preserves_eight_repeats(self):
        from types import SimpleNamespace
        from .native_cache import batch_identity
        cfg=SimpleNamespace(env=SimpleNamespace(rollout=SimpleNamespace(n=8)))
        env=SimpleNamespace(config=cfg,stage='train',checkpoint=10,decode='sample')
        batch=SimpleNamespace(non_tensor_batch={'env_kwargs':[{'task_id':'a'},{'task_id':'b'}]})
        tasks,keys=batch_identity(batch,env,True)
        self.assertEqual(len(set(keys)),16)
        self.assertEqual(keys[7],'train:10:sample:a:7')
        self.assertEqual(keys[8],'train:10:sample:b:0')
        env.checkpoint=20
        self.assertFalse(set(keys)&set(batch_identity(batch,env,True)[1]))

    def test_empty_ci_is_na_not_zero(self):
        self.assertIsNone(interval([],'empty')['estimate'])


if __name__=='__main__':unittest.main()
