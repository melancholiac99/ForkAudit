import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from .e2512_gate import compare_counts, decision
from .e2512_routing import balanced
from .e2512_native import merge_tapes
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, save, read


class SerialLayout(unittest.TestCase):
    def test_frozen_balancing_independent_of_enumeration(self):
        goals=list(range(32))
        lengths={g:100+g for g in goals}
        route,loads=balanced(goals,lengths)
        self.assertEqual((route,loads),balanced(list(reversed(goals)),lengths))
        self.assertEqual([sum(v==r for v in route.values()) for r in range(4)],[8]*4)
        self.assertLessEqual(max(loads)-min(loads),2)
        with self.assertRaises(ValueError):balanced([1,1,2,3],lengths)

    def test_gate_excludes_ties_and_missing_p100(self):
        self.assertEqual(compare_counts(334,333,800),'RUN_CONTROLS')
        self.assertEqual(compare_counts(333,333,800),'NOT_RUN')
        self.assertEqual(compare_counts(332,333,800),'NOT_RUN')
        self.assertEqual(decision(None,20)['decision'],'RUN_CONTROLS')

    def test_gate_uses_only_signed_new_sample_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);routing=root/'routing';routing.write_text('map')
            save(root/'asin_split_manifest.json',{'T_opd_eval':[{'goal_index':g} for g in range(200)]})
            l=SimpleNamespace(root=root,folder=lambda k,a:root/a/str(k))
            save(l.folder(100,'P')/'optimizer.json',{'safetensors_sha256':'actor'})
            for arm,k,signature,successes in [('B',0,'B@0',150),('P',100,'actor',151)]:
                for rank in range(4):
                    trials=[[g,i,int(4*g+i<successes)] for g in range(rank,200,4) for i in range(4)]
                    save(root/'evaluation'/f'{k:03}'/arm/f'rank{rank}'/'gate_T_opd_eval_sample.json',
                         dict(split='T_opd_eval',decode='sample',rank=rank,arm=arm,step=k,checkpoint=signature,
                              routing_sha=digest_file(routing),trials=trials))
            with patch(__package__+'.e2512_gate.ROUTING',routing),patch(__package__+'.e2512_gate.PARENT',root),\
                 patch(__package__+'.e2512_gate.owners',return_value={g:g%4 for g in range(200)}):
                self.assertEqual(decision(l,100)['counts'],{'B':150,'P':151})
                bad=root/'evaluation/100/P/rank1/gate_T_opd_eval_sample.json'
                v=read(bad);v['split']='V';save(bad,v)
                with self.assertRaises(AssertionError):decision(l,100)

    def test_new_first_tape_is_shared_across_fresh_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);routing=root/'routing';routing.write_text('test')
            l=SimpleNamespace(root=root,folder=lambda k,a:root/'pilot'/f'{k:02}'/a)
            owners={g:g%4 for g in range(32)}
            for a in ('P','R'):l.folder(1,a).mkdir(exist_ok=True,parents=True)
            save(l.folder(1,'P').parent/'tasks.json',[{'goal_index':g} for g in owners])
            for rank in range(4):
                p=root/'shared_first/native'/f'rank{rank}';p.mkdir(parents=True)
                goals=[g for g in owners if owners[g]==rank];hashes=[]
                for rep in range(2):
                    f=p/f'rollout_{rep}.jsonl'
                    f.write_text(''.join(json.dumps(dict(goal=g,rep=rep,step=0,done=True,legal=(g%2==0)))+'\n' for g in goals))
                    hashes.append(digest_file(f))
                save(p/'complete.json',dict(goals=goals,rank=rank,checkpoint='B@0',routing_sha=digest_file(routing),rollout_sha256=hashes))
            with patch(__package__+'.e2512_native.ROUTING',routing),patch(__package__+'.e2512_native.owners',return_value=owners):
                for a in ('P','R'):merge_tapes(l,1,a)
            p=read(l.folder(1,'P')/'collection.json')
            self.assertEqual(read(l.folder(1,'R')/'collection.json'),p)
            self.assertEqual((p['records'],p['legal_rate']),(64,.5))

    def test_serial_driver_only_launches_authorized_controls(self):
        import os
        from contextlib import ExitStack
        from . import e2512_driver as driver
        for choice,expected in [('NOT_RUN',['P']),('RUN_CONTROLS',['P','R-matched','R'])]:
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as temp, ExitStack() as mocks:
                root=Path(temp);l=SimpleNamespace(root=root,folder=lambda k,a:root/'pilot'/str(k)/a)
                calls=[]
                def train(layout,a):
                    calls.append(a)
                    return dict(arm=a,step=100,status='COMPLETED')
                def gate(layout,k):
                    result=dict(decision=choice)
                    save(root/'controls_gate.json',result)
                    return result
                for name,fn in dict(Layout=lambda name:l,setup=lambda layout:None,launch=lambda *args:None,
                    merge_common=lambda *args:None,behavior_report=lambda *args:None,train_arm=train,
                    endpoints_at=lambda *args:{},gate=gate,commit_delivery=lambda:None).items():
                    mocks.enter_context(patch.object(driver,name,fn))
                mocks.enter_context(patch(__package__+'.e25_delivery.final_report',lambda layout:None))
                mask=os.umask(0o077)
                try:driver.main()
                finally:os.umask(mask)
                self.assertEqual(calls,expected)
                result=read(root/'complete.json')
                self.assertEqual(list(result['endpoints']),expected)
                self.assertEqual(result['not_run'],[a for a in driver.ORDER if a not in expected])
