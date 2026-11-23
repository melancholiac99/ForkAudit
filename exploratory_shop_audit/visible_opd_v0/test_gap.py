import unittest

import numpy as np
import torch

from .gap_report import contrasts, holm
from .gap_run import window_decision
from .gap_score import target
from .e25_projector import FitData


class GapProtocol(unittest.TestCase):
    def test_parallel_lanes_join_only_after_both_endpoints(self):
        import hashlib
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        from . import gap_run
        from exploratory_shop_audit.grpo_forced_audit.common import save

        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            layout=SimpleNamespace(root=root)
            save(root/'initialization_check.json',{'passed':True})
            hashes=[]
            for rep in range(2):
                path=root/'shared_first'/f'rollout_{rep}.jsonl'
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(bytes([rep]))
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            save(root/'shared_first/collection.json',dict(checkpoint='B@0',rollout_sha256=hashes))
            save(root/'parallel_resume.json',dict(devices=gap_run.LANE_DEVICES,shared_rollout_sha256=hashes))
            def finish_arm(layout,arm):
                save(layout.root/'endpoints'/f'{arm}.json',{'update':10})
            def report(layout,ends):
                self.assertEqual(ends,{'R-gap':10,'P-gap':10})
                save(root/'endpoint_report.json',ends)
            with patch.object(gap_run,'GapLayout',return_value=layout), patch.object(gap_run,'PUBLIC',root), \
                 patch.object(gap_run,'train_arm',side_effect=finish_arm), \
                 patch.object(gap_run,'evaluate_endpoints',side_effect=report) as evaluate:
                for arm in ('P-gap','R-gap','R-gap'):
                    with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=gap_run.LANE_DEVICES[arm]):
                        gap_run.parallel_lane(arm)
                    if arm=='P-gap':
                        evaluate.assert_not_called()
                evaluate.assert_called_once()
                with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=gap_run.LANE_DEVICES['R-gap']):
                    with self.assertRaises(ValueError):
                        gap_run.parallel_lane('P-gap')

    def test_parallel_evaluation_matches_original_stages_and_holds_v(self):
        from collections import Counter
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        from . import gap_eval_parallel, gap_report, gap_run
        from exploratory_shop_audit.grpo_forced_audit.common import save

        with TemporaryDirectory() as tmp:
            layout=SimpleNamespace(root=Path(tmp))
            arms={'R-gap':45,'P-gap':50}
            original=[]
            with patch.object(gap_run,'launch',side_effect=lambda l,*args:original.append(args)), \
                 patch.object(gap_run,'qualify_pages'),patch.object(gap_report,'report'):
                gap_run.evaluate_endpoints(layout,arms)
            save(layout.root/'endpoints/R-gap.json',dict(update=45))
            actual=[]
            prepared=set()
            def stage(l,phase,k,arm,pool,*mode,devices):
                group='R-gap' if arm=='B' else arm
                self.assertEqual(devices,gap_run.LANE_DEVICES[group])
                if phase=='mass' or mode==('page',):
                    self.assertIn(pool,prepared)
                actual.append((phase,k,arm,pool,*mode))
            with patch.object(gap_eval_parallel,'launch',side_effect=stage), \
                 patch.object(gap_eval_parallel,'qualify_pages',side_effect=lambda l,p:prepared.add(p)), \
                 patch.object(gap_report,'report') as report:
                with self.assertRaises(FileNotFoundError):
                    gap_eval_parallel.evaluate(layout)
                self.assertFalse(actual)
                save(layout.root/'endpoints/P-gap.json',dict(update=50))
                gap_eval_parallel.evaluate(layout)
                self.assertEqual(Counter(original),Counter(actual))
                self.assertEqual(len(set(actual)),len(actual))
                report.assert_called_once_with(layout,arms)

    def test_stopping_requires_same_readout_in_two_disjoint_windows(self):
        row=dict(legal=71,actions=100,searches=64,query_tokens=64*15,missed_searches=0)
        first=window_decision([row]*5,.81,{})
        self.assertEqual(first['flags'],dict(legal=True,short_search=False))
        self.assertEqual(first['stop_reasons'],[])
        other=window_decision([dict(row,legal=100,query_tokens=64*14)]*5,.81,first['flags'])
        self.assertEqual(other['stop_reasons'],[])
        hit=window_decision([dict(row,legal=100,query_tokens=64*14)]*5,.81,other['flags'])
        self.assertEqual(hit['stop_reasons'],['short_search'])

    def test_full_distribution_supervision_ignores_invalid_slots(self):
        mask=torch.tensor([[True,True,False]])
        t=torch.tensor([[1.,3.,500.]])
        s=torch.tensor([[4.,0.,300.]])
        q=target(t,mask)
        ts=dict(mask=mask,s=s,teacher=q,q=target(s,mask),h=torch.ones(1,4),ids=torch.arange(3)[None,:])
        h=FitData(ts,{'goal':np.array([1]),'cls':np.array(['search'])},torch.ones(3,4),'cpu')
        self.assertTrue(torch.equal(h.target,q))
        self.assertNotEqual(h.target[0].argmax().item(),h.q[0].argmax().item())
        self.assertIsNone(h.d)
        self.assertEqual(q[0,2].item(),0.)

    def test_nomination_loads_exact_live_weights_and_restores_block(self):
        from unittest.mock import patch
        from .c_smoke import state_hash
        from .gap_targets import exact_nominations
        model=torch.nn.Linear(1,1,bias=False)
        def use(value):
            with torch.no_grad():
                model.weight.fill_(value)
        def signature():
            return state_hash(list(model.named_parameters()))
        use(1.000003)
        current=signature()
        use(1.)
        block=signature()
        def logits(m,row,fixed):
            return m.weight.item()
        with patch('exploratory_shop_audit.visible_opd_v0.gap_targets.nominate',side_effect=logits):
            live=exact_nominations(model,lambda _:use(1.000003),current,block,[(None,None)])
            self.assertNotEqual(live[0],1.)
            self.assertEqual(signature(),block)
            with self.assertRaises(AssertionError):
                exact_nominations(model,lambda _:use(1.),current,block,[(None,None)])
            self.assertEqual(signature(),block)

    def test_training_replay_writes_complete_candidate_receipt(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        from .d_speed import verify_nominations
        from .score import support
        from exploratory_shop_audit.grpo_forced_audit.common import digest_file,read,save
        raw=torch.arange(12,dtype=torch.float32)[None,:]
        ids,mask=support(raw,torch.tensor([0]))
        logits=raw.gather(1,ids)
        row=dict(prompt=[1],answer=[0],ids=ids,mask=mask,snapshot=target(logits,mask))
        batch=SimpleNamespace(rows=[row])
        with TemporaryDirectory() as folder:
            path=Path(folder)
            torch.save([logits],path/'bf16_before.pt')
            save(path/'target_manifest.json',dict(bf16_before_sha256=digest_file(path/'bf16_before.pt')))
            with patch('exploratory_shop_audit.visible_opd_v0.c_targets.response',return_value=(None,raw)):
                before,reused=verify_nominations(None,batch,[0],path,0)
                self.assertFalse(reused)
                self.assertTrue(torch.equal(before[0],logits))
                receipt=read(path/'nomination_replay_0.json')
                self.assertEqual((receipt['records'],receipt['positions'],receipt['max_probability_error']),(1,1,0.))
                row['ids']=row['ids'].roll(1,1)
                with self.assertRaises(AssertionError):
                    verify_nominations(None,batch,[0],path,0)

    def test_operator_hold_permits_current_update_only(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from .gap_run import beyond_operator_hold
        from exploratory_shop_audit.grpo_forced_audit.common import save
        with TemporaryDirectory() as folder:
            layout=SimpleNamespace(root=Path(folder))
            self.assertFalse(beyond_operator_hold(layout,4,'R-gap'))
            save(layout.root/'operator_hold.json',dict(arm='R-gap',after_update=3))
            self.assertFalse(beyond_operator_hold(layout,3,'R-gap'))
            self.assertTrue(beyond_operator_hold(layout,4,'R-gap'))
            self.assertTrue(beyond_operator_hold(layout,1,'P-gap'))

    def test_multiplicity_and_paired_resampling(self):
        self.assertEqual(holm([.03,.01]),[.03,.02])
        null=contrasts({1:1,2:0,3:1},{1:1,2:0,3:1},{1:'A',2:'A',3:'B'})
        self.assertEqual(null['p_two_sided'],1.)
        self.assertEqual(null['ci95'],[0.,0.])
