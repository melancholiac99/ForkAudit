import unittest

import numpy as np
import torch

from .gap_report import contrasts, holm
from .gap_run import window_decision
from .gap_score import target
from .e25_projector import FitData


class GapProtocol(unittest.TestCase):
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

    def test_multiplicity_and_paired_resampling(self):
        self.assertEqual(holm([.03,.01]),[.03,.02])
        null=contrasts({1:1,2:0,3:1},{1:1,2:0,3:1},{1:'A',2:'A',3:'B'})
        self.assertEqual(null['p_two_sided'],1.)
        self.assertEqual(null['ci95'],[0.,0.])
