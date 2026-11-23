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

    def test_multiplicity_and_paired_resampling(self):
        self.assertEqual(holm([.03,.01]),[.03,.02])
        null=contrasts({1:1,2:0,3:1},{1:1,2:0,3:1},{1:'A',2:'A',3:'B'})
        self.assertEqual(null['p_two_sided'],1.)
        self.assertEqual(null['ci95'],[0.,0.])
