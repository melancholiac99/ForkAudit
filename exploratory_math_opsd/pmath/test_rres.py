import unittest
import torch
from rres_teacher import subtract_union

class UnionTest(unittest.TestCase):
    def test_overlap_and_outside(self):
        t=torch.arange(80.).repeat(2,1)
        s=t.clone()
        s[:,12:32]+=100
        h=torch.ones(2,3)
        weights={'A':torch.ones(8,3),'scale':.5,'vocab':torch.ones(80,8)}
        original=t.clone()
        support=torch.zeros_like(t,dtype=torch.bool)
        for logits in (s,t):support.scatter_(1,logits.topk(32).indices,True)
        self.assertTrue((support!=torch.zeros_like(support).scatter_(1,s.topk(64).indices,True)).any())
        subtract_union(t,s,h,weights)
        self.assertTrue(torch.equal(t,original-12*support))
