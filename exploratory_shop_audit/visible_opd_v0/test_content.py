import unittest
import numpy as np
import torch
from .content_target import content_correction
from .projection import candidate_target
from .b_fit import shuffle

class Tests(unittest.TestCase):
    def test_three_distinct_views_and_zero_control(self):
        s=torch.tensor([[3.,2.,7.],[1.,7.,2.]])
        sel=torch.tensor([[1.,5.,3.],[8.,1.,3.]],requires_grad=True)
        ph=torch.tensor([[2.,1.,1.],[2.,3.,6.]])
        mask=torch.tensor([[True,True,False],[True,True,True]])
        qs=candidate_target(s,torch.zeros_like(s),1.,mask)
        d=content_correction(sel,ph,qs)
        observed=candidate_target(s,d,1.,mask)
        expected=(s+sel-ph).masked_fill(~mask,-torch.inf).softmax(-1)
        self.assertTrue(torch.allclose(observed,expected))
        self.assertFalse(torch.allclose(observed,sel.masked_fill(~mask,-torch.inf).softmax(-1)))
        self.assertFalse(d.requires_grad)
        self.assertTrue(torch.equal(content_correction(ph,ph,qs),torch.zeros_like(qs)))
        self.assertTrue(torch.allclose((qs*d).sum(-1),torch.zeros(2),atol=1e-6))
    def test_shuffle_donor_partition(self):
        x=np.array([[1.,-1.],[2.,-2.],[90.,-90.]])
        valid=np.ones((3,2),bool);q=np.full((3,2),.5)
        meta={'page':np.array(['p']*3),'cls':np.array(['click']*3),'goal':np.arange(3)}
        a,v,_=shuffle(x,q,valid,meta,[0,1],np.random.default_rng(4))
        self.assertTrue(np.array_equal(a[:2],x[[1,0]]))
        self.assertFalse(v[2])
