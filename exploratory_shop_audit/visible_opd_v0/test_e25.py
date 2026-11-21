import unittest
import torch
from .e25_probe import full_metrics
from .e25_layout import Layout
from .e25_offline import divergence

class StageE(unittest.TestCase):
    def test_full_vocab_kl_and_direction(self):
        base=torch.tensor([[0.,.1,-3.,4.],[0.,2.,-.9,1.]])
        end=base+torch.tensor([0.,0.,0.,1.])
        p=base.double().softmax(-1);q=end.double().softmax(-1)
        actual=full_metrics(base,end)
        self.assertTrue(torch.allclose(actual[:,0],(p*(p.log()-q.log())).sum(-1)))
        self.assertTrue(torch.allclose(actual[:,2],q.max(-1).values))
        self.assertTrue((actual[:,0]>0).all())

    def test_refresh_used_only_after_its_cutoff(self):
        layout=Layout('formal')
        for k,version in ((1,0),(10,0),(11,10),(20,10),(100,90)):
            self.assertEqual(layout.version(k,'P'),version)
            self.assertEqual(layout.version(k,'R'),0)
        self.assertRaises(ValueError,Layout,'../formal')

    def test_forward_kl_ignores_invalid_support(self):
        import numpy as np
        s=np.array([[0.,1.,500.]])
        mask=np.array([[True,True,False]])
        q=np.array([[.5,.5,0.]])
        val=divergence(s,q,mask,np.array([[0.,-1.,100.]]),1.)
        self.assertAlmostEqual(float(val[0]),0.)
