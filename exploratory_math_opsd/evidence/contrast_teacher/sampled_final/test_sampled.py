import unittest
import torch
from sampled import pg
class Tests(unittest.TestCase):
    def test_sign_unclipped_advantage_detached_cap(self):
        s=torch.tensor([-1.,-2.,-3.],requires_grad=True)
        p=torch.tensor([-1.,-15.,-2.],requires_grad=True)
        n=torch.tensor([-21.,-1.,-2.],requires_grad=True)
        old=s.detach()-torch.tensor([2.,0.,1.])
        out,hit=pg(s,p,n,old);out.sum().backward()
        torch.testing.assert_close(s.grad,torch.tensor([-20.,7.,0.]))
        self.assertEqual(hit.tolist(),[True,False,True]);self.assertIsNone(p.grad);self.assertIsNone(n.grad)
