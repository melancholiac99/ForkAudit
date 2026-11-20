import unittest

import torch

from .b_metrics import p1_status
from .b_ridge import expanded_grid
from .c_smoke import kl_positions


class DiagnosticsTests(unittest.TestCase):
    def test_grid_cap_and_reselection(self):
        for direction in (1,-1):
            result=expanded_grid(100,lambda alphas:{a:direction*a for a in alphas})
            self.assertEqual(result['expansions'],3)
            self.assertTrue(result['at_boundary'])
            self.assertEqual(result['alpha'],result['grid'][0 if direction==1 else -1])
            self.assertIn(str(result['alpha']),result['inner_mse'])
        result=expanded_grid(100,lambda alphas:{a:abs(a-100) for a in alphas})
        self.assertEqual((result['alpha'],result['expansions'],result['at_boundary']),(100,0,False))

    def test_power_only_limits_negative_inference(self):
        self.assertEqual([p1_status(d,p) for d in (True,False) for p in (True,False)],
                         ['PASS','PASS_LOW_POWER','FAIL_POWERED','UNDERPOWERED'])

    def test_first_order_kl_and_empty_control(self):
        x=torch.tensor([[.75,.5,5.],[.1,.2,-.5]],requires_grad=True)
        m=torch.tensor([[True,True,False],[True,True,True]])
        target=torch.tensor([[.6,.4,0.],[.3,.4,.3]])
        v,prob=kl_positions(x,target,m)
        ref=x.masked_fill(~m,-torch.inf).log_softmax(1).masked_fill(~m,0)
        expected=(torch.special.xlogy(target,target)-target*ref).sum(1)
        self.assertTrue(torch.allclose(v.float(),expected,atol=1e-6))
        grad=torch.autograd.grad(v.sum(),x)[0]
        self.assertTrue(torch.allclose(grad,torch.autograd.grad(expected.sum(),x)[0],atol=1e-7))
        v,_=kl_positions(x,prob,m)
        self.assertEqual(torch.autograd.grad(v.sum(),x)[0].abs().max(),0)

if __name__=='__main__':unittest.main()
