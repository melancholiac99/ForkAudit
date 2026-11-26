import unittest

import torch
from objective import row_loss, target


class ObjectiveTest(unittest.TestCase):
    def test_loss_and_gradient(self):
        torch.manual_seed(42)
        s = torch.randn(9, 37, dtype=torch.double, requires_grad=True)
        t = torch.randn_like(s)
        p = (s/1.1).softmax(-1)
        q = (t/1.1).softmax(-1)
        ref = q*(q.log()-p.log())
        for conditional in (False, True):
            v, _ = row_loss(s, t, [], conditional, 1.1)
            torch.testing.assert_close(v, ref)
            torch.testing.assert_close(torch.autograd.grad(v.sum(),s)[0], (p-q)/1.1)
            self.assertTrue(bool((v.sum(-1)>=0).all()))
        ids = [2,5,8]
        goal, *_ = target(p.log().detach(),q.log(),ids,True)
        v, _ = row_loss(s,t,ids,True,1.1)
        grad = torch.autograd.grad(v.sum()/len(s),s)[0]
        torch.testing.assert_close(grad,(p-goal.exp())/(1.1*len(s)))
        mask = torch.ones(37,dtype=torch.bool)
        mask[ids]=False
        for k in (mask,~mask):
            self.assertLess(float(grad[:,k].sum(-1).abs().max()),1e-6)

    def test_guard(self):
        s=torch.tensor([[50.,0.,0.],[0.,1.,2.]],requires_grad=True)
        t=torch.tensor([[1.,2.,3.],[50.,0.,0.]])
        v,stats=row_loss(s,t,[0],True,1.1)
        self.assertEqual(float(stats[-1]),2)
        torch.testing.assert_close(torch.autograd.grad(v.sum(),s)[0],torch.zeros_like(s),atol=1e-6,rtol=0)

if __name__=='__main__':
    unittest.main()
