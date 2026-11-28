import unittest
import torch
from format_target import targets,eligible_positions

class TestFormat(unittest.TestCase):
    def test_target_gradient(self):
        rng=torch.Generator().manual_seed(42)
        s,p,n=[torch.randn(4,17,dtype=torch.double,generator=rng).requires_grad_() for _ in range(3)]
        mask=torch.arange(17)%3==0
        lp,q,raw=targets(s,10*p,n,mask,1.1)
        expected=lp.exp().detach().clone()
        weights=expected[:,~mask]*raw[:,~mask].clamp(-3,3).exp()
        expected[:,~mask]=weights/weights.sum(-1,keepdim=True)*expected[:,~mask].sum(-1,keepdim=True)
        torch.testing.assert_close(q.exp(),expected)
        self.assertTrue(torch.equal(q[:,mask],lp.detach()[:,mask]))
        torch.nn.functional.kl_div(lp,q,reduction='sum',log_target=True).backward()
        torch.testing.assert_close(s.grad,(lp.exp()-expected)/1.1)
        self.assertIsNone(p.grad);self.assertIsNone(n.grad)
    def test_first(self):
        self.assertEqual(eligible_positions(torch.tensor([[2,4,-100],[5,-100,-100]])).tolist(),[[False,True,False],[False,False,False]])
