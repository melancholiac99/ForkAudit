import unittest
import torch
from objective import contrast_kl


class ContrastTests(unittest.TestCase):
    def test_bounded_target_gradient_and_mass(self):
        torch.manual_seed(42)
        s,a,b=[torch.randn(8,29,dtype=torch.double).requires_grad_() for _ in range(3)]
        mask=torch.arange(8)%2==0
        a.data[:,0]+=24
        b.data[:,1]+=24
        loss,lt,mass=contrast_kl(s,a,b,mask)
        p,q,r=[(x/1.1).softmax(-1) for x in (s,a,b)]
        raw=(q/r).log()/2
        self.assertTrue((raw>3).any() and (raw< -3).any())
        target=p*raw.clamp(-3,3).exp()
        target=torch.where(mask[:,None],target/target.sum(-1,keepdim=True),p)
        torch.testing.assert_close(lt.exp(),target)
        torch.testing.assert_close(lt.exp().sum(-1),torch.ones_like(loss))
        torch.testing.assert_close(mass,(target*(raw.abs()>3)).sum(-1)*mask)
        torch.testing.assert_close(loss,(target*(target/p).log()).sum(-1)*mask)
        loss.sum().backward()
        torch.testing.assert_close(s.grad,(p-target)*mask[:,None]/1.1)
        self.assertEqual(torch.count_nonzero(s.grad[~mask]).item(),0)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)

    def test_equal_teachers(self):
        s=torch.randn(7,20,dtype=torch.double,requires_grad=True)
        a=torch.randn_like(s)
        loss,_,mass=contrast_kl(s,a,a,torch.ones(7,dtype=torch.bool))
        self.assertLess(torch.autograd.grad(loss.sum(),s)[0].abs().max().item(),1e-12)
        self.assertEqual(mass.sum().item(),0)
