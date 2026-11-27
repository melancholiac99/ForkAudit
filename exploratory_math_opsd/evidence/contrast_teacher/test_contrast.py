import unittest
import torch
import torch.nn.functional as F
from objective import contrast_kl


class ContrastTests(unittest.TestCase):
    def setUp(self):
        rng = torch.Generator().manual_seed(42)
        self.s, self.a, self.b = [torch.randn(8, 29, generator=rng, dtype=torch.double, requires_grad=True) for _ in range(3)]
        self.mask = torch.arange(8) % 2 == 0

    def test_distribution_and_fallback(self):
        loss, lt = contrast_kl(self.s, self.a, self.b, self.mask)
        p, a, b = [(x/1.1).softmax(-1) for x in (self.s,self.a,self.b)]
        raw = p * (a/b).sqrt()
        target = torch.where(self.mask[:,None], raw/raw.sum(-1,keepdim=True), a)
        torch.testing.assert_close(lt.exp(), target)
        torch.testing.assert_close(lt.exp().sum(-1), torch.ones_like(loss))
        torch.testing.assert_close(loss, (target*(target.log()-p.log())).sum(-1))
        self.assertFalse(lt.requires_grad)
        loss.sum().backward()
        torch.testing.assert_close(self.s.grad, (p-target)/1.1)
        self.assertIsNone(self.a.grad)
        self.assertIsNone(self.b.grad)

    def test_identical_contexts(self):
        loss, _ = contrast_kl(self.s, self.a, self.a, torch.ones_like(self.mask))
        grad, = torch.autograd.grad(loss.sum(),self.s)
        self.assertLess(grad.abs().max().item(), 1e-12)


if __name__ == '__main__':
    unittest.main()
