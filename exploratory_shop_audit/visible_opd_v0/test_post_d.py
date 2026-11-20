import unittest

import torch

from .post_d_gradients import geometry


class AdjacentGradients(unittest.TestCase):
    def test_same_opposite_and_zero(self):
        a = {'u':torch.tensor([3., 0.]), 'v':torch.tensor([-4.])}
        opposite = {key:-2 * v for key, v in a.items()}
        row = geometry(a, opposite)
        self.assertEqual((row['cosine'], row['left_norm'], row['right_norm']), (-1, 5, 10))
        self.assertEqual(geometry(a, a)['cosine'], 1)
        self.assertIsNone(geometry(a, {k:v * 0 for k,v in a.items()})['cosine'])

    def test_mismatched_coordinates(self):
        with self.assertRaises(ValueError):
            geometry({'a':torch.ones(1)}, {'b':torch.ones(1)})


class Reanchoring(unittest.TestCase):
    def test_identity_precision_and_zero_mass(self):
        from .post_d_precision import reanchor
        s=torch.tensor([[1.,2.,-3.,9.]])
        q=torch.tensor([[.2,.8,0.,0.]],dtype=torch.float64)
        mask=torch.tensor([[True,True,True,False]])
        out=reanchor(s,s+11,q,mask)
        self.assertTrue(torch.allclose(out,q,atol=1e-15,rtol=0))
        shifted=s+torch.tensor([[0.,1.,0.,0.]])
        out=reanchor(s,shifted,q,mask)
        expected=(q*torch.tensor([[1.,torch.e,1.,1.]],dtype=torch.float64))
        self.assertTrue(torch.allclose(out,expected/expected.sum(),atol=1e-15,rtol=0))

    def test_target_equal_to_bf16_recovers_fp32(self):
        from .post_d_precision import reanchor
        old=torch.tensor([[-2.,1.,3.]],dtype=torch.float64)
        new=torch.tensor([[3.,7.,0.]],dtype=torch.float64)
        mask=torch.tensor([[True,True,False]])
        soft=lambda v:v.masked_fill(~mask,-torch.inf).softmax(-1)
        self.assertTrue(torch.allclose(reanchor(old,new,soft(old),mask),soft(new),atol=1e-15,rtol=0))
