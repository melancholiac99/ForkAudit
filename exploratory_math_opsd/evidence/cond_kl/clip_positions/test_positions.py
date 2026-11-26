import unittest
import torch
from positions import boundaries
class TestBoundary(unittest.TestCase):
    def test_prefix(self):
        parts=[b'3',b'.',b'1',b'.',b' ',b'x',b'\n',b' ',b'!',b'X']
        self.assertEqual(boundaries(parts,b'\n').tolist(),[True,False,False,False,True,True,False,True,False,True])
    def test_clamp_gradient(self):
        s=torch.randn(7,20,dtype=torch.float64,requires_grad=True)
        t=torch.randn_like(s);q=t.softmax(-1);k=q*(t.log_softmax(-1)-s.log_softmax(-1))
        on=torch.arange(7)%2==0
        got=k.clamp(max=torch.where(on[:,None],torch.tensor(.05,dtype=k.dtype),torch.inf)).sum()
        expect=k[on].clamp(max=.05).sum()+k[~on].sum()
        torch.testing.assert_close(got,expect)
        g,=torch.autograd.grad(got,s,retain_graph=True);ref,=torch.autograd.grad(expect,s,retain_graph=True)
        torch.testing.assert_close(g,ref)
        raw,=torch.autograd.grad(k.sum(),s)
        qa=q*(k>.05)*on[:,None];torch.testing.assert_close(g-raw,qa-qa.sum(-1,keepdim=True)*s.softmax(-1))
if __name__=='__main__':unittest.main()
