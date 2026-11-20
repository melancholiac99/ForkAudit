import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from .s21_data import balanced_batches,components,select_multiplier
from .s21_store import ARMS,pack,unpack_member

class StepDiagnostics(unittest.TestCase):
    def test_schedule(self):
        x=balanced_batches(282,1)
        self.assertEqual(list(map(len,x)),[57,57,56,56,56])
        self.assertEqual(sorted(sum(x,[])),list(range(282)))
        self.assertEqual(x,balanced_batches(282,1))
        self.assertNotEqual(x,balanced_batches(282,2))
        for n in (1,64,65,624):
            groups=balanced_batches(n,1)
            self.assertLessEqual(max(map(len,groups)),64)
            self.assertLessEqual(max(map(len,groups))-min(map(len,groups)),1)

    def test_identity_and_asymmetric_gate(self):
        mask=torch.tensor([[True,False,True]])
        old=torch.tensor([[0.,100.,1.]])
        new=torch.tensor([[2.,-200.,0.]])
        q=torch.tensor([[.45,0.,.55]])
        _,_,delta,align,move=components(old,new,q,mask)[0]
        self.assertLess(delta,0);self.assertGreater(align,0)
        self.assertAlmostEqual(delta.item(),(align-move).item(),places=14)
        rows={a:{'0.0':{'error_envelope':.001},**{k:{'alignment':.3,'delta_kl':-2.}
              for k in ('0.125','0.25','0.5','1.0','2.0','4.0')}} for a in ARMS}
        self.assertEqual(select_multiplier(rows),1.)
        rows['P']['1.0']['alignment']=.001
        self.assertEqual(select_multiplier(rows),.5)
        for k in ('0.125','0.25','0.5'):rows['P'][k]['alignment']=0.
        self.assertIsNone(select_multiplier(rows))

    def test_lossless_adapter(self):
        init={'x':torch.randn(13,21)}
        x=(init['x']-torch.randn(13,21)*1e-8).contiguous()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'adapter.zip';pack(p,[('x',x)],init)
            with zipfile.ZipFile(p) as z:
                self.assertTrue(torch.equal(torch.from_numpy(unpack_member(z,'x',init)),x))
