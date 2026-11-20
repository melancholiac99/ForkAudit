import unittest
from types import SimpleNamespace
import numpy as np
import torch

from .s22_measure import matched_null,prefix_length
from .s22_monitor import cumulative_window
from .s22_report import fit_curve,gate


class MeasurementTests(unittest.TestCase):
    def test_norm_seed_and_precision(self):
        init={'A':torch.tensor([1.,.25]),'B':torch.zeros(19)}
        a=matched_null(init,.035,42);b=matched_null(init,.035,42);other=matched_null(init,.035,43)
        self.assertTrue(all(torch.equal(a[n],b[n]) for n in a));self.assertFalse(torch.equal(a['B'],other['B']))
        n2=sum((a[n].double()-v.double()).square().sum() for n,v in init.items())
        self.assertAlmostEqual(n2.sqrt().item(),.035,places=7)
        self.assertEqual(prefix_length([1,2],[1,2]),2);self.assertEqual(prefix_length([2],[1,2]),0)

    def test_strict_gate_and_curve(self):
        samples=[0.,.1,.2,.3,.4]
        q=float(np.quantile(samples,.95))
        self.assertFalse(gate(q,samples)['passed']);self.assertTrue(gate(q+1e-10,samples)['passed'])
        self.assertAlmostEqual(fit_curve([1,4,16],[3,9,33])['slope'],2.)

    def test_probe_window_continuity(self):
        batch=SimpleNamespace(rows=[dict(answer=[0],mask=torch.tensor([[True,True]]))],weights=[1.])
        q=[torch.tensor([[.5,.5]])];updates=[]
        for u in range(10):
            updates.append(dict(number=1+u,precision='fp32',probe_before={0:torch.tensor([[.01*u,0.]])},
                                probe_after={0:torch.tensor([[.01*(u+1),0.]])},
                                fixed_target=dict(delta_kl=.1,alignment=.3,movement=.2)))
        result=cumulative_window(updates,batch,[0],q)
        self.assertAlmostEqual(result['sum_fixed_target_delta'],1.)
        self.assertGreater(result['start_to_end_probe_movement'],result['sum_adjacent_probe_movement'])
        updates[3]['number']=9
        with self.assertRaises(ValueError):cumulative_window(updates,batch,[0],q)
