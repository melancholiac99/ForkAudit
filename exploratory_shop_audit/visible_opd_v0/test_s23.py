import unittest
from types import SimpleNamespace
import torch
from .s21_data import summarize,balanced_batches

class StageDTests(unittest.TestCase):
    def test_cross_direction_reversal_and_target_independent_motion(self):
        batch=SimpleNamespace(rows=[dict(answer=[1],mask=torch.tensor([[True,True]]))],weights=[1.])
        old={0:torch.zeros(1,2)};new={0:torch.tensor([[.3,0.]])}
        good=summarize(batch,old,new,[torch.tensor([[.8,.2]])],[0])
        bad=summarize(batch,old,new,[torch.tensor([[.2,.8]])],[0])
        self.assertGreater(good['delta_kl'],0)
        self.assertLess(bad['delta_kl'],0)
        self.assertEqual(good['movement'],bad['movement'])
    def test_variable_size_single_epoch(self):
        for n in (282,319,321,707):
            parts=balanced_batches(n,4)
            self.assertEqual(sorted(i for part in parts for i in part),list(range(n)))
            self.assertLessEqual(max(map(len,parts))-min(map(len,parts)),1)
            self.assertEqual(len(parts),(n+63)//64)
