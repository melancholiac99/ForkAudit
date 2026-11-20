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

    def test_behavior_uses_episode_denominators_and_token_weighted_entropy(self):
        from .d_behavior import readout
        episode = dict(reward=1., events={'buy':1}, length=1, reason='gold_purchase',
                       tokens={'click':1}, entropy_sum={'click':.5})
        alternate = dict(episode,reason='valid_alternative_purchase',tokens={'option':3},entropy_sum={'option':4.5})
        result,by_task = readout([[episode,alternate]])
        self.assertEqual(result['all_requirements'],1)
        self.assertEqual(result['strict'],.5)
        self.assertEqual(result['strict_ratio'],2.)
        self.assertEqual(result['decision_entropy'],1.25)
        self.assertEqual(by_task.shape,(1,4))

    def test_product_cluster_resamples_paired_task_differences(self):
        import numpy as np
        from .d_behavior import cluster_draws
        data = np.array([[1.],[3.],[5.]])
        got = cluster_draws(data,['a','a','b'],61)
        self.assertEqual(set(np.unique(got)),{2.,3.,5.})
