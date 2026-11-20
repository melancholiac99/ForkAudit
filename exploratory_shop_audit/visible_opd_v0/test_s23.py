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

    def test_context_na_keeps_physical_partition_and_valid_token_weight(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from .d_data import OnlineBatch
        from exploratory_shop_audit.grpo_forced_audit.common import digest_file,save
        rows=[dict(goal=g,rep=rep,step=0,prompt=[1],answer=[2],ids=torch.tensor([[2]]),
                   mask=torch.tensor([[True]]),classes=['search'],snapshot=torch.ones(1,1))
              for g in range(32) for rep in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            for kind,val in [('student',rows),('target',[r['snapshot'] for r in rows])]:
                torch.save(val,path/f'{kind}.pt')
            save(path/'target_manifest.json',dict(student_sha256=digest_file(path/'student.pt'),
                 target_sha256=digest_file(path/'target.pt'),goals=list(range(32)),rollout_records=65,
                 na_records=[{'goal':0,'rep':0,'step':1}]))
            with patch('exploratory_shop_audit.visible_opd_v0.d_data.folder',return_value=path):
                batch=OnlineBatch(2,'P')
            self.assertEqual(batch.batch_sizes,[33,32])
            self.assertEqual(sorted(i for part in batch.batches for i in part),list(range(64)))
            self.assertEqual(sum(batch.weights),1.)
