"""CPU tests for isolation, task schedule and native-advantage preservation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent))
from native_p1 import add_auxiliary, auxiliary, schedule_task
from run import select_games


class P1Tests(unittest.TestCase):
    def test_full_schedule_and_tune_repetitions(self):
        games=json.loads((HERE.parents[1]/'exploratory_E4/games.json').read_text())
        schedule=dict(train=[select_games(games['train'],0,i,16) for i in range(12)],tune=games['tune'])
        for iteration in range(12):
            selected=[schedule_task(schedule,i,iteration)[0]['game_file'] for i in range(16)]
            self.assertEqual(len(set(selected)),16)
            self.assertTrue(set(selected).isdisjoint(g['game_file'] for g in games['tune']))
        for evaluation in range(2):
            selected=[schedule_task(schedule,1000+i,evaluation*5+j) for j in range(5) for i in range(16)]
            self.assertEqual(len(selected),80)
            for game in games['tune']:
                self.assertEqual(sorted(rep for g,_,rep in selected if g['game_file']==game['game_file']),[0,1])
        with self.assertRaises(AssertionError):schedule_task(schedule,0,12)

    def test_auxiliary_preserves_native_base_and_masks(self):
        original=torch.tensor([[2.,3.,0.],[-1.,0.,0.],[0.,0.,0.]])
        data=SimpleNamespace(batch=dict(advantages=original,returns=original,response_mask=torch.tensor([[1,1,0],[1,0,0],[1,1,1]])),
            non_tensor_batch=dict(forkaudit_auxiliary=np.array([.3,-.2,0.])))
        add_auxiliary(data)
        torch.testing.assert_close(original,torch.tensor([[2.,3.,0.],[-1.,0.,0.],[0.,0.,0.]]),rtol=0,atol=0)
        torch.testing.assert_close(data.batch['advantages'],torch.tensor([[2.3,3.3,0.],[-1.2,0.,0.],[0.,0.,0.]]))
        torch.testing.assert_close(data.batch['returns'],data.batch['advantages'])

    def test_zero_auxiliary_exact_native_identity(self):
        x=torch.randn(5,9)
        data=SimpleNamespace(batch=dict(advantages=x.clone(),returns=x.clone(),response_mask=torch.ones_like(x)),non_tensor_batch=dict(forkaudit_auxiliary=np.zeros(5)))
        add_auxiliary(data)
        self.assertTrue(torch.equal(data.batch['advantages'],x))

    def test_tune_cannot_enter_fit(self):
        plan=json.loads((HERE/'PLAN.json').read_text())
        with patch('prepare.validate_train'):
            with self.assertRaises(AssertionError):
                auxiliary([], [dict(game_file='tune',provenance=dict(split='tune'))],plan,0)

    def test_auxiliary_discards_custom_base_and_anneals(self):
        plan=json.loads((HERE/'PLAN.json').read_text())
        rows=[dict(episode_id='e',turn=1,advantage_habit=2.,advantage=999.,advantage_episode=999.)]
        with patch('prepare.validate_train'),patch('habits_v2.make_batch',return_value=(rows,{})),patch('habits_v2.oof_audit',return_value={}):
            values,report=auxiliary([],[],plan,10)
        self.assertAlmostEqual(values['e',1],.3)
        self.assertTrue(report['discarded_custom_base_advantage'])


if __name__=='__main__':unittest.main()
