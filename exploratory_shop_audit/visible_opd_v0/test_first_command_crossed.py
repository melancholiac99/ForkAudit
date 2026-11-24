"""Regression checks for the held-out calibration and the matched T_cal report."""
import unittest

import numpy as np

from exploratory_shop_audit.grpo_forced_audit.common import read
from .first_command_crossed import ARMS, DEST, huber_scale


class CrossedProjectorsTest(unittest.TestCase):
    def test_huber_calibration_ignores_duplicates_and_clips_by_class(self):
        x = np.array([[-1.,1.,900.],[-2.,2.,900.]] * 3)
        meta = {'goal':np.tile([12,25],3), 'cls':np.repeat(['a','b','c'],2)}
        mask = np.tile([True,True,False],(len(x),1))
        y = x * np.repeat([.3,-1.,3.],2)[:,None] + .1
        y[:,2] = -900.
        values = huber_scale(x,y,mask,meta)
        np.testing.assert_allclose([values[c] for c in ('a','b','c')],[.3,0.,1.])

    def test_same_positions_and_student_and_one_heldout_prediction_per_arm(self):
        data = read(DEST/'crossed_projectors.json')
        prior = read(DEST/'tcal_entropy.json')['primary_stage_B_support']
        fits = data['receipts']
        for arm in ('full_huber','content_kl'):
            subset = [r for r in fits if r['arm']==arm]
            self.assertEqual(sorted(r['fold'] for r in subset),list(range(4)))
            self.assertEqual(sum(r['heldout_positions'] for r in subset),34825)
        for name, group in data['groups'].items():
            self.assertEqual(group['positions'],prior[name]['n'])
            self.assertEqual(set(group['targets']),set(ARMS))
            for arm in ARMS:
                values = group['targets'][arm]
                for method,delta in [('entropy_nats','delta_entropy_nats'),('top1','delta_top1')]:
                    ref = prior[name][method]['student']['mean']
                    self.assertAlmostEqual(values[method]['mean']-values[delta]['mean'],ref,places=7)
