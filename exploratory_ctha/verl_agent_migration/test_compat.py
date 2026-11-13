"""Check actual compatibility risks: probability anchor, gates and upstream hook."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ['VERL_AGENT_SOURCE'])
import correction


class CorrectionTests(unittest.TestCase):
    def test_identity_and_padding(self):
        x = torch.tensor([[-1., -2., 1000.], [-4., -5., 1000.]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
        r = correction.metrics(x, x, mask)
        self.assertEqual(r['sampled_behavior_k3'], 0)
        self.assertEqual(r['clip_fraction'], 0)
        self.assertAlmostEqual(r['token_ESS_fraction'], 1)

    def test_actual_anchor_once_without_weight(self):
        rollout = torch.tensor([[-2., -3.]])
        actor = rollout + .02
        batch = dict(old_log_probs=actor, rollout_log_probs=rollout,
                     response_mask=torch.ones_like(actor))
        data = SimpleNamespace(batch=batch)
        correction.apply(data, 'native')
        torch.testing.assert_close(batch['old_log_probs'], actor)
        correction.apply(data, 'bypass_ppo_clip')
        torch.testing.assert_close(batch['old_log_probs'], rollout)
        self.assertEqual(set(batch), {'old_log_probs', 'rollout_log_probs', 'response_mask'})

    def test_extreme_ratio_stops_before_update(self):
        data = SimpleNamespace(batch=dict(old_log_probs=torch.tensor([[0.]]),
            rollout_log_probs=torch.tensor([[-5.]]), response_mask=torch.ones(1, 1)))
        with self.assertRaises(AssertionError):
            correction.apply(data, 'bypass_ppo_clip')
        self.assertEqual(data.batch['old_log_probs'].item(), 0.)

    def test_long_reply_and_sequence_ess(self):
        actor = torch.stack([torch.full((300,), .01), torch.full((300,), -.01)])
        result = correction.metrics(actor, torch.zeros_like(actor), torch.ones_like(actor))
        self.assertEqual(result['long_response_count'], 2)
        self.assertLess(result['response_ESS_fraction'], .51)
        self.assertGreater(result['token_ESS_fraction'], .99)

    def test_upstream_hook_and_idempotence(self):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        correction.install()
        installed = RayPPOTrainer.fit
        self.assertTrue(installed._forkaudit_installed)
        correction.install()
        self.assertIs(installed, RayPPOTrainer.fit)


if __name__ == '__main__':
    unittest.main()
