"""CPU checks for live-state observations, not online RL recovery evidence."""
import copy
import os
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, os.environ['VERL_AGENT_SOURCE'])
from checkpoint_evidence import fingerprint, precision


class StateEvidenceTests(unittest.TestCase):
    def test_momentum_and_step_restore(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .01)
        optimizer.step()
        manager = SimpleNamespace(model=model, optimizer=optimizer, lr_scheduler=scheduler)
        saved = copy.deepcopy(optimizer.state_dict())
        expected = fingerprint(manager)
        state = next(iter(optimizer.state.values()))
        state['exp_avg'].add_(1)
        self.assertNotEqual(fingerprint(manager)['optimizer_sha256'], expected['optimizer_sha256'])
        self.assertEqual(fingerprint(manager)['weights_sha256'], expected['weights_sha256'])
        optimizer.load_state_dict(saved)
        self.assertEqual(fingerprint(manager), expected)
        self.assertEqual(expected['adamw_steps_unique'], [1.])

    def test_bf16_trainable_master_rejected(self):
        with self.assertRaises(AssertionError):
            precision(torch.nn.Linear(2, 2).to(torch.bfloat16))


if __name__ == '__main__':
    unittest.main()
