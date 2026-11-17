"""CPU regression checks using the pinned trainer's real serialization path."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
try:
    import torch
    from verl.protocol import DataProto
    from agent_system.multi_turn_rollout.utils import adjust_batch
except ModuleNotFoundError as exc:
    if exc.name not in {'torch', 'verl', 'agent_system'}:
        raise
    raise unittest.SkipTest('Requires the pinned verl environment, with CUDA disabled') from exc
from .native_cache import save_batch


class NativeCacheTests(unittest.TestCase):
    def test_cache_does_not_lock_live_batch_or_change_native_padding(self):
        config = SimpleNamespace(
            trainer=SimpleNamespace(n_gpus_per_node=4, nnodes=1),
            algorithm=SimpleNamespace(use_kl_in_reward=False),
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=4),
                ref=SimpleNamespace(log_prob_micro_batch_size_per_gpu=4),
                actor=SimpleNamespace(use_kl_loss=True, ppo_mini_batch_size=64,
                                      ppo_micro_batch_size_per_gpu=4)))
        for rows in (16, 17, 560):
            with self.subTest(rows=rows), TemporaryDirectory() as tmp:
                tensors = {'responses': torch.arange(rows * 3).reshape(rows, 3),
                           'attention_mask': torch.ones(rows, 6, dtype=torch.long)}
                metadata = {'ids': np.arange(rows, dtype=object)}
                data = DataProto.from_dict(tensors, metadata, {'stage': 'test'})
                original_batch = data.batch
                path = Path(tmp) / 'cache.pkl'
                rng = torch.get_rng_state().clone()
                numpy_rng = np.random.get_state()
                save_batch(data, None, SimpleNamespace(_shop_batch_cache=path), True)
                self.assertIs(data.batch, original_batch)
                self.assertFalse(data.batch.is_locked)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(numpy_rng[0], np.random.get_state()[0])
                np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
                self.assertEqual(numpy_rng[2:], np.random.get_state()[2:])
                restored = DataProto.load_from_disk(str(path))
                self.assertEqual(restored.meta_info, data.meta_info)
                np.testing.assert_array_equal(restored.non_tensor_batch['ids'], metadata['ids'])
                for key, tensor in tensors.items():
                    self.assertTrue(torch.equal(data.batch[key], tensor))
                    self.assertTrue(torch.equal(restored.batch[key], tensor))
                # Match the native random padding against an uncached control.
                np.random.set_state(numpy_rng)
                control = adjust_batch(config, DataProto.from_dict(tensors, metadata))
                np.random.set_state(numpy_rng)
                actual = adjust_batch(config, data)
                for key in tensors:
                    self.assertTrue(torch.equal(control.batch[key], actual.batch[key]))
                np.testing.assert_array_equal(control.non_tensor_batch['ids'], actual.non_tensor_batch['ids'])
                for batch in (actual, restored):
                    batch.batch['response_mask'] = batch.batch['attention_mask'][:, -3:]
                    self.assertTrue(torch.equal(batch.batch['response_mask'], torch.ones(len(batch), 3)))
                # Adding training fields must not change the saved sampling artifact.
                self.assertNotIn('response_mask', DataProto.load_from_disk(str(path)).batch)
                self.assertFalse(path.with_suffix('.tmp').exists())


if __name__ == '__main__':
    unittest.main()
