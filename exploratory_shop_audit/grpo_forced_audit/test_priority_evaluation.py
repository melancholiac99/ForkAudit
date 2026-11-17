"""CPU checks for boundary gating, exact output provenance and resumability."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import INTERFACE, append, digest_file, read, save
from .priority_evaluation import inference_devices, queue, run_pending, validate_outputs, validate_request


class PriorityEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        save(self.run / 'splits.json', {'V': [{'task_id': 't', 'goal_index': 1}]})
        for name in ['PROTOCOL.md', 'snapshots/product_prices_frozen.json', 'snapshots/goals_frozen.json',
                     'training/checkpoints/global_step_30/actor/lora_adapter/adapter_model.safetensors',
                     'training/checkpoints/global_step_30/actor/lora_adapter/adapter_config.json']:
            path = self.run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('frozen')
        self.request = queue(self.run)

    def boundary(self):
        append(self.run / 'cost_ledger.jsonl', {'event': 'end', 'stage': 'train_0040', 'exit_code': 0})
        root = self.run / 'training/checkpoints/global_step_40'
        for name in ['data.pt', 'actor/lora_adapter/adapter_model.safetensors',
                     'actor/model_world_size_4_rank_0.pt', 'actor/optim_world_size_4_rank_0.pt',
                     'actor/extra_state_world_size_4_rank_0.pt']:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'saved')
        save(self.run / 'active_gpu_sessions/a.json', {'stage': 'train_0050', 'session_id': 'a', 'gpu_ids': [0, 1, 2, 3]})

    def test_request_is_idempotent_and_frozen(self):
        self.assertEqual(self.request, queue(self.run))
        self.assertEqual(read(self.run / 'priority_evaluation/status.json')['status'], 'queued')
        with self.assertRaises(AssertionError):
            queue(self.run, checkpoint_update=40)

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3'})
    def test_requires_finished_stage_and_optimizer(self):
        with self.assertRaises(AssertionError):
            validate_request(self.run, self.request)
        self.boundary()
        validate_request(self.run, self.request)
        (self.run / 'training/checkpoints/global_step_40/actor/optim_world_size_4_rank_0.pt').unlink()
        with self.assertRaises(AssertionError):
            validate_request(self.run, self.request)

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3'})
    def test_refuses_changed_input_or_wrong_gpu_lease(self):
        self.boundary()
        path = self.run / 'PROTOCOL.md'
        path.write_text('changed')
        with self.assertRaises(AssertionError):
            validate_request(self.run, self.request)
        path.write_text('frozen')
        save(self.run / 'active_gpu_sessions/b.json', {'gpu_ids': [1]})
        with self.assertRaises(AssertionError):
            validate_request(self.run, self.request)

    def write_outputs(self):
        from .rollout import jobs
        for decode, repeats in [('greedy', 1), ('sample', 4)]:
            folder = self.run / 'evaluation/base' / decode
            folder.mkdir(parents=True, exist_ok=True)
            for job in jobs(read(self.run / 'splits.json')['V'], 'V', 'base', repeats, decode):
                ep = {**job, 'checkpoint_sha': 'base-sha', 'decode': decode, 'length': 1}
                append(folder / 'episodes.jsonl', ep)
                append(folder / 'steps.jsonl', {**job, 'checkpoint_sha': 'base-sha', 'interface_version': INTERFACE,
                                               'step': 0, 'done': True, 'episode_result': ep})

    def test_complete_outputs_reject_duplicates_and_wrong_checkpoint(self):
        self.write_outputs()
        self.assertEqual(validate_outputs(self.run, self.request, 'base', 'base-sha')['sample']['episodes'], 4)
        with self.assertRaises(AssertionError):
            validate_outputs(self.run, self.request, 'base', 'wrong')
        from .common import read_rows
        path = self.run / 'evaluation/base/sample/episodes.jsonl'
        append(path, read_rows(path)[0])
        with self.assertRaises(AssertionError):
            validate_outputs(self.run, self.request, 'base', 'base-sha')

    def test_missing_steps_rejected(self):
        self.write_outputs()
        (self.run / 'evaluation/base/sample/steps.jsonl').write_text('')
        with self.assertRaises(AssertionError):
            validate_outputs(self.run, self.request, 'base', 'base-sha')

    @patch.dict(os.environ, {'SHOP_GPU_IDS': '2', 'SHOP_GPU_ID': '2'})
    def test_device_environment_is_restored_after_failure(self):
        with self.assertRaises(RuntimeError):
            with inference_devices([0, 1, 2, 3]):
                self.assertEqual(os.environ['SHOP_GPU_IDS'], '0,1,2,3')
                raise RuntimeError('test')
        self.assertEqual(os.environ['SHOP_GPU_IDS'], '2')

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3'})
    def test_failure_stays_pending_and_completion_is_one_shot(self):
        self.boundary()
        module = 'exploratory_shop_audit.grpo_forced_audit'
        with patch(module + '.rollout.evaluate', side_effect=RuntimeError('test failure')):
            with self.assertRaises(RuntimeError):
                run_pending(self.run)
        self.assertEqual(read(self.run / 'priority_evaluation/status.json')['status'], 'failed')
        with patch(module + '.rollout.evaluate') as evaluate, \
                patch(module + '.priority_evaluation.validate_outputs', return_value={}), \
                patch(module + '.priority_evaluation.analyze'):
            self.assertTrue(run_pending(self.run))
            self.assertEqual(evaluate.call_count, 2)
            self.assertFalse(run_pending(self.run))
            self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(len((self.run / 'cost_ledger.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
