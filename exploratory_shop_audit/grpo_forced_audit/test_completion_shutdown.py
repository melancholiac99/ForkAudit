"""Completion guards must never confuse partial success with permission to stop."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import append, digest_file, save
from .completion_shutdown import completion_gate, poll


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name)

    def complete(self):
        r = self.run
        save(r / 'campaign_status.json', {'status': 'complete', 'training_updates': 50})
        save(r / 'metrics.json', {'completed_final_evaluation': True, 'training_updates': 50})
        (r / 'REPORT.md').write_text('completed')
        save(r / 'splits.json', {k: [{'task_id': 'task'}] for k in ['V', 'E0', 'E1']})
        save(r / 'budget_freeze.json', {'final_repeats': 4, 'E0_tasks': 1, 'E1_tasks': 1})
        checkpoint = r / 'training/checkpoints/global_step_50'
        for name in ['data.pt', 'actor/lora_adapter/adapter_model.safetensors',
                     *[f'actor/{kind}_world_size_4_rank_{rank}.pt' for kind in ['model', 'optim', 'extra_state'] for rank in range(4)]]:
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'saved')
        sha = digest_file(checkpoint / 'actor/lora_adapter/adapter_model.safetensors')
        save(r / 'final_checkpoint.json', {'update': 50, 'sha256': sha})
        save(checkpoint / 'actor/replicated_checkpoint.json', {'all_rank_fingerprints_equal': True})
        for rank in range(4):
            save(checkpoint / f'actor/live_fingerprint_rank_{rank}.json', {'weights': 'same'})
        for label in ['base', 'final']:
            for decode, repeats in [('greedy', 1), ('sample', 4)]:
                folder = r / 'evaluation' / label / decode
                folder.mkdir(parents=True)
                for i in range(repeats):
                    append(folder / 'episodes.jsonl', {'task_id': 'task', 'trajectory_id': f'{label}:{decode}:{i}', 'checkpoint_sha': sha})
        for name in ['audit_base', 'audit_post']:
            save(r / name / 'metrics.json', {'state_records': [{'state_id': 's'}]})
            (r / name / 'natural').mkdir()
            append(r / name / 'natural/episodes.jsonl', {'task_id': 'task'})
            append(r / name / 'states.jsonl', {'state_id': 's', 'actions': ['continue', 'buy'], 'checkpoint_sha': sha})
            for i in [0, 1]:
                append(r / name / 'candidates.jsonl', {'state_id': 's', 'candidate': i, 'terminal': bool(i)})
            for bank in ['a', 'b']:
                (r / name / f'bank_{bank}').mkdir()
                for i in range(3):
                    append(r / name / f'bank_{bank}/episodes.jsonl', {'trajectory_id': f's:arm0:{bank.upper()}:{i}'})

    def test_running_or_failed_never_completes(self):
        self.assertEqual(completion_gate(self.run), ['campaign_not_complete'])
        save(self.run / 'campaign_status.json', {'status': 'failed'})
        self.assertEqual(completion_gate(self.run), ['campaign_not_complete'])

    def test_completed_fixture_and_missing_e1_replay(self):
        self.complete()
        self.assertEqual(completion_gate(self.run), [])
        (self.run / 'audit_post/bank_b/episodes.jsonl').write_text('')
        self.assertIn('incomplete_audit_bank:audit_post:b', completion_gate(self.run))

    def test_duplicate_eval_and_active_lease_block(self):
        self.complete()
        append(self.run / 'evaluation/final/sample/episodes.jsonl', {'task_id': 'task', 'trajectory_id': 'final:sample:0', 'checkpoint_sha': 'wrong'})
        save(self.run / 'active_gpu_sessions/a.json', {})
        reasons = completion_gate(self.run)
        self.assertIn('incomplete_episodes:final:sample', reasons)
        self.assertIn('wrong_checkpoint:sample', reasons)
        self.assertIn('active_gpu_lease', reasons)

    def test_missing_optimizer_blocks(self):
        self.complete()
        (self.run / 'training/checkpoints/global_step_50/actor/optim_world_size_4_rank_2.pt').unlink()
        self.assertTrue(any('missing_checkpoint_component' in x for x in completion_gate(self.run)))

    def test_unarmed_check_never_calls_shutdown(self):
        self.complete()
        with patch('subprocess.run') as command, patch('os.sync') as sync:
            command.return_value.stdout = 'inactive'
            result = poll(self.run, execute=True)
            self.assertIn('system_poweroff_not_armed', result['reasons'])
            self.assertTrue(all(call.args[0][0] == 'systemctl' for call in command.call_args_list))
            sync.assert_not_called()

    def test_current_boot_authorization_is_one_shot_and_check_only_is_safe(self):
        self.complete()
        save(self.run / 'completion_shutdown/request.json', {'researcher_authorized': True,
             'mode': 'system_poweroff', 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
        with patch('subprocess.run') as command, patch('os.sync'):
            command.return_value.stdout = 'inactive'
            command.return_value.stderr = ''
            command.return_value.returncode = 0
            self.assertTrue(poll(self.run, execute=False)['ready'])
            self.assertFalse(any(call.args[0][0] == 'shutdown' for call in command.call_args_list))
            self.assertTrue(poll(self.run, execute=True)['executed'])
            poll(self.run, execute=True)
            self.assertEqual(sum(call.args[0][0] == 'shutdown' for call in command.call_args_list), 1)

    def test_old_boot_authorization_cannot_power_off_restarted_host(self):
        self.complete()
        save(self.run / 'completion_shutdown/request.json', {'researcher_authorized': True,
             'mode': 'system_poweroff', 'boot_id': 'old-boot'})
        with patch('subprocess.run') as command:
            command.return_value.stdout = 'inactive'
            self.assertIn('authorization_not_for_current_boot', poll(self.run, execute=True)['reasons'])
            self.assertFalse(any(call.args[0][0] == 'shutdown' for call in command.call_args_list))


if __name__ == '__main__':
    unittest.main()
