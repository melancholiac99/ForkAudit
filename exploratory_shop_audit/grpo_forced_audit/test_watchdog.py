"""Operational checks; do not run or restart a model in these tests."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import watchdog
from .common import save, read


def healthy():
    return dict(complete=False, service='active', disk_free_gib=50,
                memory_available_gib=60, gpu_error=None, kernel_fault=False,
                progress_age_seconds=100, heartbeat_age_seconds=20,
                new_error=False, tensorboard_ok=True)


class WatchdogTests(unittest.TestCase):
    def test_healthy_and_completed(self):
        self.assertEqual(watchdog.issues(healthy()), {})
        self.assertEqual(watchdog.issues(dict(healthy(), complete=True, service='inactive')), {})

    def test_thresholds_and_idle_gpus(self):
        row = healthy()
        row.update(disk_free_gib=19, memory_available_gib=7,
                   progress_age_seconds=2401, heartbeat_age_seconds=181)
        self.assertEqual(set(watchdog.issues(row)),
                         {'disk_critical','memory_low','progress_stalled','heartbeat_stalled'})
        self.assertEqual(watchdog.issues(dict(healthy(), gpus=[['0','0','1']]*4)), {})

    def test_confirm_deduplicate_and_rearm(self):
        state = {}
        fault = {'service_down':'failed'}
        self.assertEqual(watchdog.confirmations(state, fault, 0), [])
        self.assertEqual(watchdog.confirmations(state, fault, 60), ['service_down'])
        state['faults']['service_down']['dispatched'] = True
        self.assertEqual(watchdog.confirmations(state, fault, 120), [])
        watchdog.confirmations(state, {}, 180)
        self.assertEqual(watchdog.confirmations(state, fault, 240), [])
        self.assertEqual(watchdog.confirmations(state, fault, 300), ['service_down'])

    def test_tool_timeout_becomes_evidence(self):
        with patch.object(watchdog.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['nvidia-smi'], 15)):
            self.assertEqual(watchdog.command(['nvidia-smi']).returncode, 124)

    def test_dispatch_once_and_existing_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            fake = subprocess.CompletedProcess([], 0, 'inactive\n', '')
            with patch.object(watchdog, 'snapshot', return_value=dict(healthy(), service='failed')), \
                 patch.object(watchdog, 'command', return_value=fake) as command:
                watchdog.poll(run)
                self.assertFalse((run/'watchdog/pending.json').exists())
                watchdog.poll(run)
                incident = read(run/'watchdog/pending.json')
                self.assertEqual(incident['codes'], ['service_down'])
                self.assertIn(['systemctl','start','--no-block','forkaudit-shop-repair.service'],
                              [c.args[0] for c in command.call_args_list])
                watchdog.poll(run)
                self.assertEqual(read(run/'watchdog/pending.json')['id'], incident['id'])
                (run/'watchdog/pending.json').unlink()
                watchdog.poll(run)
                self.assertFalse((run/'watchdog/pending.json').exists())

    def test_cooldown_and_disable(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            save(run/'watchdog/state.json', {'last_dispatch':9900})
            with patch.object(watchdog.time, 'time', return_value=10000), \
                 patch.object(watchdog, 'snapshot', return_value=dict(healthy(), new_error=True)) as snapshot, \
                 patch.object(watchdog, 'command', return_value=subprocess.CompletedProcess([],0,'inactive','')):
                watchdog.poll(run); watchdog.poll(run)
                self.assertFalse((run/'watchdog/pending.json').exists())
                (run/'watchdog/DISABLED').touch()
                snapshot.reset_mock()
                watchdog.poll(run)
                snapshot.assert_not_called()

    def test_interrupted_repair_is_not_repeated(self):
        from . import repair_incident
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            save(run/'watchdog/pending.json', {'id':'test', 'probe':True})
            save(run/'watchdog/repairs/test/status.json', {'status':'running'})
            with patch.object(repair_incident, 'DEFAULT_RUN', run), \
                 patch.object(repair_incident.subprocess, 'run') as execute:
                repair_incident.main()
                execute.assert_not_called()
            self.assertTrue(read(run/'watchdog/last_repair.json')['needs_user'])
            self.assertFalse((run/'watchdog/pending.json').exists())


if __name__ == '__main__':
    unittest.main()
