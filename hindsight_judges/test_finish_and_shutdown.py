"""No shutdown/GPU/network calls: test finalizer safety and durable failure archive."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main
from unittest.mock import patch,Mock
import finish_and_shutdown as F


class Tests(TestCase):
    def test_wait_until_queue_exits(self):
        for stage in ('score_32b','failed','completed_verified','completed_published','completed_verified_publication_failed'):
            self.assertEqual(F.classify(dict(stage=stage),True),'wait')
        self.assertEqual(F.classify(dict(stage='completed_published'),False),'verify_success')
        self.assertEqual(F.classify(dict(stage='failed'),False),'recover')

    def test_busy_gpu_never_shutdown(self):
        with patch.object(F,'gpu_idle',return_value=False),patch.object(F.subprocess,'run') as call:
            with self.assertRaises(AssertionError):F.shutdown('test')
            call.assert_not_called()

    def test_sync_before_platform_shutdown(self):
        with TemporaryDirectory() as tmp,patch.object(F,'FIN',Path(tmp)),patch.object(F,'gpu_idle',return_value=True):
            events=[]
            with patch.object(F.os,'sync',side_effect=lambda:events.append('sync')),patch.object(F,'state',side_effect=lambda *a,**k:events.append('state')):
                def execute(args,**kwargs):
                    self.assertEqual(args,['/bin/bash','/usr/bin/shutdown']);self.assertIn('sync',events)
                    events.append('shutdown');return Mock(returncode=0)
                with patch.object(F.subprocess,'run',side_effect=execute):F.shutdown('fixture')
            self.assertEqual(events.count('shutdown'),1)

    def test_failure_snapshot_roundtrip_preserves_originals(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);h=root/'hindsight_judges';run=h/'run';fin=h/'finalization'
            run.mkdir(parents=True);(h/'results').mkdir();(h/'diagnostics').mkdir()
            F.save(run/'status.json',dict(stage='failed'));(run/'error.log').write_text('failure fixture\n')
            (h/'results/partial.jsonl').write_text('{"fixture":true}\n')
            with patch.multiple(F,ROOT=root,H=h,RUN=run,FIN=fin),patch.object(F,'git',return_value='fixture-head'),patch.object(F.os,'sync'):
                dest=F.snapshot_failure('test failure')
            self.assertTrue(F.read(dest/'archive_verified.json')['passed'])
            self.assertEqual((h/'results/partial.jsonl').read_text(),'{"fixture":true}\n')

    def test_failed_audit_hands_off_without_shutdown(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            F.save(run/'launch.json',dict(pid=999999));F.save(run/'status.json',dict(stage='completed_published'))
            with patch.multiple(F,FIN=root/'final',RUN=run),patch.object(F.sys,'argv',['watcher','--thread','fixture']),patch.object(F,'identity',return_value=None),patch.object(F,'local_audit',side_effect=ValueError('bad hash')),patch.object(F,'recovery') as recover,patch.object(F,'shutdown') as shut:
                F.main();recover.assert_called_once();shut.assert_not_called()


if __name__=='__main__':main()
