"""Guard tests: never execute real shutdown or network calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main
from unittest.mock import patch,Mock
import completion_guard as G

class Tests(TestCase):
    def test_cancel_and_busy_gpu_block_shutdown(self):
        with patch.object(G,'authorized',return_value=False),patch.object(G,'state'),patch.object(G.subprocess,'run') as run:
            G.shutdown('test');run.assert_not_called()
        with patch.object(G,'authorized',return_value=True),patch.object(G,'idle',return_value=False),patch.object(G.subprocess,'run') as run:
            with self.assertRaises(AssertionError):G.shutdown('test')
            run.assert_not_called()

    def test_live_supervisor_blocks_shutdown(self):
        with patch.object(G,'authorized',return_value=True),patch.object(G,'idle',return_value=True),patch.object(G,'read',return_value=dict(pid=123)),patch.object(G,'identity',return_value='alive'),patch.object(G.subprocess,'run') as run:
            with self.assertRaises(AssertionError):G.shutdown('test')
            run.assert_not_called()

    def test_sync_precedes_platform_shutdown(self):
        with TemporaryDirectory() as tmp,patch.object(G,'FIN',Path(tmp)),patch.object(G,'authorized',return_value=True),patch.object(G,'idle',return_value=True),patch.object(G,'read',return_value=dict(pid=123)),patch.object(G,'identity',return_value=None),patch.object(G,'state'):
            events=[]
            with patch.object(G.os,'sync',side_effect=lambda:events.append('sync')):
                def execute(args,**kw):
                    self.assertEqual(args,['/bin/bash','/usr/bin/shutdown']);self.assertIn('sync',events)
                    events.append('shutdown');return Mock(returncode=0)
                with patch.object(G.subprocess,'run',side_effect=execute):G.shutdown('test')
            self.assertEqual(events.count('shutdown'),1)

    def test_failure_archive_roundtrip(self):
        with TemporaryDirectory() as tmp:
            h=Path(tmp);run=h/'run';run.mkdir();(h/'data').mkdir()
            G.save(run/'status.json',dict(stage='failed'));(h/'data/partial.json').write_text('{"partial":true}\n')
            with patch.multiple(G,H=h,ROOT=h,RUN=run,FIN=h/'finalization'),patch.object(G,'git',return_value='fixture'),patch.object(G.os,'sync'):
                dest=G.preserve('fixture failure')
            self.assertTrue(G.read(dest/'archive_verified.json')['passed'])
            self.assertEqual((h/'data/partial.json').read_text(),'{"partial":true}\n')

if __name__=='__main__':main()
