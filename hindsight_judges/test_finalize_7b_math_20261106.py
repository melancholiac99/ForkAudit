import tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import finalize_7b_math_20261106 as G

class Safety(unittest.TestCase):
    def test_both_jobs_and_supplement_required(self):
        with patch.object(G.F,'alive',return_value=True):
            self.assertEqual(G.classify([{'pid':1,'stage':'completed_verified'},{'pid':2,'stage':'waiting'}]),'wait')
            self.assertEqual(G.classify([{'pid':1,'stage':'completed_verified'},{'pid':2,'stage':'completed'}]),'audit')
            self.assertEqual(G.classify([{'pid':1,'stage':'failed'},{'pid':2,'stage':'waiting'}]),'recover')
    def test_crash_is_not_completion(self):
        with patch.object(G.F,'alive',return_value=False):
            self.assertEqual(G.classify([{'pid':1,'stage':'waiting'},{'pid':2,'stage':'completed'}]),'recover')
    def test_busy_or_cancelled_never_calls_shutdown(self):
        with tempfile.TemporaryDirectory() as td,patch.object(G.F,'FIN',Path(td)),patch.object(G.F.subprocess,'run') as run:
            G.F.save(Path(td)/'policy.json',{'authorized':True})
            with patch.object(G.F,'drained',return_value=False):
                with self.assertRaises(AssertionError):G.F.shutdown('test')
            (Path(td)/'cancelled.json').write_text('{}')
            with patch.object(G.F,'drained',return_value=True):
                with self.assertRaises(AssertionError):G.F.shutdown('test')
            run.assert_not_called()
    def test_sync_before_shutdown(self):
        with tempfile.TemporaryDirectory() as td,patch.object(G.F,'FIN',Path(td)),patch.object(G.F,'drained',return_value=True),patch.object(G.F,'state'):
            G.F.save(Path(td)/'policy.json',{'authorized':True});events=[]
            with patch.object(G.F.os,'sync',side_effect=lambda:events.append('sync')):
                def call(args,**kw):
                    self.assertIn('sync',events);self.assertEqual(args,['/bin/bash','/usr/bin/shutdown']);return Mock(returncode=0)
                with patch.object(G.F.subprocess,'run',side_effect=call) as run:G.F.shutdown('test')
                run.assert_called_once()

if __name__=='__main__':unittest.main()
