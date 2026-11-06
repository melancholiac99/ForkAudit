import tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import e4d as E

class DurableExecution(unittest.TestCase):
    def test_shards_are_disjoint_stable_and_complete(self):
        rows=[dict(point_id=f's{i}_t03') for i in range(200)]
        a=types.SimpleNamespace(shard_count=2,shard_index=0);b=types.SimpleNamespace(shard_count=2,shard_index=1)
        x={p['point_id'] for p in E.shard_points(rows,a)};y={p['point_id'] for p in E.shard_points(rows,b)}
        self.assertFalse(x&y);self.assertEqual(x|y,{r['point_id'] for r in rows})
        self.assertEqual(x,{r['point_id'] for r in E.shard_points(rows[::-1],a)})
        self.assertTrue(70<len(x)<130)
    def test_interruption_never_exposes_bare_result_and_can_retry(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'result.json';dp={'path':str(target)}
            cont=types.SimpleNamespace(key='x',dp=0,role=1,rep=0,div=0)
            def original(d,cs,prov):E.save_json(d['path'],dict(continuations=[dict(role=1,rep=0)],factual_outcomes=[1],provenance=prov))
            replay=types.SimpleNamespace(_write=original)
            def fail(k):raise RuntimeError('simulated interruption before target log')
            tracker=types.SimpleNamespace(metadata=fail)
            with self.assertRaises(RuntimeError),E.durable_target_writer(replay,tracker):replay._write(dp,[cont],{})
            self.assertFalse(target.exists());self.assertIs(replay._write,original)
            tracker.metadata=lambda k:dict(schema='credit-grok-target-v1',ever_seen=True)
            with E.durable_target_writer(replay,tracker):replay._write(dp,[cont],{})
            E.verify_target_log(E.load(target));saved=target.read_bytes()
            with self.assertRaises(RuntimeError),E.durable_target_writer(replay,tracker):replay._write(dp,[cont],{})
            self.assertEqual(target.read_bytes(),saved)
    def test_divergent_record_is_never_saved(self):
        replay=types.SimpleNamespace(_write=lambda *a:None)
        cont=types.SimpleNamespace(div=1)
        with self.assertRaises(RuntimeError),E.durable_target_writer(replay,None):replay._write({},[cont],{})

class ShutdownGates(unittest.TestCase):
    def test_separate_branches_must_both_finish(self):
        import finalize_server as F
        with patch.object(F,'alive',return_value=True):
            self.assertEqual(F.classify([dict(stage='completed_verified',pid=1),dict(stage='watching',pid=2)]),'wait')
            self.assertEqual(F.classify([dict(stage='completed_verified',pid=1),dict(stage='completed_published_shutdown_deferred',pid=2)]),'audit')
            self.assertEqual(F.classify([dict(stage='failed',pid=1),dict(stage='watching',pid=2)]),'recover')
    def test_busy_or_cancelled_never_shuts_down(self):
        import finalize_server as F
        with tempfile.TemporaryDirectory() as td,patch.object(F,'FIN',Path(td)),patch.object(F.subprocess,'run') as call:
            E.save_json(Path(td)/'policy.json',dict(authorized=True))
            with patch.object(F,'gpu_idle',return_value=False):
                with self.assertRaises(AssertionError):F.shutdown('test')
            (Path(td)/'cancelled.json').write_text('{}')
            with patch.object(F,'gpu_idle',return_value=True),patch.object(F,'live_experiment_processes',return_value=[]):
                with self.assertRaises(AssertionError):F.shutdown('test')
            call.assert_not_called()
    def test_sync_precedes_authorized_shutdown(self):
        import finalize_server as F
        with tempfile.TemporaryDirectory() as td,patch.object(F,'FIN',Path(td)),patch.object(F,'state'),patch.object(F,'gpu_idle',return_value=True),patch.object(F,'live_experiment_processes',return_value=[]):
            E.save_json(Path(td)/'policy.json',dict(authorized=True));events=[]
            with patch.object(F.os,'sync',side_effect=lambda:events.append('sync')):
                def call(args,**kw):
                    self.assertEqual(args,['/bin/bash','/usr/bin/shutdown']);self.assertIn('sync',events)
                    return types.SimpleNamespace(returncode=0)
                with patch.object(F.subprocess,'run',side_effect=call) as execute:F.shutdown('test')
                execute.assert_called_once()

if __name__=='__main__':unittest.main()
