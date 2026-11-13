import unittest
from rollout_session import Session

class SessionTests(unittest.TestCase):
    def make(self):
        calls=[]
        return Session(lambda:calls.append('enter'),lambda *args:calls.append('exit')),calls
    def test_legacy_every_call(self):
        s,c=self.make()
        for _ in range(4):s.enter();s.leave()
        self.assertEqual(c,['enter','exit']*4);s.assert_closed()
    def test_many_turns_one_transition_and_update_guard(self):
        s,c=self.make();s.begin()
        for _ in range(4):s.enter();s.leave()
        with self.assertRaises(AssertionError):s.assert_closed()
        self.assertEqual(c,['enter']);s.end();s.assert_closed()
        self.assertEqual(c,['enter','exit'])
        s.begin();s.enter();s.end();self.assertEqual(c,['enter','exit']*2)
    def test_exception_cleanup(self):
        s,c=self.make();s.begin();s.enter();s.leave(ValueError,ValueError('x'),None);s.end()
        self.assertEqual(c,['enter','exit']);s.assert_closed()
    def test_failed_cleanup_blocks_updates(self):
        def fail(*args):raise RuntimeError('cleanup failure')
        s=Session(lambda:None,fail);s.begin();s.enter()
        with self.assertRaises(RuntimeError):s.end()
        with self.assertRaises(AssertionError):s.assert_closed()
    def test_nested_and_empty_session(self):
        s,c=self.make();s.begin()
        with self.assertRaises(AssertionError):s.begin()
        s.end();s.assert_closed();self.assertEqual(c,[])


class CollectorCleanupTests(unittest.TestCase):
    def test_collector_exception_ends_remote_session(self):
        import os
        from unittest.mock import patch
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
        import rollout_session as module
        calls=[]
        def control(group,fn):
            calls.append(fn.__name__)
            return dict(enters=0,exits=0)
        def fail(*args,**kwargs):raise ValueError('environment failure')
        with patch.object(TrajectoryCollector,'vanilla_multi_turn_loop',fail),patch.object(module,'control',control),patch.dict(os.environ,{},clear=True):
            module.install_collector()
            with self.assertRaises(ValueError):TrajectoryCollector.vanilla_multi_turn_loop(None,None,None,None)
        self.assertEqual(calls,['begin','end'])

if __name__=='__main__':unittest.main()
