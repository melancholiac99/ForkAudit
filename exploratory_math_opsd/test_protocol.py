import unittest
import numpy as np
from exploratory_math_opsd.paired_report import compare,decision,flags


class ProtocolTests(unittest.TestCase):
    def test_numeric_thresholds_and_priority(self):
        self.assertEqual(flags((0,2)),dict(positive=False,negative=False,small_precise=True))
        self.assertFalse(flags((-2.0001,0))['small_precise'])
        base={'R-B':(.1,3),'T0-B':(.3,1),'R-T0':(.1,1.5)}
        self.assertEqual(decision(base),'P_COMPARISON')
        self.assertEqual(decision({**base,'R-T0':(-1,1)}),'SMALL_PRECISE')
        self.assertEqual(decision({**base,'R-T0':(-3,-1)}),'REFERENCE_HURTS')

    def test_pairing_and_one_seed(self):
        fixed=np.tile(np.arange(30)/30,(3,1,1))
        estimates,label=compare({'B':fixed,'R':fixed+.1,'T0':fixed+.1})
        self.assertEqual(label,'SMALL_PRECISE')
        rt=estimates['R-T0']['macro']
        self.assertEqual(rt['paired_ci95_pp'],[0,0])
        self.assertIsNone(rt['decoding_se_pp'])
        np.testing.assert_allclose(estimates['R-B']['macro']['paired_ci95_pp'],[10,10])

    def test_budget_boundary_and_reserved_seeds(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from exploratory_math_opsd import launch
        for elapsed,expected in [(1800,21600),(1801,15120)]:
            with TemporaryDirectory() as tmp:
                folder=Path(tmp)
                with patch.object(launch,'evaluate',return_value={'rate':10.,'wall_seconds':elapsed}) as ev, \
                        patch('exploratory_math_opsd.paired_report.summarize',return_value={}):
                    launch.evaluations(folder,{},False)
                calls={tuple(x.args[2:]) for x in ev.call_args_list}
                self.assertEqual(len(calls)*360,expected)
                for arm in ('B','R@100','T0@100'):
                    self.assertEqual({s for m,_,s in calls if m==arm},set(launch.SEEDS))
                self.assertEqual({s for m,_,s in calls if m=='R@50'},set(launch.SEEDS) if expected==21600 else {1001})
