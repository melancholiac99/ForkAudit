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
