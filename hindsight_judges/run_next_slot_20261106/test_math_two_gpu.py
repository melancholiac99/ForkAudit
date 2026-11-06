import unittest
import numpy as np
from math_two_gpu_worker import split_questions,compare_probes

class Scheduling(unittest.TestCase):
    def test_partition_and_paired_questions(self):
        rows=[{'id':str(i)} for i in range(17)]
        parts=split_questions(rows,['0','1','8'])
        self.assertEqual(set(parts['primary'])|set(parts['helper']),{r['id'] for r in rows})
        self.assertFalse(set(parts['primary'])&set(parts['helper']))
        self.assertTrue({'0','1','8'}<=set(parts['primary']))
        self.assertLessEqual(abs(len(parts['primary'])-len(parts['helper'])),1)
        self.assertEqual(parts,split_questions(rows,['0','1','8']))
    def test_cross_card_gate_rejects_tiny_sampling_differences(self):
        x=np.log(np.array([[.3,.7]],dtype='float32'))
        self.assertTrue(compare_probes(x,x.copy())['passed'])
        y=x.copy();y[0,0]+=1e-6
        out=compare_probes(x,y)
        self.assertLess(out['tv'][0],.005);self.assertFalse(out['passed'])
    def test_invalid_probes(self):
        with self.assertRaises(AssertionError):compare_probes(np.array([[np.nan]]),np.array([[0.]]))
        with self.assertRaises(AssertionError):compare_probes(np.zeros((1,2)),np.zeros((2,2)))

if __name__=='__main__':unittest.main()
