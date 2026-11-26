import unittest
import numpy as np
from scipy.stats import spearmanr

from summarize import auc_boot,corr_boot
from experiment import uniform
class TestConfidence(unittest.TestCase):
    def test_cluster_resampling_and_ties(self):
        x=np.array([1.,1.,2.,3.,3.,5.,9.]); y=np.array([0,1,0,1,0,1,0]);g=np.array([0,0,1,1,2,2,2])
        w=np.array([[1.,1.,1.],[2.,0.,1.]])
        v=np.array([6.,5.,5.,3.,4.,3.,1.]);order=np.argsort(x)
        auc=auc_boot(x[order],y[order],g[order],w)
        rho=corr_boot(x,v,g,w)
        for i,weights in enumerate(w):
            ix=np.repeat(np.arange(len(x)),weights[g].astype(int))
            pos=x[ix][y[ix]==1];neg=x[ix][y[ix]==0]
            delta=pos[:,None]-neg[None,:]
            self.assertAlmostEqual(auc[i],np.mean((delta>0)+0.5*(delta==0)))
            self.assertAlmostEqual(rho[i],spearmanr(x[ix],v[ix]).statistic)
    def test_event_offset(self):
        tokens=np.arange(100)
        j=70
        np.testing.assert_array_equal(tokens[j+np.arange(-64,0)],np.arange(6,70))
        self.assertEqual(len(set(uniform(list(tokens),20,np.random.default_rng(1)))),20)
if __name__=='__main__':unittest.main()
