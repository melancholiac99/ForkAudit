import unittest
import numpy as np
from scipy.stats import spearmanr

from summarize import auc_boot,corr_boot
from experiment import uniform, fit, read
from tempfile import TemporaryDirectory
from pathlib import Path
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
    def test_layer_selection_uses_only_training_contrast(self):
        with TemporaryDirectory() as folder:
            d=Path(folder)
            e=np.array([0.,1.,8.,9.,2.,5.,7.,3.])
            v=np.arange(8)>=4
            h=np.broadcast_to(e[:,None,None],(8,2,1)).copy()
            natural=h.copy();natural[:,1]*=-1
            # A validation-only perturbation must not influence the direction.
            h[v]*=-1000
            np.savez(d/'rows.npz',ec=e,h=h,natural=natural,valid=v,mass=e*0+.1)
            fit([d/'rows.npz'],d)
            res=read(d/'fit.json')
            self.assertEqual(res['layer_index_zero_based'],0)
            np.testing.assert_allclose(res['correlations'],[1.,-1.])
            self.assertEqual(res['train'],4)
            np.testing.assert_allclose(np.load(d/'direction.npy'),[1.])
    def test_event_offset(self):
        tokens=np.arange(100)
        j=70
        np.testing.assert_array_equal(tokens[j+np.arange(-64,0)],np.arange(6,70))
        self.assertEqual(len(set(uniform(list(tokens),20,np.random.default_rng(1)))),20)
if __name__=='__main__':unittest.main()
