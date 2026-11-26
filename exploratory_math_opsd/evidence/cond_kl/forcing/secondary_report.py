from common_io import *
from secondary import MODELS

def run():
    paths=sorted((RAW/'secondary').glob('*.npz'),key=lambda p:int(p.stem))
    assert len(paths)==1500
    with np.load(paths[0]) as z:dim=z['models'].shape[-1]
    inv=np.concatenate([1/z['mass'][z['clipped']] for z in (load_mass(p) for p in paths)])
    edges=np.quantile(inv,[.2,.4,.6,.8])
    n=len(paths);sums=np.zeros((n,7,4,dim));counts=np.zeros((n,7))
    teacher=np.zeros((n,7,dim-2));masses=np.zeros((n,7))
    for i,p in enumerate(paths):
        with np.load(p) as z:
            m=z['clipped'];mass=z['mass'];x=z['models'];t=z['teacher'];bins=np.searchsorted(edges,1/mass,side='right')
            for j,sel in enumerate([~m,m,*[m&(bins==k) for k in range(5)]]):
                counts[i,j]=sel.sum();sums[i,j]=x[sel].sum(0);masses[i,j]=mass[sel].sum()
                teacher[i,j]=np.r_[t[sel].sum(),t[sel].sum(0)]
    assert counts[:,:2].sum()==1185192
    assert np.all(counts[:,1]==counts[:,2:].sum(1))
    def evaluate(w):
        den=w@counts
        vals=(w@sums.reshape(n,-1)).reshape(-1,7,4,dim)/den[:,:,None,None]
        vals[:,:,:,1]=np.exp(vals[:,:,:,1])
        ref=(w@teacher.reshape(n,-1)).reshape(-1,7,dim-2)/den[:,:,None]
        diff=vals[:,:,1:2]-vals
        extra=np.stack([vals[:,:,1,1]/vals[:,:,0,1],(w@masses)/den],axis=-1)
        return np.concatenate([vals.reshape(-1,7,4*dim),diff.reshape(-1,7,4*dim),ref,extra],axis=-1)
    point=evaluate(np.ones((1,n)))[0]
    rng=np.random.default_rng(SEED)
    draws=[]
    for _ in range(100):
        ix=rng.integers(n,size=(100,n))
        w=np.stack([np.bincount(row,minlength=n) for row in ix]).astype(float)
        draws.append(evaluate(w))
    lim=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
    ids=read(HERE/'reflection_ids.json')['included_ids']
    refl=['reflection_total',*['token_'+str(i) for i in ids]]
    names=['top1_matches_teacher2','rank2_5_geom_ratio',*refl]
    keys=[f'{model}/{metric}' for model in MODELS for metric in names]
    keys+=['R_minus_'+v for v in keys]+['teacher/'+v for v in refl]+['R_over_B_rank2_5_ratio','q_U']
    out=dict(inverse_mass_quintile_edges=edges.tolist(),groups={})
    for j,g in enumerate(['unclipped','clipped',*[f'clipped_Q{k}' for k in range(1,6)]]):
        out['groups'][g]=dict(positions=int(counts[:,j].sum()),metrics={k:dict(mean=point[j,h],ci95=lim[:,j,h].tolist()) for h,k in enumerate(keys)})
    save(HERE/'secondary.json',out)
def load_mass(p):
    with np.load(p) as z:return {k:z[k] for k in ('mass','clipped')}
if __name__=='__main__':run()
