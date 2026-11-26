"""Cross-sample high-minus-low contrasts within fixed baseline strata."""
from common_io import *

def contrast(length, gain, sample):
    scores=[]
    for j in range(2):
        order=np.argsort(length[sample,j],axis=-1,kind='stable')
        held=np.take_along_axis(gain[sample,1-j],order,axis=-1)
        # Give the middle ordered question half weight in both groups when n is odd.
        n=held.shape[-1]
        h=n//2
        scores.append((held[...,-h:].sum(-1)-held[...,:h].sum(-1))/(n/2))
    return np.mean(scores,axis=0)

def run():
    b,r=archived('B'),archived('R@100')
    levels={v['problem']:v['level'] for v in map(json.loads,Path('/mnt/data2/ForkAudit/math-opsd/math500/test.jsonl').read_text().splitlines())}
    assert [v['problem'] for v in b]==[v['problem'] for v in r] and len(levels)==len(b)==500
    def arr(rows,k):
        return np.array([[v[k] for v in row['generations']] for row in rows],dtype=int)
    length=arr(r,'generated_tokens')-arr(b,'generated_tokens')
    bc=arr(b,'correct')
    gain=arr(r,'correct')-bc
    all_results={}
    for key,group in [('base',bc.mean(1)),('level',np.array([levels[v['problem']] for v in b]))]:
        strata={}; boot=[]; mean=0.
        rng=np.random.default_rng(SEED)
        for g in sorted(set(group)):
            ids=np.flatnonzero(group==g)
            # Order ties by fixed problem text, also within bootstrap replicates.
            ids=np.array(sorted(ids,key=lambda i:b[i]['problem']))
            x,y=length[ids],gain[ids]
            n=len(ids)
            estimate=contrast(x,y,np.arange(n))
            draws=np.concatenate([contrast(x,y,np.sort(rng.integers(n,size=(100,n)),axis=1)) for _ in range(100)])
            boot.append(draws*n/500);mean+=estimate*n/500
            strata[str(g)]=dict(n=n,diff_pp=100*estimate,ci95_pp=(100*np.quantile(draws,[.025,.975])).tolist())
        all_results[key]=dict(strata=strata,weighted_pp=100*mean,ci95_pp=(100*np.quantile(np.sum(boot,axis=0),[.025,.975])).tolist())
    save(HERE/'difficulty.json',all_results)
    print(json.dumps(all_results))
if __name__=='__main__':run()
