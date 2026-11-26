from common_io import *
from budget import MODELS

def run():
    labels=['B','R@100','B-forced','R-cond-forced']
    baseline=[archived(m) for m in labels[:2]]
    target=np.array([[g['generated_tokens'] for g in row['generations']] for row in baseline[1]])
    scores=[np.array([[g['correct'] for g in row['generations']] for row in rows]).mean(1) for rows in baseline]
    rng=np.random.default_rng(SEED)
    draws=rng.integers(500,size=(10000,500))
    def interval(x):
        x=x.mean(1) if x.ndim==2 else x
        return dict(mean=float(x.mean()),ci95=np.quantile(x[draws].mean(1),[.025,.975]).tolist())
    out={}
    for arm in MODELS:
        raw=[read(RAW/arm/f'{i}.json') for i in range(500)]
        def field(k):return np.array([[g[k] for g in row] for row in raw])
        assert np.array_equal(field('target'),target)
        length=field('tokens');ins=field('insertions');score=field('correct').mean(1)
        scores.append(score)
        correlations=[]
        for chunk in np.array_split(draws,100):
            a=length[chunk].reshape(len(chunk),-1);b=target[chunk].reshape(len(chunk),-1)
            a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
            correlations.extend((a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1)))
        out[arm]=dict(accuracy=interval(score),length=interval(length),minus_target=interval(length-target),
              correlation=dict(mean=np.corrcoef(length.ravel(),target.ravel())[0,1],ci95=np.quantile(correlations,[.025,.975]).tolist()),
              insertions=np.bincount(ins.flatten(),minlength=9).tolist(),mean_absolute_length_error=interval(abs(length-target)))
    out['contrasts_pp']={f'{labels[a]} minus {labels[b]}':interval(100*(scores[a]-scores[b])) for a,b in ((2,0),(2,1),(3,2),(3,1))}
    save(HERE/'budget.json',out)
if __name__=='__main__':run()
