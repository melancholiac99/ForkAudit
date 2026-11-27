"""Reduced follow-up after training all three arms to 200 updates."""
import os
import sys
from pathlib import Path
import numpy as np
from evaluate import HERE, ROOT, ARMS, paired, run_cells, scores
from launch import evaluate, read, save, sha

NAMES=(*ARMS,'R-extension')
MODELS=[a+'@200' for a in NAMES]
PROBES=ROOT/'run/cond-kl/endpoint200'


def aime():
    receipts={}; ids=None; rates={}
    for model in ('B','R@100','R-extension@200'):
        receipts[model]=evaluate(ROOT,os.environ.copy(),model,'aime24',1001)
        rows=read(ROOT/'run/evaluation'/model/'aime24/1001.json')['results']
        problems=[q['problem'] for q in rows]
        assert ids is None or problems==ids
        ids=problems
        rates[model]=np.array([sum(g['correct'] for g in q['generations']) for q in rows])
    assert receipts[MODELS[-1]]['adapter_sha256']==read(ROOT/'run/R-extension/extended.json')['adapters']['200']
    w=np.random.default_rng(20261124).multinomial(30,np.full(30,1/30),10000)
    differences={}
    for ref in ('B','R@100'):
        hits=rates[MODELS[-1]]-rates[ref]
        differences[ref]=dict(pp=float(hits.sum()*100/360),ci95=np.quantile((w@hits)*100/360,[.025,.975]).tolist())
    below_b=bool(rates[MODELS[-1]].sum()<rates['B'].sum())
    passes=bool(differences['R@100']['ci95'][1]<0 and below_b)
    save(HERE/'r200_aime24.json',dict(receipts=receipts,R200_minus=differences,below_B=below_b,short_term_criterion_met=passes))
    lines=['# R@200 AIME24','',f"最终正确 {rates[MODELS[-1]].sum()}/360，准确率 {receipts[MODELS[-1]]['rate']:.2f}%。"]
    for ref,v in differences.items():lines.append(f"\n相对 {ref} 的差值为 {v['pp']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] 个百分点。")
    lines.append('\n'+('满足登记的短期收益判据。' if passes else '未满足登记的短期收益判据。'))
    (HERE/'R200_AIME24.md').write_text('\n'.join(lines)+'\n')


def probe(lane, names=NAMES, step_number=200):
    sys.path.insert(0,str(HERE.parent))
    import measure
    def checkpoint(arm,step):
        assert step==step_number
        root=ROOT/'run'/arm
        path=root/f'checkpoint-{step}'
        assert sha(path/'adapter_model.safetensors')==read(root/('extended.json' if step==200 else 'complete.json'))['adapters'][str(step)]
        return path
    measure.STEPS=(step_number,)
    measure.WORK=PROBES
    PROBES.mkdir(exist_ok=True)
    measure.checkpoint=checkpoint
    for arm in names:measure.score('fixed',arm,lane)


def collect(names=NAMES, step=200, filename='fixed_at_200.json'):
    from evaluate import effects
    sums=[]; counts=[]
    cut=read(HERE.parents[1]/'stop_decomposition/results.json')['content_u80']
    masks=sorted((ROOT/'run/stop-decomposition-v1').glob('*.npz'))
    assert len(masks)==1500
    for f in masks:
        with np.load(f) as z:
            n=((z['p']<.01)&(z['u']>=cut)).sum()
            s=(z['p']>=.05).sum()
        row=[]
        for a in names:
            with np.load(PROBES/a/('fixed-'+f.name)) as z:
                assert (z['counts'][0,1:3]==[n,s]).all()
                v=z['sums'][0,:,1,:]
                row.append([v[1,4],v[1,2],v[1,3],v[2,0]])
        sums.append(row);counts.append([[n,n,n,s]]*len(names))
    table=paired(np.array(sums),np.array(counts),effects.FIXED,[f'{a}@{step}' for a in names])
    save(HERE/filename,dict(metrics=table,content_positions=int(np.sum(counts,axis=0)[0,0]),stop05_positions=int(np.sum(counts,axis=0)[0,3]),adapters={a:read(ROOT/'run'/a/('extended.json' if step==200 else 'complete.json'))['adapters'][str(step)] for a in names}))


if __name__=='__main__':
    if sys.argv[1]=='aime':
        if (ROOT/'run/R-lambda15/complete.json').exists():
            from midpoint import run as midpoint_run
            midpoint_run()
            from overlap_r200 import run
            run()
        else:
            aime()
    elif sys.argv[1]=='probe':probe(int(sys.argv[2]))
    elif sys.argv[1]=='collect':collect()
    else:raise ValueError(sys.argv)
