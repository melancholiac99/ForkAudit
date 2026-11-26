"""Paired evaluation, registered 16k gates, and archival receipts."""
import json
import sys
from pathlib import Path
import numpy as np
from train_lambda import ROOT, HERE, ARMS, train
from launch import read, save, sha
import math500_long as long
sys.path.insert(0,str(HERE.parent/'clip_positions'))
import effects

REFS=('R-noclip','R-cond','R')
LONG_OUT=long.OUT
SHORT_OUT=ROOT/'run/math500-lambda-4k-n2'
SEED=20261124


def paired(sums,den,metrics,names):
    n=len(sums)
    rng=np.random.default_rng(SEED)
    keys=[(i,j) for i,a in enumerate(names) if a.split('@')[0] in ARMS for j,b in enumerate(names)
          if b.split('@')[0] in (*REFS,'R-extension')]
    labels=[*names,*[names[i]+' minus '+names[j] for i,j in keys]]
    draws=[]
    for batch in range(101):
        weights=np.ones((1,n)) if batch==0 else rng.multinomial(n,np.full(n,1/n),100)
        vals=((weights@sums.reshape(n,-1))/(weights@den.reshape(n,-1))).reshape(len(weights),len(names),-1)
        diffs=[vals[:,i:i+1]-vals[:,j:j+1] for i,j in keys]
        draws.append(np.concatenate([vals,*diffs],axis=1))
    vals=np.concatenate(draws)
    ci=np.quantile(vals[1:],[.025,.975],axis=0)
    return {a:{m:dict(mean=float(vals[0,i,k]),ci95=ci[:,i,k].tolist()) for k,m in enumerate(metrics)}
            for i,a in enumerate(labels)}


def run_cells(names,budget):
    rows,hashes=long.prepare()
    for model in names:
        a,k=model.split('@')
        path=ROOT/'run'/a
        digest=sha(path/f'checkpoint-{k}/adapter_model.safetensors')
        receipt=read(path/('extended.json' if int(k)>100 else 'complete.json'))
        assert digest==(receipt.get('adapters',{}).get(k) or receipt['adapter_sha256'])
        hashes[model]=digest
    long.prepare=lambda:(rows,hashes)
    long.CELLS=[(False,name) for name in names]
    long.OUT=LONG_OUT if budget==16384 else SHORT_OUT
    long.REPORT=HERE/f'receipts-{budget}-{names[0]}.json'
    long.main(budget)


def scores(names,budget):
    sys.path.insert(0,str(HERE.parents[1]/'reasoning_organization'))
    import analyze as org
    org.init()
    base=LONG_OUT if budget==16384 else SHORT_OUT
    order=None
    per_model=[]
    for model in names:
        folder=base/('nonthinking-'+model)
        assert sha(folder/'answers.json')==read(folder/'receipt.json')['answers_sha256']
        rows=read(folder/'answers.json')['results']
        questions=[q['problem'] for q in rows]
        assert len(set(questions))==len(rows)==500 and (order is None or order==questions)
        order=questions
        by_question=[]
        for q in rows:
            assert len(q['generations'])==2
            samples=[]
            for g in q['generations']:
                txt=g['full_generation']
                item=org.first(txt,org.box(txt))
                initial=int(bool(item) and org.grade(item[1],q['ground_truth']))
                final=int(g['correct'])
                samples.append([100*final,g['generated_tokens'],100*(g['finish_reason']=='length'),
                                100*initial,100*(1-initial)*final,100*initial*(1-final)])
            by_question.append(np.mean(samples,axis=0))
        per_model.append(by_question)
    data=np.transpose(per_model,(1,0,2))
    return data,paired(data,np.ones_like(data),('accuracy_pp','tokens','truncated_pct','first_accuracy_pp',
                                                 'wrong_to_correct_pp','correct_to_wrong_pp'),names)


def dynamics():
    result={}
    for arm in ARMS:
        by_step={}
        for step in (25,50,75,100):
            rows=[]
            for rank in range(4):
                p=ROOT/'run'/arm/f'lambda-rank-{rank}.jsonl'
                batch=[x for line in p.read_text().splitlines() if (x:=json.loads(line))['step']==step]
                assert len(batch)==2,(arm,rank,step,len(batch))
                rows.extend(batch)
            sums=np.sum([x['sums'] for x in rows],axis=0)
            n=sum(x['count'] for x in rows)
            by_step[step]=dict(positions=n,mean_size=sums[0]/n,mean_qa=sums[1]/n,mean_pa=sums[2]/n,skipped=int(sums[3]))
        result[arm]=by_step
    return result


def main():
    arms=(*ARMS,*REFS)
    keys=[a+'@100' for a in arms]
    fixed=effects.fixed_data(arms,lambda x,d,m:paired(x,d,m,arms))
    run_cells(keys,16384)
    base,metrics=scores(keys,16384)
    qualified=[]
    for a in ARMS:
        lo,hi=metrics[a+'@100 minus R@100']['accuracy_pp']['ci95']
        if lo<=0<=hi:
            qualified.append(a)
    reports=dict(training=dynamics(),fixed=fixed,math16k=metrics,qualified=qualified)
    save(HERE/'results.json',reports)
    if qualified:
        short=[a+'@100' for a in (*qualified,*REFS)]
        run_cells(short,4096)
        reports['math4k']=scores(short,4096)[1]
        save(HERE/'results.json',reports)
        for arm in [*qualified,'R']:
            train(arm,200)
        for k in (150,200):
            models=[a+'@'+str(k) for a in [*qualified,'R-extension']]
            run_cells(models,16384)
            later,table=scores(models,16384)
            difference=later[:,:,0:1]-base[:,[keys.index(a+'@100') for a in [*qualified,'R']],0:1]
            reports[f'math16k_{k}']=table
            reports[f'change_from_100_{k}']=paired(difference,np.ones_like(difference),('accuracy_change_pp',),models)
            save(HERE/'results.json',reports)
    lines=['# λ 族结果','','16k 配对区间含零的组：'+(', '.join(qualified) or '无')]
    for name,table in reports.items():
        if name=='qualified':continue
        lines+=['','## '+name,'']
        for arm,values in table.items():
            if name=='training':
                for k,v in values.items(): lines.append(f'- {arm}@{k} '+json.dumps(v))
            else:
                lines+=['- '+arm+' '+', '.join(f"{m} {v['mean']:.5g} [{v['ci95'][0]:.5g}, {v['ci95'][1]:.5g}]" for m,v in values.items())]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
