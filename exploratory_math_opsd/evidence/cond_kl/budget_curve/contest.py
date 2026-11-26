"""Score archived contest responses, conditionally generate the remaining cells."""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from launch import evaluate, read, save, sha
sys.path.insert(0,str(HERE.parents[1]/'reasoning_organization'))
import analyze as org

ROOT=org.ROOT
BOUNDARY='R-clip-boundary@100'
INTERIOR='R-clip-interior@100'
BENCHES=('aime24','aime25','hmmt25')
METRICS=('final','first','wrong_to_correct','correct_to_wrong','first_coverage')


def measure(arm,bench,env):
    # org.init selects CPU for the grader. The evaluator receives the original GPU environment.
    receipt=evaluate(ROOT,env,arm,bench,1001)
    data=read(ROOT/'run/evaluation'/arm/bench/'1001.json')
    ids=[]; rows=[]
    for q in data['results']:
        ids.append(q['problem'])
        samples=[]
        for g in q['generations']:
            txt=g['full_generation']
            item=org.first(txt,org.box(txt))
            final=int(g['correct'])
            first=int(bool(item) and (final if item[1]==g['predicted_answer'] else org.grade(item[1],q['ground_truth'])))
            samples.append([final,first,(1-first)*final,first*(1-final),int(bool(item))])
        assert len(samples)==12
        rows.append(np.mean(samples,axis=0))
    a=100*np.array(rows)
    np.testing.assert_allclose(a[:,0],a[:,1]+a[:,2]-a[:,3],atol=1e-12)
    return ids,a,receipt


def summarize(draws,arm):
    views={key:value for key,value in zip(('B','R@100',arm),draws)}
    views.update({arm+' minus '+key:draws[2]-draws[i] for i,key in enumerate(('B','R@100'))})
    return {key:{m:dict(pp=float(v[0,j]),ci95=np.quantile(v[1:,j],[.025,.975]).tolist())
                 for j,m in enumerate(METRICS)} for key,v in views.items()}


def publish(arm,result):
    dest=HERE/('boundary_contests' if arm==BOUNDARY else 'interior_contests')
    save(dest.with_suffix('.json'),result)
    lines=['# '+arm+' 思考模式比赛','',
           '各基准 30 题，每题 12 条。首答案为 explicit 口径；最终正确率 = 首答案正确率 + 先错后对 − 先对后错。按题配对重抽 10,000 次，种子 20261124。']
    for name,table in result['scores'].items():
        lines+=['','## '+name,'','|模型或配对差|最终|首答案|先错后对|先对后错|覆盖率|','|---|---:|---:|---:|---:|---:|']
        for key,row in table.items():
            cells=['%.2f [%.2f, %.2f]'%(row[m]['pp'],*row[m]['ci95']) for m in METRICS]
            lines.append('|'+key+'|'+'|'.join(cells)+'|')
    if 'decision' in result:lines+=['','运行判定 `'+json.dumps(result['decision'],ensure_ascii=False)+'`']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def run(arm,env,benches=BENCHES):
    ck=ROOT/'run'/arm.split('@')[0]
    assert sha(ck/'checkpoint-100/adapter_model.safetensors')==read(ck/'complete.json')['adapters']['100']
    rng=np.random.default_rng(20261124)
    total=[]; result=dict(scores={},receipts={})
    for bench in benches:
        cells=[measure(a,bench,env) for a in ('B','R@100',arm)]
        assert cells[0][0]==cells[1][0]==cells[2][0] and len(set(cells[0][0]))==30
        raw=np.stack([c[1] for c in cells]);result['receipts'][bench]=[c[2] for c in cells]
        w=np.vstack([np.ones(30),rng.multinomial(30,np.full(30,1/30),10000)])/30
        draws=w@raw
        total.append(draws)
        result['scores'][bench]=summarize(draws,arm)
        if arm==BOUNDARY and bench=='aime24':
            # Integer comparison includes exact ties.
            hits=[round(r[:,0].sum()*12/100) for r in raw]
            result['aime24_gate']=dict(boundary_correct=hits[2],base_correct=hits[0],passed=hits[2]>=hits[0])
            if hits[2]<hits[0]:
                result['decision']=dict(not_run=list(BENCHES[1:]),reason='AIME24 below B')
                publish(arm,result);return
        publish(arm,result)
    if benches!=BENCHES:
        return
    all_scores=summarize(np.mean(total,axis=0),arm)
    result['scores']['equal_weight']=all_scores
    if arm==BOUNDARY:
        lower,upper=all_scores[arm+' minus R@100']['final']['ci95']
        diff=all_scores[arm+' minus B']['final']['pp']
        decision=dict(noninferior_to_R=lower> -1.5,below_R=upper<0,delta_from_B_pp=diff)
        rule=HERE/'interior_gate.json'
        if rule.exists():
            threshold=read(rule)['abs_pp_max']
            decision.update(abs_pp_max=threshold,interior_trigger=upper<0 and abs(diff)<=threshold+1e-12)
        else:
            decision['interior_status']='needs researcher decision' if upper<0 else 'not triggered'
        result['decision']=decision
    publish(arm,result)
    if result.get('decision',{}).get('interior_trigger'):
        run(INTERIOR,env)


if __name__=='__main__':
    env=os.environ.copy()
    # analyze.py sets CUDA_VISIBLE_DEVICES='' on import; four_engines assigns
    # CUDA_VISIBLE_DEVICES individually to each of its four evaluator subprocesses.
    org.init()
    if sys.argv[1:]==['interior-aime24']:
        run(INTERIOR,env,('aime24',))
    elif not sys.argv[1:]:
        run(BOUNDARY,env)
    else:
        raise SystemExit('Expected no arguments or interior-aime24')
