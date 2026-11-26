"""Post-hoc AIME24 factorial and paired explicit-answer timing analysis."""
import hashlib
import sys
from pathlib import Path
import numpy as np
from contest import HERE,ROOT,BOUNDARY,INTERIOR,org,save,read,sha
sys.path.insert(0,str(HERE.parents[1]/'reasoning_organization'))
import first_scores

ARMS=('B','R-noclip@100',BOUNDARY,INTERIOR,'R@100','R-cond@100')


def main():
    org.init()
    ids=None; models={}; sources={}
    for name in ARMS:
        folder=ROOT/'run/evaluation'/name/'aime24'
        digest=sha(folder/'1001.json')
        assert digest==read(folder/'1001.receipt.json')['sha256']
        cache=first_scores.RAW/f'aime24-{name}.json'
        if not cache.is_file():first_scores.measure(('aime24',name))
        data=read(cache);assert digest==data['sha']
        rows=read(folder/'1001.json')['results']
        keys=[hashlib.sha256(q['problem'].encode()).hexdigest() for q in rows]
        assert set(keys)==set(data['qs']) and len(keys)==30 and all(len(q['generations'])==12 for q in rows)
        assert ids is None or keys==ids
        ids=keys
        models[name]=np.array([data['qs'][k][0] for k in keys])
        sources[name]=digest
    rng=np.random.default_rng(20261124)
    w=np.vstack([np.ones(30),rng.multinomial(30,np.full(30,1/30),10000)])/30
    def calc(v):
        a=w@v
        return [float(a[0]),*np.quantile(a[1:],[.025,.975]).tolist()]
    n,b,i,r=[models[a] for a in ARMS[1:5]]
    factorial={k:calc(100*a[:,7]) for k,a in [('boundary_main',((b-n)+(r-i))/2),
                ('interior_main',((i-n)+(r-b))/2),('interaction',(r-i)-(b-n))]}
    columns=('first','wrong_right','right_wrong','final')
    decomp={}
    for ref in ('R-noclip@100','R-cond@100',BOUNDARY):
        diff=(i-models[ref])[:,4:8]*100
        np.testing.assert_allclose(diff[:,0]+diff[:,1]-diff[:,2],diff[:,3],atol=1e-12)
        decomp[ref]={k:calc(diff[:,j]) for j,k in enumerate(columns)}
    timing={a:{k:calc(models[a][:,col]) for k,col in [('before',1),('after',2),('coverage_percent',3)]} for a in ARMS[:5]}
    for a in timing:timing[a]['coverage_percent']=[v*100 for v in timing[a]['coverage_percent']]
    result=dict(factorial_pp=factorial,interior_minus_pp=decomp,timing_tokens=timing,archive_sha256=sources)
    save(HERE/'aime24_factorial.json',result)
    def fmt(x):return '%.2f [%.2f, %.2f]'%tuple(x)
    lines=['# AIME24 思考模式：位置截断与首答案（事后 CPU 分析）','',
           '30 题，每题 12 次。先题内平均，再按题配对重抽 10,000 次（种子 20261124）。准确率差以百分点计，括号内为 95% 百分位区间。主效应是另一因素两档下的平均效应，交互项为双截 − 仅句内 − 仅边界 + 无截断。','','|效应|估计 [95% 区间]|','|---|---:|']
    for k,v in factorial.items():lines.append('|'+k+'|'+fmt(v)+'|')
    lines+=['','|句内组减对照|首答案正确|先错后对|先对后错|最终正确|','|---|---:|---:|---:|---:|']
    for k,v in decomp.items():lines.append('|'+k+'|'+'|'.join(fmt(x) for x in v.values())+'|')
    lines+=['','|模型|首答案前 token|首答案后 token|explicit 检出率 %|','|---|---:|---:|---:|']
    for k,v in timing.items():lines.append('|'+k+'|'+'|'.join(fmt(x) for x in v.values())+'|')
    lines+=['','使用原分词器重新编码归档文本。前段包含提出首个 explicit 答案的最后一个 token；未检出首答案时，前段为整条回答、后段为零，首答案正确记零。']
    (HERE/'aime24_factorial.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
