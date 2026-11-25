"""Task-clustered counts of literal reasoning expressions in archived AIME answers."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from common import ROOT,json_save
from rres_chinese import OUT

TERMS=('Wait','But','Hmm','Let me check','verify','double-check','Alternatively')
BENCHES=('aime24','aime25')
MODELS=('B','R@100','P@100','Rres@100')
patterns=[re.compile(r'(?<!\w)'+r'\s+'.join(re.escape(w) for w in term.split())+r'(?!\w)',re.I) for term in TERMS]
def summarize():
    draws=np.random.default_rng(20261124).integers(30,size=(10000,2,30))
    report={};hashes={};orders={}
    for model in MODELS:
        data=[];hashes[model]={}
        for bench in BENCHES:
            p=ROOT/'run/evaluation'/model/bench/'1001.json'
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest==json.loads(p.with_suffix('.receipt.json').read_text())['sha256']
            hashes[model][bench]=digest
            r=json.loads(p.read_text())
            assert r['dataset']==bench and r['val_n']==12 and r['seed']==1001
            rows=sorted(r['results'],key=lambda i:i['problem'])
            keys=[i['problem'] for i in rows]
            assert len(set(keys))==len(keys)==30 and keys==orders.setdefault(bench,keys)
            counts=[]
            for i in rows:
                gs=i['generations'];assert len(gs)==12
                counts.append([sum(len(pattern.findall(g['full_generation'])) for g in gs)/12 for pattern in patterns])
            data.append(counts)
        data=np.asarray(data)
        sample=data[np.arange(2)[None,:,None],draws].mean(axis=2)
        def estimates(mean,dist):
            bounds=np.percentile(dist,[2.5,97.5],axis=0)
            return {t:dict(mean_per_response=mean[j],ci95=bounds[:,j].tolist()) for j,t in enumerate(TERMS)}
        report[model]={b:estimates(data[j].mean(0),sample[:,j]) for j,b in enumerate(BENCHES)}
        report[model]['pooled']=estimates(data.mean((0,1)),sample.mean(1))
    json_save(OUT/'response_terms.json',dict(models=report,sha256=hashes,bootstrap_seed=20261124,draws=10000))
    md=['# AIME 回答中的词语频次','','每个模型 720 条完整生成回答，含思考正文。逐词不区分大小写，按完整词或短语匹配；Let me check 的词间允许空白，double-check 使用字面连字符。先对每题的 12 条回答取平均，再在每个基准的 30 题内按题重抽 10,000 次（种子 20261124），两个基准等权。表中为每条平均次数及 95% 百分位区间。']
    for b in ('pooled',*BENCHES):
        md+=['',f'## {b}','','|词语|'+'|'.join(MODELS)+'|','|---|'+'---:|'*4]
        for term in TERMS:
            cells=[]
            for m in MODELS:
                s=report[m][b][term];lo,hi=s['ci95']
                cells.append(f"{s['mean_per_response']:.3f} [{lo:.3f}, {hi:.3f}]")
            md+=['|'+term+'|'+'|'.join(cells)+'|']
    (OUT/'RESPONSE_TERMS.md').write_text('\n'.join(md)+'\n')
if __name__=='__main__':summarize()
