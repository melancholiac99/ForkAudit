import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from common import ROOT,WORK,MODEL,json_save
from token_kl_report import classify,GROUPS
from token_kl import TEMPERATURES
from rres_chinese import OUT
from residual_gpu import OUT as BLOCKS

def diagnostics():
    source=OUT.parent/'pmath/token_kl/results.json'
    prior=json.loads(source.read_text())
    total=outside=None
    n=questions=nc=0
    err=mx=0.
    for lane in range(4):
        for j in range(0,375,5):
            x=torch.load(BLOCKS/f'w{lane}-{j}.pt',weights_only=True)
            if total is None:
                total=torch.zeros_like(x['mass'])
                outside=torch.zeros_like(x['outside'])
            total+=x['mass'];outside+=x['outside']
            n+=x['positions'];questions+=x['questions']
            err+=x['label_absolute_sum'];nc+=x['label_count'];mx=max(mx,x['label_max'])
    assert n==prior['positions'] and questions==1500 and torch.isfinite(total).all()
    tok=AutoTokenizer.from_pretrained(MODEL)
    size=len(tok)
    labels=[tok.decode([i],clean_up_tokenization_spaces=False) if i<size else '' for i in range(total.size(-1))]
    groups=torch.tensor([classify(s) for s in labels])
    think=tok.convert_tokens_to_ids('<think>')
    result=dict(positions=n,questions=questions,scale=prior['scale'],temperatures={},
                gpu_reconstruction_centered_label_mae=err/nc,gpu_reconstruction_centered_label_max=mx)
    md=['# Rres 训练前诊断','',f'原四折，{questions:,} 题，{n:,} 位置；λ={prior["scale"]:.8f}。支持集取归档的老师与学生 top-32 的并集，重叠仅减一次，按全词表归一化。每位置先取正 KL 再汇总。R 和 Rres 在四卡原 BF16 路径复算，P 复用此前同口径的题外统计。本次教师前向与原归档的候选中心化标签平均绝对差为 '+f'{err/nc:.6f}，最大 {mx:.6f}。']
    for k,temp in enumerate(TEMPERATURES):
        ref=prior['temperatures'][str(temp)]
        sub={'P':ref['P']}
        for a,name in enumerate(('R','Rres')):
            vec=total[k,a];den=vec.sum().item()
            shares=dict(zip(GROUPS,[vec[groups==g].sum().item()/den for g in range(5)]))
            assert abs(sum(shares.values())-1)<1e-9
            sub[name]=dict(groups=shares,positive_per_position=den/n,
                    outside_C_share=outside[k,a].item()/den,think_share=vec[think].item()/den,
                    top20=[dict(id=i,token=labels[i],share=vec[i].item()/den) for i in vec.topk(20).indices.tolist()])
        sub['ratios_to_R']={g:{name:sub[name]['groups'][g]/sub['R']['groups'][g] for name in ('P','Rres')} for g in GROUPS}
        result['temperatures'][str(temp)]=sub
        md+=['',f'## T={temp}','','|组别|R|P|Rres|P/R|Rres/R|','|---|---:|---:|---:|---:|---:|']
        for g in GROUPS:
            md+=['|'+g+'|'+ '|'.join(f'{sub[name]["groups"][g]:.3%}' for name in ('R','P','Rres'))+'|'+ '|'.join(f'{sub["ratios_to_R"][g][name]:.3f}' for name in ('P','Rres'))+'|']
        md+=['',f'Rres 的 `<think>` 占 {sub["Rres"]["think_share"]:.3%}，C 外 token 占 {sub["Rres"]["outside_C_share"]:.3%}。两者可重叠。','','|排名|R|P|Rres|','|---:|---|---|---|']
        for i in range(20):
            cols=[]
            for name in ('R','P','Rres'):
                item=sub[name]['top20'][i]
                token=json.dumps(item['token'],ensure_ascii=False).replace('|','&#124;')
                cols.append(f'`{token}` {item["share"]:.3%}')
            md+=['|'+str(i+1)+'|'+'|'.join(cols)+'|']
    md+=['','## 归档回答中的汉字','','|模型|含汉字回答|含汉字比例|逐回答汉字占比均值|含汉字和拉丁字母|','|---|---:|---:|---:|---:|']
    for m,data in json.loads((OUT/'han.json').read_text()).items():
        x=data['all'];md+=[f'|{m}|{x["with_han"]}/{x["responses"]}|{x["with_han_fraction"]:.4%}|{x["mean_han_character_fraction"]:.8%}|{x["with_han_and_latin"]}|']
    (OUT/'DIAGNOSTICS.md').write_text('\n'.join(md)+'\n')
    json_save(OUT/'residual_token_kl.json',result)

def scores():
    benches=('aime24','aime25','hmmt25')
    stats={};diff=[];counts={}
    for bench in benches:
        by_arm={}
        stats[bench]={}
        for model in ('R@100','Rres@100'):
            p=ROOT/'run/evaluation'/model/bench/'1001.json'
            rows=json.loads(p.read_text())['results']
            by_arm[model]={r['problem']:r['num_correct']/12 for r in rows}
            gens=[g for r in rows for g in r['generations']]
            assert len(gens)==360 and len(by_arm[model])==30
            stats[bench][model]=dict(accuracy_percent=100*sum(g['correct'] for g in gens)/360,
                mean_generated_tokens=np.mean([g['generated_tokens'] for g in gens]))
        assert set(by_arm['R@100'])==set(by_arm['Rres@100'])
        diff.append([100*(by_arm['Rres@100'][q]-by_arm['R@100'][q]) for q in sorted(by_arm['R@100'])])
    diff=np.array(diff)
    draws=np.random.default_rng(20261124).integers(30,size=(10000,3,30))
    boot=diff[np.arange(3)[None,:,None],draws].mean(-1)
    def pair(x,b):return dict(difference_pp=x.mean(),ci95_pp=np.percentile(b,[2.5,97.5]).tolist())
    for i,b in enumerate(benches):stats[b]['Rres_minus_R']=pair(diff[i],boot[:,i])
    pooled={m:{k:np.mean([stats[b][m][k] for b in benches]) for k in ('accuracy_percent','mean_generated_tokens')} for m in ('R@100','Rres@100')}
    pooled['Rres_minus_R']=pair(diff,boot.mean(1))
    json_save(OUT/'evaluation.json',dict(seed=1001,benchmarks=stats,pooled=pooled))
