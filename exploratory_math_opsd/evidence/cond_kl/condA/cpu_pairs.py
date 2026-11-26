"""Paired CPU summaries from the archived responses and natural-EOS interventions."""
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'forcing'),str(HERE.parent/'forcing/budget32'),str(HERE.parents[1]/'reasoning_organization')]
from common_io import RAW,SEED,read,save,archived
from paired import ratio
import analyze as org

def main():
    org.init()
    from evaluate_math import extract_boxed_answer
    result={};lines=['# 同轨迹转移与 R−T0 首答案差','','区间按题配对重抽 10,000 次。转移计数区间对应重新抽取 500 题后的事件数。无自然 EOS 的回答单列。']
    for arm in ('B','T0@100'):
        archive=archived(arm);rows=[]
        for i,q in enumerate(archive):
            for v in read(RAW/'budget32'/arm/f'{i}.json'):
                edits=v['forced_at'];eligible=bool(edits) or v['finish_reason']=='stop'
                if not eligible:rows.append((0,0,0));continue
                ids=v['token_ids'][:edits[0][0]] if edits else v['token_ids']
                initial=org.grade(extract_boxed_answer(org.tok.decode(ids,skip_special_tokens=True)),q['ground_truth'])
                rows.append((1,int(initial),int(v['correct'])))
        eligible,initial,final=np.array(rows).T
        stats=dict(no_natural_eos=int((1-eligible).sum()))
        for name,src,dest in [('correct_to_wrong',initial,1-final),('wrong_to_correct',1-initial,final)]:
            hits=eligible*src*dest; count=int(hits.sum())
            count_ci=ratio(1000*hits,np.ones(1000))['ci95']
            stats[name]=dict(count=count,count_ci95=count_ci,fraction_of_eligible=ratio(hits,eligible),given_initial_state=ratio(hits,eligible*src))
            lines.append(f"{arm} {name}：{count} 条 [{count_ci[0]:.0f}, {count_ci[1]:.0f}]。")
        result[arm]=stats
    root=org.ROOT/'run/first-answer-decomposition'
    arms=[read(root/f'nonthinking-{a}@100.json')['qs'] for a in ('R','T0')]
    assert set(arms[0])==set(arms[1]) and len(arms[0])==500
    delta=np.array([np.array(arms[0][q])-np.array(arms[1][q]) for q in sorted(arms[0])])*100
    ix=np.random.default_rng(SEED).integers(500,size=(10000,500))
    comp={}
    for j,rule in enumerate(('explicit','boxed')):
        for k,metric in [(4,'first_accuracy'),(5,'wrong_to_correct')]:
            x=delta[:,j,k];ci=np.quantile(x[ix].mean(1),[.025,.975])
            comp[rule+'_'+metric]=dict(pp=float(x.mean()),ci95_pp=ci.tolist())
            lines.append(f"R−T0 {rule} {metric}：{x.mean():+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] 个百分点。")
    result['R_minus_T0']=comp
    save(HERE/'cpu_pairs.json',result);(HERE/'CPU_PAIRS.md').write_text('\n\n'.join(lines)+'\n')
if __name__=='__main__':main()
