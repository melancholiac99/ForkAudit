"""Question-paired bootstrap and prefix audit against original decoded archives."""
import numpy as np
from transformers import AutoTokenizer
from common_io import ROOT, SEED, RAW, read, archived

def ratio(numerator,denominator):
    n=np.asarray(numerator,float).reshape(500,2).sum(1)
    d=np.asarray(denominator,float).reshape(500,2).sum(1)
    if d.sum()==0:return dict(mean=None,ci95=None)
    rng=np.random.default_rng(SEED); draws=[]
    for _ in range(100):
        ix=rng.integers(500,size=(100,500))
        den=d[ix].sum(1);ok=den>0
        draws.extend(n[ix].sum(1)[ok]/den[ok])
    return dict(mean=float(n.sum()/d.sum()),ci95=np.quantile(draws,[.025,.975]).tolist())

def supplement(result,lines):
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    eos=set(read(ROOT/'models/Qwen3-1.7B/generation_config.json')['eos_token_id'])
    rg=archived('R@100')
    r=np.array([v['correct'] for q in rg for v in q['generations']],bool)
    lines+=['','## 首次追加、转移及长度合格门','',
        '原回答仅存解码文本、长度、停止原因和判分，原始 token ID 相同比例不可识别。下面另核对首次追加前的自然 token 解码文本及其与原文本重新分词的一致性；无追加时比完整自然文本。']
    for arm in ('B','T0@100'):
        orig=[v for q in archived(arm) for v in q['generations']]
        new=[v for i in range(500) for v in read(RAW/'budget32'/arm/f'{i}.json')]
        text_eq=[];canonical=[];end_match=[]
        for o,v in zip(orig,new):
            ids=v['token_ids'];edits=v['forced_at']
            n=edits[0][0] if edits else len(ids)-int(ids[-1] in eos)
            prefix=ids[:n];txt=o['full_generation']
            text_eq.append(tok.decode(prefix,skip_special_tokens=True,clean_up_tokenization_spaces=False)==txt)
            canonical.append(prefix==tok.encode(txt,add_special_tokens=False))
            end_match.append((n+1==o['generated_tokens'] and o['finish_reason']=='stop') if edits else (len(ids)==o['generated_tokens'] and o['finish_reason']==v['finish_reason']))
        base=np.array([v['correct'] for v in orig],bool); forced=np.array([v['correct'] for v in new],bool)
        length=np.array([v['tokens'] for v in new]);L=np.array([v['target'] for v in new])
        edits=np.array([v['insertions'] for v in new]);added=edits>0
        gain=length-np.array([v['generated_tokens'] for v in orig])
        ones=np.ones(1000);ok=length*10>=L*9
        stats=dict(original_native_id_match=None,text_match=ratio(text_eq,ones),
            canonical_id_match=ratio(canonical,ones),text_and_stop_position_match=ratio(np.array(text_eq)&end_match,ones),
            truncated=ratio(length==16384,ones),tokens_per_insertion=ratio(gain*added,edits),
            mean_answer_tokens_per_insertion=ratio(gain/np.maximum(edits,1)*added,added),
            mean_length_ratio=float(length.sum()/L.sum()))
        for name,condition,event in [('correct_to_wrong',base,~forced),('wrong_to_correct',~base,forced)]:
            hit=condition&event
            stats[name]=dict(count=int(hit.sum()),from_count=int(condition.sum()),overall=ratio(hit,ones),conditional=ratio(hit,condition))
        if arm=='B':
            mask=r&~base
            stats['R_correct_B_wrong']=dict(count=int(mask.sum()),rescued_count=int((mask&forced).sum()),
                rescued=ratio(mask&forced,mask),length_minus_L=ratio(mask*(length-L),mask),matched=ratio(mask&ok,mask))
        passed=length.sum()>=.9*L.sum() and ok.mean()>=.8
        rr=result['contrasts_pp'][arm+'-强制32 − R@100']['all_questions']['ci95']
        own=result['contrasts_pp'][arm+'-强制32 − '+arm]['all_questions']['ci95']
        verdict='匹配不合格，仅作部分证据'
        if passed:
            verdict=('收益主要来自推理时间（登记判读）' if rr[0]<=0<=rr[1] and own[0]>0 else
                     '首答案之后加长不能替代 R 的收益（须核对追加时机及原前缀）' if rr[1]<0 else '无定论')
        stats.update(length_match_pass=bool(passed),verdict=verdict)
        result['models'][arm]['supplement']=stats
        def fmt(x,scale=1):
            if x['mean'] is None:return '无样本'
            return ('%.2f [%.2f, %.2f]' % tuple(scale*z for z in [x['mean'],*x['ci95']]))
        lines+=['',f'### {arm}','',f"平均长度 / 平均 L 为 {stats['mean_length_ratio']:.3f}。{verdict}。",
                '与原回答文本相同（%）为 '+fmt(stats['text_match'],100)+'；与原文本重分词后 ID 相同（%）为 '+fmt(stats['canonical_id_match'],100)+'。',
                '达到 16,384 的比例（%）为 '+fmt(stats['truncated'],100)+'；每次追加净增长 token（合计净增长 / 合计追加）为 '+fmt(stats['tokens_per_insertion'])+'。']
        for k,label in [('correct_to_wrong','对→错'),('wrong_to_correct','错→对')]:
            v=stats[k];lines.append(f"{label}共 {v['count']} 条，占全部回答（%）{fmt(v['overall'],100)}，占原状态（%）{fmt(v['conditional'],100)}。")
        from first_forced import attach
        first_audit=attach(arm,stats,lines)
        if passed and rr[1]<0 and any(v['inserted_before_first_count'] or v['original_text_equal_before_edit']['mean']<1 for v in first_audit.values()):
            stats['interpretation_limit']='部分追加发生在首次可判分答案前或原回答文本未复现；不能把总体对照解释为纯粹首答案之后加长。'
            lines.append(stats['interpretation_limit'])
        if arm=='B':
            v=stats['R_correct_B_wrong'];lines.append(f"R 对、原 B 错共 {v['count']} 条，强制后改对 {v['rescued_count']} 条，改对率（%）{fmt(v['rescued'],100)}；平均长度减 L 为 {fmt(v['length_minus_L'])}，达标率（%）为 {fmt(v['matched'],100)}。")
