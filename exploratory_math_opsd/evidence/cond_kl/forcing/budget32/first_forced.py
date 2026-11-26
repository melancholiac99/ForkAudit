"""Pair first responses with the natural prefix immediately before the first EOS edit."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'reasoning_organization'))
import analyze as org
from paired import ratio
from common_io import RAW,read,archived,save

def check(arm):
    org.init()
    from evaluate_math import extract_boxed_answer
    base=archived(arm)
    natural_final=[]
    values={k:[] for k in ('explicit','boxed')}
    for i,q in enumerate(base):
        for o,v in zip(q['generations'],read(RAW/'budget32'/arm/f'{i}.json')):
            ids=v['token_ids'];edits=v['forced_at']
            prefix=org.tok.decode(ids[:edits[0][0]] if edits else ids,skip_special_tokens=True)
            full=org.tok.decode(ids,skip_special_tokens=True)
            orig=o['full_generation']
            natural_final.append([bool(org.grade(extract_boxed_answer(prefix),q['ground_truth'])),bool(v['correct']),bool(o['correct'])])
            texts=(orig,prefix,full)
            boxes=[org.box(t) for t in texts]
            for rule,items in [('boxed',boxes),('explicit',[org.first(t,b) for t,b in zip(texts,boxes)])]:
                init,pre,post=items
                def score(x):
                    return bool(x and org.grade(x[1],q['ground_truth']))
                a,b=score(init),score(post)
                values[rule].append([a,b,bool(edits),bool(pre),bool(post),bool(init),prefix==orig,
                    bool(pre and post and pre==post),bool(init and post and init[1]==post[1]),
                    int(v['correct'])-int(o['correct']),int(not b and v['correct'])-int(not a and o['correct']),
                    int(b and not v['correct'])-int(a and not o['correct'])])
    out={}
    for k,rows in values.items():
        a,b,edit,pre,post,init,txt,same_pre,same_orig,final,wr,rw=np.array(rows,int).T
        one=np.ones(len(a))
        np.testing.assert_array_equal(final,b-a+wr-rw)
        out[k]=dict(original=ratio(a,one),forced=ratio(b,one),first_change=ratio(b-a,one),
            first_correctness_changed_count=int((a!=b).sum()),missing_original=int((1-init).sum()),missing_forced=int((1-post).sum()),
            inserted_answers=int(edit.sum()),inserted_after_first_count=int((pre*edit).sum()),inserted_before_first_count=int(((1-pre)*edit).sum()),
            inserted_after_first_fraction=ratio(pre*edit,edit),same_first_text_orig_full_count=int(same_orig.sum()),
            original_text_equal_before_edit=ratio(txt*edit,edit),after_first_and_same_prefix=ratio(edit*txt*same_pre,edit),
            final_change=ratio(final,one),wrong_right_change=ratio(wr,one),right_wrong_change=ratio(rw,one))
    initial,finished,arch=np.array(natural_final,int).T
    np.testing.assert_array_equal(finished-arch,(initial-arch)+(finished-initial))
    for v in out.values():
        v['same_rollout_final_decomposition']=dict(natural_eos_accuracy=ratio(initial,np.ones(len(initial))),
            natural_eos_minus_original=ratio(initial-arch,np.ones(len(initial))),
            final_minus_natural_eos=ratio(finished-initial,np.ones(len(initial))),
            natural_wrong_final_right=int(((1-initial)*finished).sum()),natural_right_final_wrong=int((initial*(1-finished)).sum()))
    return out

def attach(arm,stats,lines):
    scores=check(arm);stats['first_answer_audit']=scores
    for rule,v in scores.items():
        lines.append(f"{rule} 首答案正确率：原 {v['original']['mean']:.2%}，强制 {v['forced']['mean']:.2%}；"
            f"逐条首答案正确性改变 {v['first_correctness_changed_count']} 条；"
            f"在有追加的 {v['inserted_answers']} 条中，首次追加前已有可判分答案 {v['inserted_after_first_count']} 条。")
    v=scores['explicit']['same_rollout_final_decomposition']
    lines.append(f"同一强制轨迹在首次自然 EOS 收尾时正确率为 {v['natural_eos_accuracy']['mean']:.2%}，到最终答案净变化 {100*v['final_minus_natural_eos']['mean']:+.2f} 个百分点。")
    return scores
