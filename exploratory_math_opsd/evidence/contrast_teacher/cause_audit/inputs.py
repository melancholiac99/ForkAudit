"""Trace the first 100 updates using real CPU collation and logged microbatch masks."""
import hashlib,json,os,sys
from collections import Counter
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
W=Path('/mnt/data2/ForkAudit/contrast')
ROOT=Path('/mnt/data/ForkAudit/math-opsd')
D=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'source'))
from training_parts import ContrastCollator
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def desc(v):
    return dict(mean=float(np.mean(v)),p50=float(np.median(v)),p90=float(np.percentile(v,90)),max=float(max(v)))
def main():
    os.environ['CONTRAST_NEGATIVES']=str(W/'negative.json')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B',local_files_only=True)
    collate=ContrastCollator(tok,max_length=20000,reason_first=False,student_thinking=False,teacher_thinking=True)
    jobs=read(W/'manifest.json');neg=read(W/'negative.json')
    logs=[[json.loads(l) for l in (W/f'checkpoints/supervision-rank-{rank}.jsonl').read_text().splitlines()] for rank in range(4)]
    sums=Counter();styles=Counter();lens={k:[] for k in ('positive','negative','relative_offset','first_token_loss_weight','rank_valid')}
    excluded=Counter();special=set(tok.all_special_tokens)
    for step in range(100):
        for micro in range(2):
            for rank in range(4):
                start=step*32+micro*16+rank*4;batch=jobs[start:start+4]
                x=collate(batch);val=x['has_negative'].tolist();log=logs[rank][step*2+micro]
                assert log['step']==step+1 and bool(log['supervised'])==any(val)
                sums['microbatches']+=1;sums['eligible']+=log['eligible'];sums['supervised']+=log['supervised']
                sums['all_zero_microbatches']+=not any(val)
                lens['rank_valid'].append(log['supervised'])
                lengths={}
                for key in ('student_prompts','teacher_prompts','negative_prompts'):
                    ids=x[key];lengths[key]=[(ids.shape[1],int((r!=tok.pad_token_id).sum())) for r in ids]
                clean=[s.replace(tok.pad_token,'') for s in tok.batch_decode(x['student_prompts'],skip_special_tokens=False)]
                rec=tok(clean,return_tensors='pt',padding='longest',truncation=True,max_length=18976,add_special_tokens=False).input_ids
                sums['roundtrip_batch_width_mismatch']+=rec.shape[1]!=x['student_prompt_length']
                for a,b in zip(rec,x['student_prompts']):
                    sums['roundtrip_unpadded_id_mismatch']+=a[a!=tok.pad_token_id].tolist()!=b[b!=tok.pad_token_id].tolist()
                exposed=0
                for i,j in enumerate(batch):
                    pos=j['solution'];minus=neg[j['problem']]
                    if not val[i]:
                        excluded['questions']+=1;continue
                    sums['supervised_questions']+=1
                    gaps={k:w-l for k,(w,l) in ((k,v[i]) for k,v in lengths.items())}
                    # First predictor index is batch_width-1 in all three branches.
                    for k in gaps:sums[k+'_first_PAD']+=bool(gaps[k])
                    bad=bool(gaps['teacher_prompts'] or gaps['negative_prompts'])
                    sums['either_teacher_first_PAD']+=bad
                    sums['any_first_PAD']+=any(gaps.values())
                    exposed+=bad
                    lens['relative_offset'].append(abs(gaps['teacher_prompts']-gaps['negative_prompts']))
                    sums['different_teacher_gap']+=gaps['teacher_prompts']!=gaps['negative_prompts']
                    pos=collate.positive_proofs.get(j['problem'],pos)
                    for label,txt in [('positive',pos),('negative',minus)]:
                        lens[label].append(len(tok.encode(txt,add_special_tokens=False)))
                        for word in ('Wait','</think>','<think>','\\boxed','<|im_end|>','<|im_start|>','###','**'):
                            styles[label+' '+word]+=word.lower() in txt.lower()
                lens['first_token_loss_weight'].append(exposed/max(1,log['supervised']))
    used=W/'source'
    assert sha(used/'contrast_objective.py')==sha(D.parent/'objective.py')
    assert sha(used/'training_parts.py')==sha(D.parent/'training_parts.py')
    assert sha(W/'negative.json')==read(D.parent/'negative_contexts.json')['sha256']
    ret=dict(counts=sums,style_questions=styles,distributions={k:desc(v) for k,v in lens.items()},excluded=excluded,
             sources={str(p.name):sha(p) for p in [used/'opsd_trainer.py',used/'training_parts.py',used/'contrast_objective.py',used/'data_collator.py',W/'manifest.json',W/'negative.json']})
    (D/'inputs.json').write_text(json.dumps(ret,indent=2)+'\n')
if __name__=='__main__':main()
