"""Teacher-force the matched negatives on exactly the same archived answer prefix."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from check import HERE,ROOT,DATA,read,save
WORK=DATA.parent/'contrast'
sys.path.insert(0,str(WORK/'source'))
from data_collator import SelfDistillationDataCollator

def prepare():
    jobs=read(WORK/'manifest.json')[:3200]
    neg=read(WORK/'negative.json')
    archive={}
    for f in (ROOT/'run/R-contrast/generations').glob('*.json'):
        for row in read(f)['generations']:
            if row['step']<25 and row['completion'].endswith('<|im_end|>'):
                key=row['prompt'].split('Problem: ',1)[1].rsplit('\n\nPlease reason',1)[0]
                assert key not in archive
                archive[key]=row['completion']
    pool=sorted([dict(j,negative=neg[j['problem']]) for j in jobs if neg[j['problem']]],key=lambda j:j['idx'])
    save(DATA/'pilot.json',pool[:50])
    save(DATA/'diag_pool.json',[j for j in pool if j['problem'] in archive])
    return jobs,neg,archive

@torch.inference_mode()
def run():
    jobs,old,archive=prepare()
    edits={r['idx']:r for r in read(HERE/'diagnostic_edits.json') if r['qualified']}
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    collate=SelfDistillationDataCollator(tok,max_length=20000,reason_first=False,student_thinking=False,teacher_thinking=True)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/Qwen3-1.7B',dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
    allrows=[];metrics={}
    for i,j in enumerate(jobs[:800]):
        question=j['problem']
        if j['idx'] not in edits:continue
        ans=archive[question]
        enc=tok(ans,add_special_tokens=False,return_offsets_mapping=True)
        y=enc.input_ids
        assert y[-1] in model.generation_config.eos_token_id
        batch=jobs[i//4*4:i//4*4+4]
        def score(contexts):
            batch_inputs=collate([dict(job,solution=c) for job,c in zip(batch,contexts)])
            prompt=batch_inputs['teacher_prompts'][i%4].tolist()
            ids=torch.tensor([prompt+y],device='cuda')
            mask=ids.ne(tok.pad_token_id).long()
            hidden=model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state[0,len(prompt)-1:-1]
            result=[]
            for k in range(0,len(y),64):
                logits=model.lm_head(hidden[k:k+64]).float()/1.1
                ids_y=ids[0,len(prompt)+k:len(prompt)+k+len(logits)]
                result.append((logits.gather(1,ids_y[:,None])[:,0]-logits.logsumexp(-1)).cpu())
            return torch.cat(result).numpy()
        pos=score([b['solution'] for b in batch])
        orig=[old[b['problem']] or b['solution'] for b in batch]
        neg1=score(orig)
        neg2=score([edits[b['idx']]['text'] if b['idx'] in edits else default for b,default in zip(batch,orig)])
        offsets=enc.offset_mapping
        masks={'EOS':np.isin(y,model.generation_config.eos_token_id),'all':np.ones(len(y),bool)}
        for key,pat in [('Final',r'\bfinal\b'),('Answer',r'\banswer\b'),('###',r'#{3}'),('boxed',r'\\boxed\b')]:
            spans=[m.span() for m in re.finditer(pat,ans,re.I)]
            masks[key]=np.array([any(a<hi and b>lo for lo,hi in spans) for a,b in offsets])
        for arm,negative in [('original',neg1),('rewrite',neg2)]:
            diff=.5*(pos-negative)
            for key,m in masks.items():
                if m.any():metrics.setdefault(arm+'/'+key,[]).append([j['idx'],int(m.sum()),float(diff[m].sum(dtype=float))])
            if arm=='rewrite':
                for t,((a,b),d) in enumerate(zip(offsets,diff)):
                    allrows.append(dict(idx=j['idx'],t=t,signal=float(d),token=ans[a:b],prefix=ans[max(0,a-130):a],following=ans[b:b+120]))
    stats={}
    for k,rows in metrics.items():
        arr=np.array(rows);a=arr[:,1:]
        samp=np.random.default_rng(20261124).integers(len(a),size=(10000,len(a)))
        boot=a[samp].sum(1);stats[k]=dict(questions=len(a),tokens=int(a[:,0].sum()),mean=float(a[:,1].sum()/a[:,0].sum()),ci=np.quantile(boot[:,1]/boot[:,0],[.025,.975]).tolist())
    top=sorted(allrows,key=lambda v:(-abs(v['signal']),v['idx'],v['t']))[:int(np.ceil(.01*len(allrows)))]
    selection=np.random.default_rng(20261124).choice(len(top),min(10,len(top)),replace=False).tolist() if top else []
    interval=stats.get('rewrite/EOS',{}).get('ci')
    save(HERE/'signal.json',dict(stats=stats,gate_pass=bool(interval and interval[0]<=0<=interval[1]),unit='actual answer token',by_question=metrics))
    save(HERE/'top.json',dict(total=len(allrows),top=top,example_indices=selection))
if __name__=='__main__':run()
