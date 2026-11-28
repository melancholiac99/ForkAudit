import numpy as np
from score import prepare,read,save,HERE,ROOT,SelfDistillationDataCollator
from transformers import AutoModelForCausalLM,AutoTokenizer
from format_target import targets,format_mask
import torch,re

@torch.inference_mode()
def main():
    jobs,old,archive=prepare()
    edits={r['idx']:r for r in read(HERE/'diagnostic_edits.json') if r['qualified']}
    assert len(edits)==16
    path=ROOT/'models/Qwen3-1.7B'
    tok=AutoTokenizer.from_pretrained(path)
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
    col=SelfDistillationDataCollator(tok,max_length=20000,reason_first=False,student_thinking=False,teacher_thinking=True)
    mask=format_mask(model.config.vocab_size,torch.device('cuda'))
    rows=[]
    for i,j in enumerate(jobs[:800]):
        if j['idx'] not in edits:continue
        text=archive[j['problem']]
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        y=enc.input_ids
        batch=jobs[i//4*4:i//4*4+4]
        def hidden(context,key):
            prompts=col([dict(b,solution=c) for b,c in zip(batch,context)])[key]
            ids=torch.tensor([prompts[i%4].tolist()+y],device='cuda')
            return model.model(ids,attention_mask=ids.ne(tok.pad_token_id).long(),use_cache=False).last_hidden_state[0,prompts.shape[1]-1:-1]
        ref=[b['solution'] for b in batch]
        h=[hidden(ref,'student_prompts'),hidden(ref,'teacher_prompts'),hidden([edits[b['idx']]['text'] if b['idx'] in edits else old[b['problem']] or b['solution'] for b in batch],'teacher_prompts')]
        spans=[m.span() for m in re.finditer(r'\bfinal\b',text,re.I)]
        for k in range(0,len(y),64):
            s,p,n=[model.lm_head(v[k:k+64]).float() for v in h]
            lp,q,raw=targets(s,p,n,mask,1.1)
            # Match the old report's subtraction order for raw signal ranking.
            p=p/1.1;n=n/1.1
            sel=torch.tensor(y[k:k+len(s)],device='cuda')[:,None]
            before=.5*((p.gather(1,sel)[:,0]-p.logsumexp(-1))-(n.gather(1,sel)[:,0]-n.logsumexp(-1)))
            oldq=(lp+raw.clamp(-3,3)).log_softmax(-1)
            data=torch.stack([before,before.masked_fill(mask[sel[:,0]],0),(oldq-lp).gather(1,sel)[:,0],(q-lp).gather(1,sel)[:,0]]).T.cpu().tolist()
            for t,(v,w,o,z) in enumerate(data,k):
                a,b=enc.offset_mapping[t]
                rows.append(dict(idx=j['idx'],t=t,token=text[a:b],prefix=text[max(0,a-70):a],following=text[b:b+45],raw=v,masked=w if t else 0.,old_target=o,target=z if t else 0.,final=any(a<hi and b>lo for lo,hi in spans),eos=y[t] in model.generation_config.eos_token_id))
    result={}
    for key in ['raw','masked','old_target','target']:
        top=sorted(rows,key=lambda r:(-abs(r[key]),r['idx'],r['t']))[:109]
        result[key]=dict(top=top)
        for subset in ('eos','final'):
            a=[[r[key] for r in rows if r['idx']==i and r[subset]] for i in sorted(edits)]
            vals=np.array([[sum(x),len(x)] for x in a])
            draws=vals[np.random.default_rng(20261124).integers(16,size=(10000,16))].sum(1)
            result[key][subset]=dict(mean=float(vals[:,0].sum()/vals[:,1].sum()),ci=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist())
    original=read(HERE/'top.json')['top']
    assert [(r['idx'],r['t']) for r in result['raw']['top']]==[(r['idx'],r['t']) for r in original]
    save(HERE/'format_probe.json',result)
if __name__=='__main__':main()
