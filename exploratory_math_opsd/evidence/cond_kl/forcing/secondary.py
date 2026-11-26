"""Measure teacher ranks and all-vocabulary clipping on archived math probe questions."""
import sys
from common_io import *

MODELS=('B','R@100','R-noclip@100','R-cond@100')
WORDS=['Wait','But','Alternatively','Hmm']
def measure(lane):
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    sys.path.insert(0,str(HERE.parent))
    from measure import old,checkpoint
    from launch import sha
    folder=RAW/'secondary'
    folder.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(old.MODEL)
    enc={s:tok.encode(s,add_special_tokens=False) for w in WORDS for s in (w,' '+w)}
    ids=sorted({x[0] for x in enc.values() if len(x)==1})
    if lane==0:save(HERE/'reflection_ids.json',dict(encodings=enc,included_ids=ids))
    base=old.probe.base_model()
    actor=None
    for model in MODELS[1:]:
        arm=model.split('@')[0]
        ck=old.ckpt(100) if arm=='R' else checkpoint(arm,100)
        if actor is None:actor=PeftModel.from_pretrained(old.probe.base_model(),str(ck),adapter_name=arm)
        else:actor.load_adapter(str(ck),arm,is_trainable=False)
        v=get_peft_model_state_dict(actor,adapter_name=arm,save_embedding_layers=False)
        weights=load_file(str(ck/'adapter_model.safetensors'))
        assert v.keys()==weights.keys() and all(torch.equal(v[k].cpu(),x) for k,x in weights.items())
    actor.eval()
    with torch.inference_mode():
        for idx,sp,tp,y in old.archive(lane):
            p=folder/f'{idx}.npz'
            if p.exists():continue
            si,sm=old.probe.inputs(sp,y)
            s=old.probe.forward(base,si,sm,len(sp)); t=old.probe.forward(base,*old.probe.inputs(tp,y),len(tp))
            ranks=[];teacher_ref=[];teacher_logs=[];clipped=[];mass=[]
            for i in range(0,len(y),32):
                sl=slice(i,i+32)
                lp=(s[sl].double()/1.1).log_softmax(-1)
                lq=(t[sl].double()/1.1).log_softmax(-1)
                q=lq.exp(); mask=q*(lq-lp)>.05
                tmp=t[sl].clone()
                best=[]
                for _ in range(5):
                    top=tmp.argmax(-1,keepdim=True);best.append(top);tmp.scatter_(-1,top,-torch.inf)
                top=torch.cat(best,dim=-1)
                ranks.append(top);clipped.append(mask.gather(-1,top[:,:1]).squeeze(1));mass.append((q*(~mask)).sum(-1))
                teacher_logs.append(lq.gather(-1,top[:,1:]).mean(-1))
                teacher_ref.append(q[:,ids])
            top=torch.cat(ranks)
            tlog=torch.cat(teacher_logs)
            qref=torch.cat(teacher_ref)
            models=[]
            for name in MODELS:
                if name=='B': x=s
                else:
                    arm=name.split('@')[0];actor.set_adapter(arm);assert actor.active_adapter==arm
                    x=old.probe.forward(actor,si,sm,len(sp))
                parts=[]
                for i in range(0,len(y),32):
                    sl=slice(i,i+32);lp=(x[sl].double()/1.1).log_softmax(-1)
                    agreement=(x[sl].argmax(-1)==top[sl,1]).double()
                    logratio=lp.gather(1,top[sl,1:]).mean(1)-tlog[sl]
                    reflection=lp[:,ids].exp()
                    parts.append(torch.cat([agreement[:,None],logratio[:,None],reflection.sum(1,keepdim=True),reflection],1))
                models.append(torch.cat(parts).cpu().numpy())
            old.atomic(p,clipped=torch.cat(clipped).cpu().numpy(),mass=torch.cat(mass).cpu().numpy(),teacher=qref.cpu().numpy(),models=np.stack(models,axis=1))
if __name__=='__main__':
    import time
    defer=RAW/'secondary-after-confidence'
    while defer.exists() and not (HERE.parents[1]/'latent_confidence/REPORT.md').is_file():
        time.sleep(10)
    measure(int(sys.argv[1]))
