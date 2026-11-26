"""Reuse the registered group masks, teacher probabilities and tokenizer IDs."""
import sys
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel,get_peft_model_state_dict
from safetensors.torch import load_file
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from measure import old,checkpoint
from launch import read
@torch.inference_mode()
def run(lane):
    root=old.ROOT/'run/cond-kl/forcing'
    out=old.ROOT/'run/cond-kl/R-condA-secondary';out.mkdir(exist_ok=True)
    ck=checkpoint('R-condA',100);model=PeftModel.from_pretrained(old.probe.base_model(),str(ck)).eval()
    weights=load_file(str(ck/'adapter_model.safetensors'))
    actual=get_peft_model_state_dict(model,save_embedding_layers=False)
    assert weights.keys()==actual.keys() and all(torch.equal(actual[k].cpu(),v) for k,v in weights.items())
    metadata=read(HERE.parent/'forcing/reflection_ids.json')
    ids=metadata['included_ids']
    pairs=[[ids.index(metadata['encodings'][form][0]) for form in (word,' '+word)] for word in ('Wait','But','Alternatively','Hmm')]
    base=old.probe.base_model()
    for idx,sp,tp,y in old.archive(lane):
        f=out/f'{idx}.npz'
        if f.exists():continue
        c=old.probe.forward(model,*old.probe.inputs(sp,y),len(sp));t=old.probe.forward(base,*old.probe.inputs(tp,y),len(tp))
        with np.load(root/'secondary'/f'{idx}.npz') as z:
            clipped=z['clipped']; previous=z['models'];teacher=z['teacher']
            orig=np.concatenate([previous[:,:,:3],np.stack([previous[:,:,3:][:,:,pair].sum(-1) for pair in pairs],-1)],-1)
        vals=[]
        for i in range(0,len(y),32):
            lc=(c[i:i+32].double()/1.1).log_softmax(-1);tlog=t[i:i+32].clone();lt=(tlog.double()/1.1).log_softmax(-1)
            order=[]
            for _ in range(5):
                top=tlog.argmax(-1,keepdim=True);order.append(top);tlog.scatter_(1,top,-torch.inf)
            top=torch.cat(order[1:],dim=1)
            reflection=lc[:,ids].exp()
            vals.append(torch.stack([(lc.argmax(-1)==order[1][:,0]).double(),(lc-lt).gather(1,top).mean(1),reflection.sum(-1),*[reflection[:,pair].sum(-1) for pair in pairs]],1).cpu().numpy())
        x=np.concatenate([orig,np.concatenate(vals)[:,None,:]],1)
        groups=[~clipped,clipped]
        old.atomic(f,counts=np.array([g.sum() for g in groups]),sums=np.array([x[g].sum(0) for g in groups]),teacher=np.array([teacher[g].sum() for g in groups]))
if __name__=='__main__':run(int(sys.argv[1]))
