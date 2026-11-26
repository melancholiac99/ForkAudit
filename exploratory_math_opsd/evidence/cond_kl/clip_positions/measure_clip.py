import json,sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from positions import boundaries,pieces
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from measure import old
OUT=old.ROOT/'run/cond-kl/clip-prefix-v2'
@torch.inference_mode()
def run(lane):
    OUT.mkdir(exist_ok=True)
    prior=json.loads((HERE.parents[1]/'stop_decomposition/results.json').read_text())
    vocab=pieces(AutoTokenizer.from_pretrained(old.MODEL))
    base=old.probe.base_model()
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    ck=old.ckpt(50)
    actor=PeftModel.from_pretrained(old.probe.base_model(),str(ck)).eval()
    loaded=get_peft_model_state_dict(actor,save_embedding_layers=False);source=load_file(str(ck/'adapter_model.safetensors'))
    assert loaded.keys()==source.keys() and all(torch.equal(loaded[k].cpu(),x) for k,x in source.items())
    rows=old.archive(lane)
    for i,(idx,sp,tp,y) in enumerate(rows):
        dst=OUT/f'{idx}.npz'
        if dst.exists():continue
        bound=boundaries([vocab[t] for t in y],b''.join(vocab[t] for t in sp))
        with np.load(old.probe.OUT/f'{idx}.npz') as a:
            masks=np.stack((np.ones(len(y),bool),a['p']>=.05,(a['p']<.01)&(a['u']>=prior['content_u80'])))
        t=old.probe.forward(base,*old.probe.inputs(tp,y),len(tp))
        result=[]
        for student in (base,actor):
            s=old.probe.forward(student,*old.probe.inputs(sp,y),len(sp))
            chunks=[]
            for start in range(0,len(y),32):
                sl=slice(start,start+32);lp=(s[sl].double()/1.1).log_softmax(-1);lq=(t[sl].double()/1.1).log_softmax(-1)
                q=lq.exp();k=q*(lq-lp);big=k>.05;qa=q*big
                norm=torch.linalg.vector_norm(qa-qa.sum(-1,keepdim=True)*lp.exp(),dim=-1)
                chunks.append(torch.stack((big.sum(-1),torch.relu(k-.05).sum(-1),(k*big).sum(-1),norm),1).cpu().numpy())
            v=np.column_stack((np.ones(len(y)),np.concatenate(chunks)))
            assert np.allclose(v[:,3]-v[:,1]*.05,v[:,2])
            result.append(np.stack(((masks&bound)@v,(masks&~bound)@v),axis=1))
            del s
        size=sum(len(row[3]) for row in rows[i//4*4:i//4*4+4])
        old.atomic(dst,sums=np.array(result),length=len(y),micro_length=size)
if __name__=='__main__':run(int(sys.argv[1]))
