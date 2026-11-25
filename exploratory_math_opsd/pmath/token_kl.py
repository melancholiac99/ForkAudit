"""Cross-fitted positive forward KL by vocabulary ID, on the archived completions."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from common import MODEL, WORK, save
from fit import split

TEMPERATURES=(1.1,1.)
TARGETS=('R','P','residual')
OUT=WORK/'positive-token-kl'

@torch.inference_mode()
def process(lane):
    OUT.mkdir(exist_ok=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2').cuda().eval()
    head=model.get_output_embeddings()
    folds=split()
    scale=torch.load(WORK/'projector.pt',weights_only=True)['scale']
    params=[]
    for fold in range(4):
        state=torch.load(WORK/f'fit-{fold}.pt',weights_only=True,map_location='cuda')
        params.append((state['A'],F.linear(head.weight.float(),state['B'])))
    rows=torch.load(WORK/f'rollouts-{lane}.pt',weights_only=True)
    for block in range(0,len(rows),25):
        dest=OUT/f'lane{lane}-{block}.pt'
        if dest.exists():continue
        mass=torch.zeros(2,3,head.weight.shape[0],device='cuda',dtype=torch.float64)
        signed=torch.zeros_like(mass)
        count=0
        fold_counts=[0]*4
        max_label_error=0.
        for idx,_,teacher,answer in rows[block:block+25]:
            archived=torch.load(WORK/f'features-{idx}.pt',weights_only=True,map_location='cuda')
            hs=archived['h']
            ht=model.model(input_ids=torch.tensor([teacher+answer],device='cuda'),
                    use_cache=False).last_hidden_state[0,len(teacher)-1:-1]
            assert hs.shape==ht.shape and len(hs)==len(answer)
            count+=len(hs)
            f=folds[idx];fold_counts[f]+=1
            A,rows_emb=params[f]
            for j in range(0,len(answer),64):
                h=hs[j:j+64]
                s=head(h).float()
                t=head(ht[j:j+64]).float()
                if j==0:
                    c=archived['ids'][0].long()
                    valid=archived['valid'][0]
                    d=(t[0]-s[0])[c]
                    d-= (s[0,c].masked_fill(~valid,-torch.inf).softmax(0)*d).sum()
                    max_label_error=max(max_label_error,(d-archived['d'][0]).abs().max().item())
                top=s.topk(64,dim=-1).indices
                correction=scale*torch.einsum('nr,nkr->nk',F.linear(h.float(),A),rows_emb[top])
                p=s.scatter_add(1,top,correction)
                residual=t.scatter_add(1,top,-correction)
                for k,temp in enumerate(TEMPERATURES):
                    lp=F.log_softmax(s/temp,dim=-1)
                    for a,target in enumerate((t,p,residual)):
                        lq=F.log_softmax(target/temp,dim=-1)
                        contrib=lq.exp()*(lq-lp)
                        mass[k,a]+=contrib.clamp_min(0).sum(0)
                        signed[k,a]+=contrib.sum(0)
        assert torch.isfinite(mass).all() and torch.isfinite(signed).all()
        save(dest,dict(mass=mass.cpu(),signed=signed.cpu(),positions=count,
                       questions=len(rows[block:block+25]),folds=fold_counts,max_label_error=max_label_error))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('lane',type=int)
    process(parser.parse_args().lane)
