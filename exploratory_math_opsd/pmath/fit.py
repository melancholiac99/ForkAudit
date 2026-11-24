import sys, random
import torch
from torch.nn import functional as F
from common import WORK,save,json_save

def split():
    rows=sorted(int(p.stem.split('-')[1]) for p in WORK.glob('features-*.pt'))
    assert len(rows)==1500
    random.Random(42).shuffle(rows)
    return {idx:i%4 for i,idx in enumerate(rows)}

def load(idx):
    return torch.load(WORK/f'features-{idx}.pt',weights_only=True,map_location='cuda')

def fit(fold):
    dest=WORK/f'fit-{fold}.pt'
    if dest.exists():return
    folds=split()
    ids=[i for i,f in folds.items() if f!=fold]
    torch.manual_seed(42)
    head=torch.load(WORK/'head.pt',weights_only=True,map_location='cuda').float()
    dim=head.shape[1]
    A=torch.nn.Parameter(torch.randn(8,dim,device='cuda')/dim**.5)
    B=torch.nn.Parameter(torch.randn_like(A)/dim**.5)
    opt=torch.optim.Adam([A,B],lr=.001)
    def predict(x):
        tokens,back=torch.unique(x['ids'].long(),return_inverse=True)
        left=F.linear(x['h'].float(),A)
        right=F.linear(head[tokens],B)
        return torch.einsum('pr,pcr->pc',left,right[back])
    def weights(x):
        return x['valid']/x['valid'].sum(dim=1,keepdim=True)
    for epoch in range(3):
        random.Random(42+epoch).shuffle(ids)
        for i in ids:
            x=load(i)
            loss=(weights(x)*F.huber_loss(predict(x),x['d'],reduction='none')).sum()
            # Fixed position normalization makes each position carry the same objective weight.
            opt.zero_grad()
            (loss/1024).backward()
            opt.step()
    save(dest,dict(A=A.detach().cpu(),B=B.detach().cpu()))

@torch.no_grad()
def heldout(fold,scale=None):
    state=torch.load(WORK/f'fit-{fold}.pt',weights_only=True,map_location='cuda')
    head=torch.load(WORK/'head.pt',weights_only=True,map_location='cuda').float()
    vocab=F.linear(head,state['B'])
    xx=xy=yy=hub=post=0.
    n=0
    for i,f in split().items():
        if f!=fold:continue
        x=load(i)
        p=torch.einsum('pr,pcr->pc',F.linear(x['h'].float(),state['A']),vocab[x['ids'].long()])
        w=x['valid']/x['valid'].sum(1,keepdim=True)
        y=x['d']
        xx+=(w*p.square()).sum().item()
        xy+=(w*p*y).sum().item()
        yy+=(w*y.square()).sum().item()
        hub+=(w*F.huber_loss(p,y,reduction='none')).sum().item()
        if scale is not None:post+=(w*F.huber_loss(p*scale,y,reduction='none')).sum().item()
        n+=len(p)
    out=dict(n=n,xx=xx,xy=xy,yy=yy,huber=hub)
    if scale is not None:out["shrunk_huber"]=post
    return out

def finish():
    assert all((WORK/f'fit-{f}.pt').exists() for f in range(4))
    parts=[heldout(f) for f in range(4)]
    total={k:sum(r[k] for r in parts) for k in parts[0]}
    slope=total['xy']/total['xx']
    scale=max(0.,min(1.,slope))
    fit(-1)
    state=torch.load(WORK/'fit--1.pt',weights_only=True)
    save(WORK/'projector.pt',{**state,'scale':scale})
    json_save(WORK/'oof.json',dict(folds=parts,positions=total['n'],questions=1500,
        slope=slope,scale=scale,huber=total['huber']/total['n'],
        mse=(total['yy']-2*total['xy']+total['xx'])/total['n'],
        shrunk_mse=(total['yy']-2*scale*total['xy']+scale**2*total['xx'])/total['n'],
        shrunk_huber=sum(heldout(f,scale)['shrunk_huber'] for f in range(4))/total['n']))
if __name__=='__main__':
    if sys.argv[1]=='finish':finish()
    else:fit(int(sys.argv[1]))
