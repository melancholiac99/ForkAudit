"""Fit a content-only rank-eight projector under a declared task/ASIN partition."""
import numpy as np
import torch
from torch.nn import functional as F
from .b_fit import SEED, weights
from .projection import VisibleProjection

class FitData:
    def __init__(self,ts,meta,head,device):
        self.meta=meta;self.device=device
        self.h=F.normalize(ts['h'].to(device=device,dtype=torch.float32))
        ids,inv=torch.unique(ts['ids'],return_inverse=True)
        self.e=F.normalize(head[ids].to(device=device,dtype=torch.float32),dim=-1)
        self.ids=inv.to(device);self.q=ts['q'].to(device);self.mask=ts['mask'].to(device)
        self.s=ts['s'].float().to(device)
        self.d=ts['d'].float().to(device) if 'd' in ts else None
        if 'teacher' in ts:
            self.target=ts['teacher'].detach().float().to(device)
        else:
            self.target=(self.s+self.d).masked_fill(~self.mask,-torch.inf).softmax(-1)

    def predict(self,net,i):
        ix,loc=torch.unique(self.ids[i],return_inverse=True)
        x=(net.state(self.h[i])[:,None,:]*net.token(self.e[ix])[loc]).sum(-1)
        return x-(x*self.q[i]).sum(-1,keepdim=True)

    def fit(self,indices,fold,loss_type='huber',initial_state=None):
        if not len(indices):raise ValueError('empty training fold')
        seed=SEED+fold;torch.manual_seed(seed)
        net=VisibleProjection(self.h.shape[1]).to(self.device)
        if initial_state is not None:
            net.load_state_dict(initial_state)
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        w=weights(self.meta,indices)
        weight=torch.zeros(len(self.h),device=self.device)
        weight[indices]=torch.from_numpy(w/w.mean()).to(self.device)
        rng=np.random.default_rng(seed)
        for epoch in range(10):
            order=rng.permutation(indices)
            for block in range(0,len(order),256):
                i=torch.as_tensor(order[block:block+256],device=self.device)
                pred=self.predict(net,i)
                if loss_type=='huber':
                    pos=(F.huber_loss(pred,self.d[i],reduction='none',delta=1.)*self.mask[i]).sum(-1)/self.mask[i].sum(-1)
                elif loss_type=='kl':
                    logp=(self.s[i]+pred).masked_fill(~self.mask[i],-torch.inf).log_softmax(-1).masked_fill(~self.mask[i],0)
                    q=self.target[i];pos=(torch.special.xlogy(q,q)-q*logp).sum(-1)
                else:raise ValueError(loss_type)
                loss=(weight[i]*pos).mean()
                opt.zero_grad();loss.backward();opt.step()
        return net.eval()
