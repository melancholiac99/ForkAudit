"""Sampled contrastive score-function loss with a detached SDPO upper importance cap."""
import json
import math
import os
from pathlib import Path
import torch
from torch.distributed import all_reduce,get_world_size,is_initialized
from torch.utils.checkpoint import checkpoint

def rollout_metadata(outputs, ids, padded, trainer):
    rows=[o for group in outputs for o in group.outputs]
    assert trainer.vllm_tensor_parallel_size==1 and len(rows)==len(ids)==len(padded)
    logp=torch.zeros_like(padded,dtype=torch.float32)
    for i,(row,seq) in enumerate(zip(rows,ids)):
        assert len(row.logprobs)==len(seq)<=padded.shape[1] and len(seq)>0
        assert list(seq)==padded[i,:len(seq)].tolist()
        values=[v[token].logprob for token,v in zip(seq,row.logprobs)]
        assert all(math.isfinite(x) and x<=0.00001 for x in values)
        logp[i,:len(seq)]=torch.tensor(values,device=padded.device)
        trainer.repair_counts[0]+=int(row.finish_reason=='stop' and seq[-1] in trainer._repair_eos)
        trainer.repair_counts[1]+=1
    return logp

def pg(student,plus,minus,old):
    advantage=(.5*(plus-minus)).detach()
    delta=(student-old).detach()
    weight=delta.clamp(-20,20).exp().clamp(max=2.)
    return -weight*advantage*student,delta>math.log(2.)

def sampled_logp(logits,ids):
    x=logits.float()/1.1
    return x.gather(-1,ids[:,None])[:,0]-x.logsumexp(-1)

def loss(student,positive,negative,labels,has_negative,old,step,output):
    valid=(labels.ne(-100)&has_negative[:,None]).nonzero()
    value=student[:0].sum(dtype=torch.float32)
    hits=student.new_zeros((),dtype=torch.long)
    def block(s,p,n,ids,logged):
        term,clipped=pg(sampled_logp(s,ids),sampled_logp(p,ids),sampled_logp(n,ids),logged)
        return term.sum(),clipped.sum()
    assert old.shape==labels.shape
    for offsets in valid.split(48):
        if not len(offsets):continue
        i,j=offsets.unbind(1)
        subtotal,clipped=checkpoint(block,student[i,j],positive[i,j],negative[i,j],labels[i,j],old[i,j],use_reentrant=False)
        value=value+subtotal; hits=hits+clipped
    count=student.new_tensor(len(valid),dtype=torch.long)
    if is_initialized():all_reduce(count)
    with (Path(output)/('ratio-rank-'+os.environ['RANK']+'.jsonl')).open('a') as f:
        print(json.dumps(dict(step=step,supervised=len(valid),clipped=int(hits),eligible=int(labels.ne(-100).sum()))),file=f)
    return value*(get_world_size() if is_initialized() else 1)/count.clamp_min(1)
