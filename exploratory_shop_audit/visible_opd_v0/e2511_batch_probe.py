"""Measure paired BF16 scoring on saved training prefixes during a spare GPU interval."""
import os
import time

import torch
from .c_targets import actor, response
from .e25_probe import RUN, configure
from .s21_data import SmokeBatch
from exploratory_shop_audit.grpo_forced_audit.common import save


@torch.inference_mode()
def main():
    configure()
    model=actor().eval()
    rows=SmokeBatch().rows
    length=lambda row:len(row['prompt'])+len(row['answer'])
    ordered=sorted(range(len(rows)),key=lambda i:(length(rows[i]),i))
    # Within each length stratum take two adjacent, distinct native prefixes.
    pairs=[]
    for n in range(12):
        start=n*(len(ordered)-3)//11
        i=ordered[start]
        for j in ordered[start+1:]:
            if rows[j]['prompt']!=rows[i]['prompt']:
                pairs.append((i,j))
                break
    def single(pair):
        return [response(model,rows[i]['prompt'],rows[i]['answer'])[1] for i in pair]
    def batched(pair):
        tokens=[rows[i]['prompt']+rows[i]['answer'] for i in pair]
        width=max(map(len,tokens))
        ids=torch.zeros((2,width),dtype=torch.long,device='cuda')
        mask=torch.zeros_like(ids)
        for n,t in enumerate(tokens):
            ids[n,:len(t)]=torch.tensor(t,device='cuda')
            mask[n,:len(t)]=1
        base=model.get_base_model()
        h=base.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        return [base.lm_head(h[n,len(rows[i]['prompt'])-1:len(tokens[n])-1]).float() for n,i in enumerate(pair)]
    single(pairs[0]);batched(pairs[0])
    stats=[]
    for n,pair in enumerate(pairs):
        out={};times={};peaks={}
        order=[('single',single),('pair',batched)]
        for name,fn in (order if n%2==0 else reversed(order)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start=time.monotonic()
            values=fn(pair)
            torch.cuda.synchronize()
            times[name]=time.monotonic()-start
            peaks[name]=torch.cuda.max_memory_allocated()
            out[name]=[v.cpu() for v in values]
            del values
        mismatches=valid=top8_changed=positions=0
        for i,x,y in zip(pair,out['single'],out['pair'],strict=True):
            r=rows[i];at=r['ids'];mask=r['mask']
            mismatches+=int(((x.gather(1,at)!=y.gather(1,at))&mask).sum())
            valid+=int(mask.sum())
            top8_changed+=int((x.topk(8,dim=-1).indices.sort(1).values!=y.topk(8,dim=-1).indices.sort(1).values).any(1).sum())
            positions+=len(r['answer'])
        stats.append(dict(records=pair,lengths=[length(rows[i]) for i in pair],timing_s=times,
                          peak_allocated_bytes=peaks,candidate_mismatches=mismatches,candidates=valid,
                          positions=positions,top8_changed_positions=top8_changed))
    a=sum(x['timing_s']['single'] for x in stats);b=sum(x['timing_s']['pair'] for x in stats)
    save(RUN/'batch_probe_2511.json',dict(device=int(os.environ['LOCAL_RANK']),
        scope='BF16 student forward only; 12 length-matched pairs on original training prefixes, native unpadded single versus right-padded batch of two; alternating timings after warmup',
        pairs=stats,speedup=a/b,candidate_mismatches=sum(x['candidate_mismatches'] for x in stats),
        top8_changed_positions=sum(x['top8_changed_positions'] for x in stats),training_settings_changed=False))


if __name__=='__main__':
    main()
