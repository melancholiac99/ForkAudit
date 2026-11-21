"""Measure the realized BF16 residual of every LoRA linear at decision-prefix positions."""
import sys
from collections import Counter
import torch
from .c_targets import actor,response
from .d_data import snapshot
from .e25_probe import RUN,PUBLIC,configure,pooled
from .s21_data import SmokeBatch
from exploratory_shop_audit.grpo_forced_audit.common import read,save

@torch.inference_mode()
def run(rank):
    assert (RUN/'movement.json').exists()
    model=actor().eval();batch=SmokeBatch();chunks=pooled(batch,rank)
    layers={n:m for n,m in model.named_modules() if hasattr(m,'lora_B') and 'default' in m.lora_B}
    kinds=Counter(n.rsplit('.',1)[-1] for n in layers)
    assert sum(kinds.values()) and all(m.lora_B['default'].weight.dtype==torch.float32 for m in layers.values())
    context={};values={};handles=[]
    def hooks(n,m):
        kind=n.rsplit('.',1)[-1];capture={}
        def store(which):
            def inside(_,args,out):
                capture[which]=out[0,context['start']:context['end']].detach()
            return inside
        def finish(_,args,out):
            base=capture.pop('base');delta=capture.pop('delta')*m.scaling['default']
            assert base.dtype==out.dtype==torch.bfloat16 and delta.dtype==torch.float32
            effective=out[0,context['start']:context['end']]
            rounded=(base.float()+delta).bfloat16()
            mismatch=(rounded!=effective).sum()
            change=effective.float()-base.float()
            nz=delta!=0
            vec=torch.stack((delta.double().square().sum(),change.double().square().sum(),
                (delta.double()*change).sum(),nz.sum(),(nz & (change==0)).sum(),mismatch))
            values[kind]+=vec*context['weight']
            counts[kind]+=torch.stack((nz.sum(),(nz & (change==0)).sum(),mismatch))*context['copies']
        handles.extend((m.base_layer.register_forward_hook(store('base')),m.lora_B['default'].register_forward_hook(store('delta')),m.register_forward_hook(finish)))
    for n,m in layers.items():hooks(n,m)
    for arm in ('P','R'):
        out=RUN/'deadzone'/f'{arm}_{rank}.json'
        if out.exists():continue
        snapshot(model,10,arm)
        values={k:torch.zeros(6,device='cuda',dtype=torch.float64) for k in kinds}
        counts={k:torch.zeros(3,device='cuda',dtype=torch.int64) for k in kinds}
        for ix in chunks:
            r=batch.rows[ix[0]];start=len(r['prompt'])-1
            context.update(start=start,end=start+len(r['answer']),weight=sum(batch.weights[j] for j in ix),copies=len(ix))
            response(model,r['prompt'],r['answer'])
        assert all(int(v[-1].item())==0 for v in counts.values())
        save(out,dict(layers=kinds,values={k:v.tolist() for k,v in values.items()},counts={k:v.tolist() for k,v in counts.items()}))
    for h in handles:h.remove()

def report():
    result={}
    for arm in ('P','R'):
        shards=[read(RUN/'deadzone'/f'{arm}_{r}.json') for r in range(4)];results={}
        for kind,layers in shards[0]['layers'].items():
            d,e,dot,nz,lost,err=map(sum,zip(*(s['values'][kind] for s in shards)))
            count,zero,mismatch=map(sum,zip(*(s['counts'][kind] for s in shards)))
            assert d>0 and nz>0 and err==mismatch==0
            results[kind]=dict(layers=layers,norm_ratio=(e/d)**.5,cosine=dot/(d*e)**.5 if e else None,
                zero_given_nonzero_weighted=lost/nz,nonzero_elements=count,rounded_to_zero_elements=zero,
                zero_given_nonzero_unweighted=zero/count,actual_forward_mismatches=mismatch)
        result[arm]=results
    save(PUBLIC/'lora_deadzone.json',result);save(RUN/'lora_deadzone.json',result)

if __name__=='__main__':
    if sys.argv[1:]==['report']:report()
    else:run(configure())
