"""Three frozen BF16 forwards with a shared cached prompt per candidate sequence."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from .b_collect import RUN,PARENT
from .score import support,measure,centered
from exploratory_shop_audit.grpo_forced_audit.common import MODEL,read,save,append

def padded(sequences,device):
    return torch.nn.utils.rnn.pad_sequence([torch.tensor(s,device=device) for s in sequences],batch_first=True,padding_value=0)

def forward(model,prompt,seqs):
    dev=model.device
    base=model.model(input_ids=torch.tensor([prompt],device=dev),use_cache=True)
    cache=base.past_key_values
    cache.batch_repeat_interleave(len(seqs))
    tokens=padded(seqs,dev)
    out=model.model(input_ids=tokens,past_key_values=cache,use_cache=True).last_hidden_state
    # Right padding cannot affect earlier positions in a causal decoder.
    start=base.last_hidden_state[:,-1:].expand(len(seqs),-1,-1)
    pred=torch.cat([start,out[:,:-1]],1)
    return [pred[i,:len(s)] for i,s in enumerate(seqs)],torch.stack([out[i,len(s)-1] for i,s in enumerate(seqs)])

def distribution(model,h):
    return model.lm_head(h).float().log_softmax(-1)

@torch.inference_mode()
def score_row(model,tok,row,repeat):
    actual=row['response'];dev=model.device
    actions=row['legal_clicks'][:8]
    seqs=[actual]+[tok.encode('click['+a+']',add_special_tokens=False) for a in actions]
    compact={};ret={}
    for mode in ('none','full','placeholder')+ (('selfcheck',) if repeat else ()):
        ids=row['views']['placeholder' if mode=='selfcheck' else mode]
        h,ends=forward(model,ids,seqs)
        logp=distribution(model,h[0])
        if not logp.isfinite().all():raise ValueError('nonfinite response scores')
        if mode=='none':
            candidates,mask=support(logp,torch.tensor(actual,device=dev))
            ret.update(h=h[0].cpu(),ids=candidates.cpu(),mask=mask.cpu(),action_h=ends[1:].cpu())
        subset,q,mass=measure(logp,candidates,mask)
        compact[mode]=subset
        ret['mass_'+mode]=mass.cpu()
        if mode!='selfcheck':
            scores=[distribution(model,hs).gather(1,torch.tensor(seq,device=dev)[:,None]).mean().item() for hs,seq in zip(h[1:],seqs[1:])]
            ret['action_'+mode]=torch.tensor(scores)
        if mode=='none':qs=q
    s,sel,ph=map(compact.get,('none','full','placeholder'))
    d=centered(sel-ph,qs);old=centered(sel-s,qs)
    def prob(z):return z.masked_fill(~mask,-torch.inf).softmax(1)
    err=(prob(s+d)-prob(s+sel-ph)).abs().max().item()
    assert err<=1e-5 and d.isfinite().all()
    zero=(centered(compact['selfcheck']-ph,qs).abs().max().item() if repeat else None)
    assert zero is None or zero<=1e-5
    ret.update(s=s.cpu(),q=qs.cpu(),d=d.cpu(),old=old.cpu(),identity_error=err,self_error=zero,
               goal=row['goal'],step=row['step'],page=row['page'],classes=row['classes'],actions=actions)
    return ret

def worker(rank,pilot):
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16).eval().cuda()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    head=RUN/'head.pt'
    if rank==0 and not head.exists():
        torch.save(model.lm_head.weight.detach().cpu(),head.with_suffix('.tmp'));head.with_suffix('.tmp').rename(head)
    tasks=read(PARENT/'asin_split_manifest.json')['T_cal'];idx={t['goal_index']:i for i,t in enumerate(tasks)}
    folder=RUN/'features';folder.mkdir(exist_ok=True)
    for p in sorted(RUN.glob('tape_*.jsonl')):
        for line in p.open():
            row=json.loads(line);i=idx[row['goal']]
            if (i<8)!=pilot or i%4!=rank:continue
            out=folder/f"{i:03d}_{row['step']:02d}.pt"
            if out.exists():continue
            if any(row['na'].values()):raise ValueError('SCORER_FAILURE: '+str(row['na']))
            data=score_row(model,tok,row,pilot)
            tmp=out.with_suffix('.tmp');torch.save(data,tmp);tmp.rename(out)

def score(pilot):
    start=time.monotonic();ps=[]
    try:
        for j in range(4):
            cmd=[sys.executable,'-m',__spec__.name,'--rank',str(j)]+(['--pilot'] if pilot else [])
            ps.append(subprocess.Popen(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(j)}))
        codes=[p.wait() for p in ps]
        if any(codes):raise RuntimeError(codes)
    finally:
        for p in ps:
            if p.poll() is None:p.terminate();p.wait()
        append(RUN/'b_cost.jsonl',dict(stage='three_view_scoring',pilot=pilot,gpu_hours=4*(time.monotonic()-start)/3600))
    if pilot:
        fs=[torch.load(p,weights_only=True) for p in sorted((RUN/'features').glob('*.pt'))]
        assert len({x['goal'] for x in fs})==8
        save(RUN/'p0.json',dict(pass_=True,**{'pass':True},tasks=8,steps=len(fs),max_identity_error=max(x['identity_error'] for x in fs),max_self_error=max(x['self_error'] for x in fs),expected_cost_factor=1.5))
if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.rank is None:score(a.pilot)
    else:worker(a.rank,a.pilot)
