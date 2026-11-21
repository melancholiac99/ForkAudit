"""Frozen per-update KL and ten-step full-vocabulary and projector diagnostics."""
import os
import time
import torch
from transformers import AutoTokenizer
from .c_targets import actor,response,STAGE_B
from .e25_layout import Layout
from .e25_probe import GROUPS,SEED,add,baseline_logits,full_metrics,labels,pooled,RUN,PUBLIC
from .e25_refresh import fold_of
from .d_data import PARENT
from .s21_data import SmokeBatch,summarize
from .s21_store import RUN as S21
from .s22_measure import fp32_scores,restore,shard_order
from .s22_report import checkpoint
from .post_d_precision import reanchor
from .score import support
from exploratory_shop_audit.grpo_forced_audit.common import MODEL,read,save,digest_file

def fp32_probe_actor(layout):
    from .e25_fp32_cache import verified_path
    path=verified_path() if layout.name=='formal_2512' else MODEL
    start=time.monotonic()
    model=actor(dtype=torch.float32,base_path=path,load_on_rank=bool(os.environ.get('OPD_MIGRATION_GROUP'))).eval()
    print(f'FP32 actor loaded from {path} in {time.monotonic()-start:.2f}s',flush=True)
    return model


def at(model,layout,k,a):
    if k:layout.snapshot(model,k,a)
    else:
        p=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
        initial=torch.load(S21/'original/initial.pt',weights_only=True,map_location='cpu')
        restore(p,initial,initial)

@torch.inference_mode()
def per_update(rank,l,k,arms):
    model=fp32_probe_actor(l)
    for a in arms:
        path=l.folder(k,a);out=path/f'fp32_part_{rank}.json'
        if out.exists():continue
        batch,q=l.batch(k,a);ids=shard_order(batch,rank)
        at(model,l,k-1,a);old=fp32_scores(model,batch,ids)
        at(model,l,k,a);new=fp32_scores(model,batch,ids)
        if k==1 and l.name!='formal_2512':bf=checkpoint('bf16','initial')[0]
        else:
            p=path/'bf16_before.pt';assert digest_file(p)==read(path/'target_manifest.json')['bf16_before_sha256']
            bf=torch.load(p,weights_only=True)
        target={i:reanchor(bf[i],old[i],q[i],batch.rows[i]['mask']) for i in ids}
        save(out,summarize(batch,old,new,target,ids))

def merge_update(l,k,arms):
    for a in arms:
        p=l.folder(k,a);rows=[read(p/f'fp32_part_{r}.json') for r in range(4)]
        assert abs(sum(r['global_weight_mass'] for r in rows)-1)<1e-10
        v={key:sum(r[key]*r['global_weight_mass'] for r in rows) for key in ('kl_before','kl_after','delta_kl','alignment','movement')}
        bf=read(p/'optimizer.json')['bf16_fixed_target']
        assert abs(v['delta_kl']-v['alignment']+v['movement'])<1e-10 and abs(bf['delta_kl']-bf['alignment']+bf['movement'])<1e-9
        save(p/'measurement.json',dict(arm=a,update=k,bf16=bf,fp32_reanchored=v))

@torch.inference_mode()
def common(rank,l,k,arms):
    b=SmokeBatch()
    if l.name=='formal_2512':
        from .e2512_routing import owners
        chunks=pooled(b,rank,owners())
    else:chunks=pooled(b,rank)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    cats=[labels(tok,r) for r in b.rows]
    sp=read(PARENT/'asin_split_manifest.json')
    folds={t['goal_index']:fold_of(t['asin'],sp['cal_folds']) for t in sp['T_opd']}
    cal=read(STAGE_B/'reports/calibration_strength_report.json')['classes']
    for a in arms:
        out=l.folder(k,a)/f'common_{rank}.json'
        if out.exists():continue
        model=actor().eval();at(model,l,k,a)
        reference_root=l.root if l.name=='formal_2512' else RUN
        if l.name=='formal_2512' and k==0:
            base=baseline_logits(model,b,chunks,rank,torch.bfloat16,root=reference_root)
        else:
            base=torch.load(reference_root/'base_logits'/str(torch.bfloat16)/f'rank_{rank}.pt',weights_only=True,mmap=True)
        prev=curr=None
        if a=='P':
            prev=l.projectors(model,k,a);curr=l.projectors(model,k+1,a)
        totals={g:[0,0.,0.,0.,0.] for g in ('all',*GROUPS)}
        per_id={}; proj=torch.zeros(4,dtype=torch.float64,device='cuda')
        for ix in chunks:
            i=ix[0];r=b.rows[i];h,raw=response(model,r['prompt'],r['answer'])
            per_id[i]=raw.bfloat16().cpu()
            v=full_metrics(base[i].cuda(),raw)
            for j in ix:
                group=cats[j];add(totals,'all',v,list(range(len(group))),b.weights[j])
                for g in set(group):add(totals,g,v,[t for t,x in enumerate(group) if x==g],b.weights[j])
            if prev is not None:
                ids,mask=support(raw,torch.tensor(r['answer'],device='cuda'))
                s=raw.gather(1,ids);q=s.masked_fill(~mask,-torch.inf).softmax(-1)
                head=model.get_base_model().lm_head.weight[ids]
                for j in ix:
                    row=b.rows[j];coef=torch.tensor([cal[c]['b'] for c in row['classes']],device='cuda')[:,None]
                    f=folds[row['goal']]
                    u=(prev[f](h,head,q)*coef).double();w=(curr[f](h,head,q)*coef).double()
                    proj+=b.weights[j]*torch.stack(tuple((q*x).sum() for x in ((u-w).square(),u*w,u.square(),w.square())))
        del model,prev,curr,base
        torch.cuda.empty_cache()
        model=fp32_probe_actor(l);at(model,l,k,a)
        cross={g:[0.,0.] for g in totals}
        fp_base = None
        if l.name=='calibration_2511':
            # Reuse the signed preflight's step-zero reference; the loaded actor is already at step k.
            reference=RUN/'base_logits'/str(torch.float32)/f'rank_{rank}.pt'
            signature=read(PUBLIC/'common_reference_signatures_2511.json')['files']
            assert digest_file(reference)==signature[str(reference.relative_to(RUN/'base_logits'))]
            fp_base=torch.load(reference,weights_only=True,mmap=True,map_location='cpu')
        fp_move = {g:0. for g in totals}
        for ix in chunks:
            i=ix[0];r=b.rows[i];raw=response(model,r['prompt'],r['answer'])[1]
            v=full_metrics(per_id[i].cuda(),raw)[:,0].tolist()
            fv=full_metrics(fp_base[i].cuda(),raw)[:,0].tolist() if fp_base is not None else None
            for j in ix:
                for pos,(g,val) in enumerate(zip(cats[j],v,strict=True)):
                    for key in (g,'all'):
                        cross[key][0]+=b.weights[j]*val;cross[key][1]+=b.weights[j]
                        if fv is not None:fp_move[key]+=b.weights[j]*fv[pos]
        save(out,dict(stats=totals,cross=cross,fp32_movement=fp_move if fp_base is not None else None,projector_moments=proj.tolist(),compared_versions=[l.version(k,a),l.version(k+1,a)] if a=='P' else None))
        del model,per_id;torch.cuda.empty_cache()

def merge_common(l,k,arms,dest=None):
    result={}
    for a in arms:
        rows=[read(l.folder(k,a)/f'common_{r}.json') for r in range(4)]
        stats={}
        for g in rows[0]['stats']:
            n,w,kl,h,p=map(sum,zip(*(r['stats'][g] for r in rows)))
            x,z=map(sum,zip(*(r['cross'][g] for r in rows)))
            assert abs(w-z)<1e-10
            stats[g]=dict(positions=n,mass=w,kl=kl/w if w else None,entropy=h/w if w else None,
                          top1=p/w if w else None,cross_precision=x/w if w else None)
            if l.name=='calibration_2511':
                stats[g]['fp32_kl']=sum(r['fp32_movement'][g] for r in rows)/w if w else None
        assert stats['all']['positions']==4448 and abs(stats['all']['mass']-1)<1e-10
        mse,dot,n0,n1=map(sum,zip(*(r['projector_moments'] for r in rows)))
        decision=[stats[g] for g in ('click','option')]
        mass=sum(g['mass'] for g in decision)
        stats['decision_click_option']=dict(positions=sum(g['positions'] for g in decision),mass=mass,
             **{f:sum(g['mass']*g[f] for g in decision if g['mass'])/mass for f in ('kl','entropy','top1','cross_precision')})
        result[a]=dict(stats=stats,projector_mse=mse if a=='P' else None,projector_cosine=dot/(n0*n1)**.5 if n0*n1 else None,
                       versions=rows[0]['compared_versions'])
    save(dest or l.root/'monitor'/f'{k:03}.json',result)
    return result
