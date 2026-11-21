"""Build current BF16 detached targets; keep privileged views outside the actor input."""
import os
from collections import Counter, defaultdict
import torch
from .c_targets import actor, projections, visible_targets, response, STAGE_B, SEED
from .c_smoke import state_hash
from .score import support
from .content_target import content_correction
from .projection import candidate_target
from .d_data import folder, snapshot, PARENT, SMOKE
from .s21_store import ARMS
from exploratory_shop_audit.grpo_forced_audit.common import read, read_rows, save, stable_seed, digest_file

@torch.inference_mode()
def build(k,arm,layout=None,*,goals=None,output=None):
    loc=folder if layout is None else layout.folder
    source = loc(k,arm)
    path = source if output is None else output
    path.mkdir(parents=True,exist_ok=True)
    if (path/'target_manifest.json').exists():return
    model=actor().eval()
    (snapshot if layout is None else layout.snapshot)(model,k-1,arm)
    sha=state_hash([(n,p) for n,p in model.named_parameters() if p.requires_grad])
    if k>1:assert sha==read(loc(k-1,arm)/'optimizer.json')['after_sha']
    else:assert sha==read(SMOKE/'arm_R.json')['initial_adapter_sha256']
    tasks=read(source.parent/'tasks.json')
    if goals is not None:tasks=[t for t in tasks if t['goal_index'] in goals]
    routing=read(PARENT/'asin_split_manifest.json')['cal_folds']
    folds={t['goal_index']:routing.get(t['asin'],stable_seed(SEED,'projection_fold',t['asin'])%4) for t in tasks}
    nets=projections(model,folds) if layout is None else layout.projectors(model,k,arm)
    cal=read(STAGE_B/'reports/calibration_strength_report.json')['classes']
    # Cache only completely identical texts, response boundaries, classes and fold routing on this snapshot.
    cache={};verified_duplicate=False;reused=0
    masses=[]
    na=[]; rows=[]; targets=[]; student_logits=[]; episodes=Counter(); done=set()
    for rep in range(2):
        for r in read_rows((SMOKE if k==1 else source)/f'rollout_{rep}.jsonl'):
            if goals is not None and r['goal'] not in goals:continue
            key=(r['goal'],rep)
            assert r['goal'] in folds and r['rep']==rep and r['step']==episodes[key] and key not in done
            episodes[key]+=1
            if r['done']:done.add(key)
            if any(r['na'].values()):
                assert all(why in (None,'CONTEXT_OVERFLOW','PACKET_OVERFLOW') for why in r['na'].values()),r['na']
                na.append(dict(goal=key[0],rep=rep,step=r['step'],page=r['page'],
                               reasons=r['na'],positions=len(r['response']),classes=dict(Counter(r['classes']))))
                continue
            prompt=r['views']['none'];answer=r['response']
            key=(r['goal'],tuple(prompt),tuple(answer),tuple(r['views']['full']),
                 tuple(r['views']['placeholder']),tuple(r['classes']))
            prior=cache.get(key)
            if prior is not None and verified_duplicate:
                rows.append(dict(rows[prior],rep=rep,step=r['step']))
                targets.append(targets[prior]);student_logits.append(student_logits[prior])
                if layout is not None:masses.append(masses[prior])
                reused+=1
                continue
            h,logits=response(model,prompt,answer)
            ids,mask=support(logits,torch.tensor(answer,device='cuda'))
            s=logits.gather(1,ids)
            q=candidate_target(s,torch.zeros_like(s),1,mask)
            def subset(view):
                return response(model,r['views'][view],answer)[1].log_softmax(-1).gather(1,ids)
            # Three detached forward passes in every arm. Projected targets read visible tensors only.
            selected,placeholder=subset('full'),subset('placeholder')
            d=content_correction(selected,placeholder,q)
            if layout is not None:
                base_log=s-logits.logsumexp(-1,keepdim=True)
                masses.append(torch.stack([(v.exp()*mask).sum(-1) for v in (base_log,selected,placeholder)],-1).cpu())
            if layout is not None and arm in ('P','P-KL'):
                t=layout.projected(nets,arm,h,model.get_base_model().lm_head.weight,ids,s,mask,q,folds[r['goal']],cal,r['classes'])
            elif arm in ('P','P-shuffled'):
                t=visible_targets(h,model.get_base_model().lm_head.weight,ids,s,mask,q,nets,folds[r['goal']],cal,r['classes'])[arm]
            else:
                scale=1. if arm in ('R','R-matched') else torch.tensor([cal[c]['beta']['scale'] for c in r['classes']],device='cuda')[:,None]
                t=candidate_target(s,d,scale,mask)
            assert not t.requires_grad and t.isfinite().all() and (t[~mask]==0).all()
            rows.append(dict(goal=r['goal'],rep=rep,step=r['step'],prompt=prompt,answer=answer,
                ids=ids.cpu(),mask=mask.cpu(),classes=r['classes'],snapshot=q.cpu()))
            targets.append(t.cpu())
            student_logits.append(s.cpu())
            if prior is not None:
                for field in ('ids','mask','snapshot'):
                    assert torch.equal(rows[-1][field],rows[prior][field])
                assert torch.equal(student_logits[-1],student_logits[prior]) and torch.equal(targets[-1],targets[prior])
                verified_duplicate=True
            cache[key]=len(rows)-1
    assert done==set(episodes)=={(g,rep) for g in folds for rep in range(2)}
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['goal'],rows[i]['rep'],rows[i]['step']))
    for name,values in [('student',rows),('target',targets),('bf16_before',student_logits)]:
        torch.save([values[i] for i in order],path/f'{name}.pt')
    extra={}
    if layout is not None:
        mass_file=path/'candidate_mass.pt'
        torch.save([masses[i] for i in order],mass_file)
        extra['mass_sha256']=digest_file(mass_file)
        if k==1 and arm!='P-KL':
            b,original=layout.batch(k,arm)
            ref=[j for j,row in enumerate(b.rows) if goals is None or row['goal'] in goals]
            assert len(order)==len(ref)
            for j,i in zip(ref,order,strict=True):
                assert torch.equal(rows[i]['ids'],b.rows[j]['ids']) and torch.equal(targets[i],original[j])
    save(path/'target_manifest.json',dict(**extra,scorer_sha=sha,na_records=na,rollout_records=sum(episodes.values()),identical_records_reused=reused,duplicate_verified=verified_duplicate,goals=sorted(folds),projection_age=k-1 if layout is None else k-1-layout.version(k,arm),
        projector_fit_sha=(layout.projector_sha(k,arm) if layout is not None else None),
        student_sha256=digest_file(path/'student.pt'),bf16_before_sha256=digest_file(path/'bf16_before.pt'),target_sha256=digest_file(path/'target.pt')))

if __name__=='__main__':
    import sys
    os.umask(0o077)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    build(int(sys.argv[1]),ARMS[rank])
