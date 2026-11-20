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
def build(k,arm):
    path = folder(k,arm)
    if (path/'target_manifest.json').exists():return
    model=actor().eval()
    snapshot(model,k-1,arm)
    sha=state_hash([(n,p) for n,p in model.named_parameters() if p.requires_grad])
    assert sha==read(folder(k-1,arm)/'optimizer.json')['after_sha']
    tasks=read(path.parent/'tasks.json')
    routing=read(PARENT/'asin_split_manifest.json')['cal_folds']
    folds={t['goal_index']:routing.get(t['asin'],stable_seed(SEED,'projection_fold',t['asin'])%4) for t in tasks}
    nets=projections(model,folds)
    cal=read(STAGE_B/'reports/calibration_strength_report.json')['classes']
    rows=[]; targets=[]; episodes=Counter(); done=set()
    for rep in range(2):
        for r in read_rows(path/f'rollout_{rep}.jsonl'):
            key=(r['goal'],rep)
            assert r['goal'] in folds and r['rep']==rep and r['step']==episodes[key] and key not in done
            episodes[key]+=1
            if r['done']:done.add(key)
            assert not any(r['na'].values()),(key,r['na'])
            prompt=r['views']['none'];answer=r['response']
            h,logits=response(model,prompt,answer)
            ids,mask=support(logits,torch.tensor(answer,device='cuda'))
            s=logits.gather(1,ids)
            q=candidate_target(s,torch.zeros_like(s),1,mask)
            def subset(view):
                return response(model,r['views'][view],answer)[1].log_softmax(-1).gather(1,ids)
            # Three detached forward passes in every arm. Projected targets read visible tensors only.
            d=content_correction(subset('full'),subset('placeholder'),q)
            if arm in ('P','P-shuffled'):
                t=visible_targets(h,model.get_base_model().lm_head.weight,ids,s,mask,q,nets,folds[r['goal']],cal,r['classes'])[arm]
            else:
                scale=1. if arm=='R' else torch.tensor([cal[c]['beta']['scale'] for c in r['classes']],device='cuda')[:,None]
                t=candidate_target(s,d,scale,mask)
            assert not t.requires_grad and t.isfinite().all() and (t[~mask]==0).all()
            rows.append(dict(goal=r['goal'],rep=rep,step=r['step'],prompt=prompt,answer=answer,
                ids=ids.cpu(),mask=mask.cpu(),classes=r['classes'],snapshot=q.cpu()))
            targets.append(t.cpu())
    assert done==set(episodes)=={(g,rep) for g in folds for rep in range(2)}
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['goal'],rows[i]['rep'],rows[i]['step']))
    for name,values in [('student',rows),('target',targets)]:
        torch.save([values[i] for i in order],path/f'{name}.pt')
    save(path/'target_manifest.json',dict(scorer_sha=sha,goals=sorted(folds),projection_age=k-1,
        student_sha256=digest_file(path/'student.pt'),target_sha256=digest_file(path/'target.pt')))

if __name__=='__main__':
    import sys
    os.umask(0o077)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    build(int(sys.argv[1]),ARMS[rank])
