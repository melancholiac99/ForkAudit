"""Compare four frozen distributions on the same candidate set at each T_cal position."""
from collections import Counter
from hashlib import sha256

import numpy as np
import torch

from .action_ending import locate, vocabulary
from .action_ending_tcal import oof
from .b_collect import PARENT, RUN as B
from .gap_cpu_actions import rows
from .gap_cpu_packet import cluster_mean, DRAWS, SEED, log_probs
from .gap_score import PUBLIC, RUN
from .projection import VisibleProjection
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save

MODELS=('student','full_teacher','kl_projector','huber_projector')
DEST=PUBLIC/'first_command_replay'


def quantities(logits,mask):
    lp=log_probs(logits,mask)
    prob=np.where(mask,np.exp(lp),0.)
    return -(prob*lp).sum(1),prob.max(1)


def measures(s,t,kl,huber,mask):
    pairs=[quantities(v,mask) for v in (s,t,s+kl,s+huber)]
    entropy=np.column_stack([v[0] for v in pairs])
    top=np.column_stack([v[1] for v in pairs])
    return np.column_stack([entropy,top,entropy[:,1:]-entropy[:,0:1],top[:,1:]-top[:,0:1]])


def grouped(vals,goals,classes,roles):
    groups={'all':np.ones(len(vals),bool)}
    groups.update({c:classes==c for c in sorted(set(classes))})
    groups.update({r:roles==r for r in ('bracket','following')})
    result={}
    for name,mask in groups.items():
        stat=cluster_mean(goals[mask],vals[mask])
        pos=0
        item=dict(n=stat['n'],tasks=stat['tasks'])
        for key,models in (('entropy_nats',MODELS),('top1',MODELS),('delta_entropy_nats',MODELS[1:]),('delta_top1',MODELS[1:])):
            item[key]={m:dict(mean=stat['mean'][i],ci95=stat['ci95'][i]) for i,m in enumerate(models,pos)}
            pos+=len(models)
        result[name]=item
    return result


@torch.inference_mode()
def main():
    folder=RUN/'initial'
    sources={}
    def track(name,path):
        sources[name]=digest_file(path)
        return path
    split=read(track('split',PARENT/'asin_split_manifest.json'))
    tasks=split['T_cal']
    order={r['goal_index']:i for i,r in enumerate(tasks)}
    folds={r['goal_index']:split['cal_folds'][r['asin']] for r in tasks}
    tapes=sorted(B.glob('tape_*.jsonl'))
    rowmap={}; shards=[[] for _ in range(4)]
    tokens,_=vocabulary()
    for path in tapes:
        for r in rows(track(path.name,path)):
            key=r['goal'],r['step']
            assert key not in rowmap and r['rep']==0 and not any(r['na'].values())
            ans=r['response']
            pos,_,_=locate([tokens[i] for i in ans])
            roles=['none']*len(ans)
            for p in pos:roles[p.index]='bracket' if p.role=='bracket' else 'following'
            rowmap[key]=ans,r['classes'],roles
            shards[order[key[0]]%4].append(key)
    n=sum(len(v[0]) for v in rowmap.values())
    assert n==34825 and len(tasks)==512
    old,oldfold=oof([track('B/fold'+str(i),B/f'fold_{i}.npz') for i in range(4)],n,'idx','content')
    new,newfold=oof([track('initial/oof'+str(i),folder/f'oof_{i}.npz') for i in range(4)],n,'ix','pred')
    hp=read(track('B/scales',B/'reports/calibration_strength_report.json'))['classes']
    cal=read(track('initial/scales',folder/'calibration.json'))
    head=torch.load(track('B/head',B/'head.pt'),mmap=True,weights_only=True,map_location='cpu')
    def net(path,label):
        model=VisibleProjection(head.shape[1]).cpu().eval()
        model.load_state_dict(torch.load(track(label,path),weights_only=True,map_location='cpu'))
        return model
    kn=[net(folder/f'fold_{i}.pt','initial/net'+str(i)) for i in range(4)]
    hn=[net(B/f'projector_versions/content_{i}.pt','B/net'+str(i)) for i in range(4)]
    assert [sources['initial/net'+str(i)] for i in range(4)]==cal['folds']
    native={}; offset=0
    for rank,keys in enumerate(shards):
        feat=track('initial/features'+str(rank),folder/f'features_{rank}.pt')
        receipt=read(folder/f'score_{rank}.json')
        assert sources['initial/features'+str(rank)]==receipt['sha'] and receipt['records']==len(keys)
        assert {p.name:sources[p.name] for p in tapes}=={k.rsplit('/',1)[-1]:v for k,v in receipt['input_hashes'].items()}
        data=torch.load(feat,mmap=True,weights_only=True,map_location='cpu')
        idx=0
        for g,k in keys:
            ans,cls,_=rowmap[g,k]; length=len(ans); sl=slice(idx,idx+length)
            assert data['goal'][sl]==[g]*length and data['cls'][sl]==cls
            assert data['fold'][sl]==[folds[g]]*length and (newfold[offset+idx:offset+idx+length]==folds[g]).all()
            native[g,k]=(data,sl,offset+idx)
            idx+=length
        assert idx==len(data['ids'])
        offset+=idx
    assert offset==n
    gvec,cvec,rvec=[],[],[]
    stage_values,initial_values=[],[]
    errors=Counter(); same=0; offset=0
    feature_hash=sha256()
    for path in sorted((B/'features').glob('*.pt')):
        feature_hash.update((path.name+' '+digest_file(path)+'\n').encode())
        b=torch.load(path,mmap=True,weights_only=True,map_location='cpu')
        g,k=b['goal'],b['step']
        a,sl,start=native.pop((g,k))
        ans,cls,roles=rowmap[g,k]; size=len(ans); f=folds[g]
        assert b['classes']==cls and (oldfold[offset:offset+size]==f).all()
        ks=np.array([cal['classes'][c] for c in cls])[:,None]
        hs=np.array([hp[c]['b'] for c in cls])[:,None]
        for base in (True,False):
            if base:
                ids,mask,s,h,q=(b[name] for name in ('ids','mask','s','h','q'))
                slt=s.double().numpy()+b['old'].double().numpy()
            else:
                ids,mask,s,h,q=(a[name][sl] for name in ('ids','mask','s','h','q'))
                slt=a['t'][sl].double().numpy()
            assert ids[:,-1].tolist()==ans
            # Evaluate each fold's frozen network on this exact set, including added sampled tokens.
            embeds=head[ids]
            kl=kn[f](h,embeds,q).numpy()
            hu=hn[f](h,embeds,q).numpy()
            idxs=mask.numpy()
            key='huber_oof_replay' if base else 'kl_oof_replay'
            reference=old[offset:offset+size] if base else new[start:start+size]
            replay=hu if base else kl
            errors[key]=max(errors[key],float(abs(reference-replay)[idxs].max()))
            # Preserve the original OOF output when its native support is being used.
            if base:hu=reference
            else:kl=reference
            values=measures(s.double().numpy(),slt,kl*ks,hu*hs,idxs)
            (stage_values if base else initial_values).append(values)
        for j in range(size):
            old_ids=b['ids'][j][b['mask'][j]].tolist()
            new_ids=a['ids'][sl][j][a['mask'][sl][j]].tolist()
            same+=set(old_ids)==set(new_ids)
        gvec.extend([g]*size);cvec.extend(cls);rvec.extend(roles);offset+=size
    assert not native and offset==n
    info={k:float(v) for k,v in errors.items()}
    assert max(info.values())<.005,info
    c,g,r=np.array(cvec),np.array(gvec),np.array(rvec)
    assert sum(r=='bracket')==sum(r=='following')==2143
    save(DEST/'tcal_entropy.json',dict(positions=n,tasks=len(tasks),seed=SEED,task_cluster_draws=DRAWS,
        same_candidate_sets=int(same),task_weight='equal positions; resample whole tasks',
        primary_stage_B_support=grouped(np.vstack(stage_values),g,c,r),
        sensitivity_block1_support=grouped(np.vstack(initial_values),g,c,r),
        top1_definition='maximum probability of each distribution on the same support; paired target minus student',
        oof_replay_max_abs=info,source_sha256=sources,feature_digest=feature_hash.hexdigest()))


if __name__=='__main__':
    main()
