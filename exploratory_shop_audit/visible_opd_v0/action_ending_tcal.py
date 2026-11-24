"""Compare both OOF projectors and two packet labels at the same ending candidates."""
from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
import torch

from .action_ending import after_role, candidate_classes, locate, vocabulary
from .action_ending_train import DEST
from .b_collect import PARENT, RUN as B
from .gap_cpu_actions import rows
from .gap_cpu_packet import cluster_mean, DRAWS, SEED
from .gap_score import RUN
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save

COLUMNS = ['full_minus_none_centered','full_minus_placeholder_centered','huber_oof_times_b','kl_oof_times_lambda']


def oof(paths,n,index_col,value_col):
    pred = np.empty((n,9),dtype=np.float32)
    folds = np.full(n,-1)
    for fold,path in enumerate(paths):
        with np.load(path) as f:
            ix = f[index_col]
            assert len(np.unique(ix))==len(ix) and (folds[ix]==-1).all()
            pred[ix] = f[value_col]
            folds[ix] = fold
    assert (folds>=0).all() and np.isfinite(pred).all()
    return pred,folds


def summary(records):
    vals = [r[-1] for r in records]
    result = dict(n=len(vals),tasks=len({r[0] for r in records}),mean=None,median=None,positive_share=None)
    if vals:
        x = np.asarray(vals)
        result.update(mean=x.mean(0).tolist(),median=np.median(x,axis=0).tolist(),positive_share=(x>0).mean(0).tolist())
        stats = cluster_mean([r[0] for r in records],np.column_stack((x,x>0)))
        ci = stats['ci95']
        result.update(mean_ci95=ci[:x.shape[1]] if ci else None,positive_share_ci95=ci[x.shape[1]:] if ci else None)
    return result


def main():
    tokens,eos = vocabulary()
    folder = RUN/'initial'
    split_path = PARENT/'asin_split_manifest.json'
    split = read(split_path)
    tasks = split['T_cal']
    order = {t['goal_index']:i for i,t in enumerate(tasks)}
    goal_fold = {t['goal_index']:split['cal_folds'][t['asin']] for t in tasks}
    files = sorted(B.glob('tape_*.jsonl'))
    source = {p.name:digest_file(p) for p in files}
    source[split_path.name] = digest_file(split_path)
    refmap,ordered = {},[[] for _ in range(4)]
    for path in files:
        for r in rows(path):
            g,k = r['goal'],r['step']
            assert (g,k) not in refmap and r['rep']==0 and not any(r['na'].values())
            refmap[g,k] = (r['response'],r['classes'])
            ordered[order[g]%4].append((g,k))
    n = sum(len(ans) for ans,_ in refmap.values())
    assert len(order)==512 and n==34825
    parts,idx_new = [],{}
    offset = 0
    for rank,keys in enumerate(ordered):
        f = folder/f'features_{rank}.pt'
        receipt = read(folder/f'score_{rank}.json')
        assert digest_file(f)==receipt['sha'] and receipt['records']==len(keys) and not receipt['skipped']
        assert {p.name:source[p.name] for p in files}=={key.rsplit('/',1)[-1]:v for key,v in receipt['input_hashes'].items()}
        source['initial/'+f.name] = receipt['sha']
        data = torch.load(f,map_location='cpu',weights_only=True,mmap=True)
        # Hold only support and ordering for this analysis.
        parts.append((data['ids'].numpy(),data['mask'].numpy()))
        local = 0
        for g,k in keys:
            ans,cls = refmap[g,k]
            size = len(ans); sl = slice(local,local+size)
            assert list(data['goal'][sl])==[g]*size and list(data['cls'][sl])==cls
            assert list(data['fold'][sl])==[goal_fold[g]]*size
            idx_new[g,k] = offset+local
            local += size
        assert local==len(parts[-1][0])
        offset += local
    assert offset==n
    ids_new,mask_new = map(np.vstack,zip(*parts))
    old_f = [B/f'fold_{i}.npz' for i in range(4)]
    new_f = [folder/f'oof_{i}.npz' for i in range(4)]
    for label,fileset in [('B',old_f),('initial',new_f)]:
        source.update({label+'/'+f.name:digest_file(f) for f in fileset})
    old_pred,old_fold = oof(old_f,n,'idx','content')
    new_pred,new_fold = oof(new_f,n,'ix','pred')
    hp,kp = B/'reports/calibration_strength_report.json',folder/'calibration.json'
    huber = {k:v['b'] for k,v in read(hp)['classes'].items()}
    kl = read(kp)['classes']
    source['B/'+hp.name],source['initial/'+kp.name] = digest_file(hp),digest_file(kp)
    matched,strict,only_old,only_new = [],[],[],[]
    native_old,native_new = [],[]
    counts = defaultdict(Counter)
    seen = set()
    feature_hash = sha256()
    offset = 0
    for f in sorted((B/'features').glob('*.pt')):
        feature_hash.update((f.name+' '+digest_file(f)+'\n').encode())
        r = torch.load(f,map_location='cpu',mmap=True,weights_only=True)
        g,k = r['goal'],r['step']
        assert (g,k) not in seen
        seen.add((g,k))
        ans,cls = refmap[g,k]
        assert cls==r['classes']
        ni = idx_new[g,k]
        assert (old_fold[offset:offset+len(ans)]==goal_fold[g]).all()
        assert (new_fold[ni:ni+len(ans)]==goal_fold[g]).all()
        ids,mask = r['ids'].numpy(),r['mask'].numpy()
        actual = np.array(ans)[:,None]
        assert np.array_equal(ids[:,-1],actual[:,0]) and np.array_equal(ids_new[ni:ni+len(ans),-1],actual[:,0])
        assert np.all(((ids==actual)&mask).sum(1)==1)
        assert np.all(((ids_new[ni:ni+len(ans)]==actual)&mask_new[ni:ni+len(ans)]).sum(1)==1)
        parts = [tokens[i] for i in ans]
        pos,_,why = locate(parts)
        sampled = after_role(parts[pos[0].index][len(pos[0].prefix):]) if pos else None
        counts['replies'][why]+=1
        for p in pos:
            j = p.index
            it,jt = offset+j,ni+j
            oi = np.flatnonzero(mask[j]); nj = np.flatnonzero(mask_new[jt])
            oc,_ = candidate_classes(p,ids[j,oi],tokens,eos)
            nc,_ = candidate_classes(p,ids_new[jt,nj],tokens,eos)
            old = {int(ids[j,oi[c]]):int(oi[c]) for c in oc}
            new = {int(ids_new[jt,nj[c]]):int(nj[c]) for c in nc}
            identical = set(ids[j,oi])==set(ids_new[jt,nj])
            names = [p.role]
            if p.role=='bracket':
                names.append(sampled.replace('after_','bracket_sampled_'))
            for name in names:
                stats = counts[name]
                stats.update(['located'])
                stats['same_full_support']+=identical
                stats['old_stop_present']+=bool(old)
                stats['new_stop_present']+=bool(new)
                stats['common_stop_present']+=bool(old.keys() & new.keys())
            for tid in sorted(old.keys()|new.keys()):
                a,b = old.get(tid),new.get(tid)
                key=(g,k,j,p.role,tid,sampled)
                if a is not None:
                    left=[float(r['old'][j,a]),float(r['d'][j,a]),float(old_pred[it,a])*huber[cls[j]]]
                    native_old.append((*key,left))
                if b is not None:
                    right=[float(new_pred[jt,b])*kl[cls[j]]]
                    native_new.append((*key,right))
                if a is None:
                    only_new.append((*key,right))
                elif b is None:
                    only_old.append((*key,left))
                else:
                    row=(*key,left+right)
                    matched.append(row)
                    if identical:
                        strict.append(row)
        offset+=len(ans)
    assert offset==n and seen==set(refmap)
    assert len(native_old)==len(matched)+len(only_old) and len(native_new)==len(matched)+len(only_new)
    positions = []
    for name in ('bracket','following','after_bare','after_fused_newline','after_other_suffix',
                 'bracket_sampled_fused_newline','bracket_sampled_bare'):
        def take(row):
            if name.startswith('bracket_sampled_'):
                return row[3]=='bracket' and row[5]==name.replace('bracket_sampled_', 'after_')
            return row[3]==name or (name=='following' and row[3]!='bracket')
        cc = (sum((v for r,v in counts.items() if r.startswith('after_')),Counter())
              if name=='following' else counts[name])
        positions.append(dict(position=name,coverage=dict(cc),**summary([r for r in matched if take(r)]),
                              identical_support_only=summary([r for r in strict if take(r)])))
    by_id = []
    for tid in sorted({r[4] for r in matched}):
        by_id.append(dict(token_id=tid,token=tokens[tid].decode(errors='backslashreplace'),
                          **summary([r for r in matched if r[4]==tid])))
    save(DEST/'tcal_action_ending.json',dict(columns=COLUMNS,tasks=len(order),all_positions=n,
         replies=len(seen),reply_coverage=dict(counts['replies']),seed=SEED,task_bootstrap_draws=DRAWS,
         unit='one matched ending candidate per observed position and vocabulary ID',overall=summary(matched),
         positions=positions,by_token=by_id,identical_support_only=summary(strict),
         native_B=summary(native_old),native_block1=summary(native_new),
         old_only_candidates=len(only_old),new_only_candidates=len(only_new),
         shrinkage=dict(huber=huber,kl=kl),source_sha256=source,feature_files_digest=feature_hash.hexdigest()))


if __name__=='__main__':
    main()
