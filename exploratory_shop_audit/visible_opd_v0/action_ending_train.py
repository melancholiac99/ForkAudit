"""§26 current-actor vs archived target at the end of the first command, CPU only."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch

from .action_ending import after_role, candidate_classes, locate, vocabulary
from .gap_cpu_actions import FORMAL, legality, rows, tapes
from .gap_cpu_packet import DRAWS, OUT, SEED, cluster_mean
from exploratory_shop_audit.grpo_forced_audit.common import MODEL, digest_file, read, save

DEST = OUT/'action_ending'
ROLES = ('all','bracket','following','after_bare','after_fused_newline','after_other_suffix',
         'bracket_sampled_fused_newline','bracket_sampled_bare')
COLUMNS = ['q_stop','actor_stop','q_stop_lt_actor_stop','q_continue','actor_continue']


def record_key(row):
    return row['goal'],row['rep'],row['step']


def update(arm,k,tokens,eos,sources):
    folder = FORMAL/'pilot'/f'{k:02}'/arm
    manifest = read(folder/'target_manifest.json')
    sources[f'{arm}/{k}/target_manifest'] = digest_file(folder/'target_manifest.json')
    saved = {}
    for name in ('student','target','bf16_before'):
        path = folder/(name+'.pt')
        assert digest_file(path)==manifest[name+'_sha256']
        saved[name] = torch.load(path,weights_only=True,map_location='cpu')
    scored = {record_key(s):(s,q,b) for s,q,b in zip(*saved.values(),strict=True)}
    missing = {record_key(r) for r in manifest['na_records']}
    assert len(scored)+len(missing)==manifest['rollout_records'] and not missing.intersection(scored)
    obs,cov = [],Counter()
    for path in tapes(FORMAL,arm,k):
        sources[f'{arm}/{k}/{path.name}'] = digest_file(path)
        for raw in rows(path):
            key = record_key(raw)
            answer = raw['response']
            parts = [tokens[i] for i in answer]
            pos,multi,why = locate(parts)
            sampled_role = after_role(parts[pos[0].index][len(pos[0].prefix):]) if pos else None
            legal = legality(raw)
            if 'legal' in raw:
                assert legal==raw['legal']
            cov.update(['replies',why])
            cov['positions_located'] += len(pos)
            cov['legal_replies'] += legal
            cov['next_command_replies'] += bool(multi)
            if multi:
                cov['next_command_'+multi] += 1
            cov['legal_and_next_command'] += legal and bool(multi)
            if key in missing:
                missing.remove(key)
                cov['structural_na_replies'] += 1
                cov['structural_na_positions'] += len(pos)
                continue
            student,q,before = scored.pop(key)
            assert answer==student['answer']
            mask = student['mask']
            actor = student['snapshot']
            check = before.float().masked_fill(~mask,-torch.inf).softmax(-1)
            assert torch.allclose(actor,check,rtol=0,atol=1e-6)
            assert torch.all(q[~mask]==0) and torch.all(actor[~mask]==0)
            assert torch.allclose(q.sum(1),torch.ones(len(q)),rtol=0,atol=1e-6)
            for p in pos:
                j = p.index
                valid = mask[j]
                ids = student['ids'][j,valid].tolist()
                assert len(ids)==len(set(ids)) and answer[j] in ids
                stop,cont = candidate_classes(p,ids,tokens,eos)
                x,y = q[j,valid].double().numpy(),actor[j,valid].double().numpy()
                obs.append((key[0],legal,multi,p.role,bool(stop),bool(cont),
                            x[stop].sum(),y[stop].sum(),x[cont].sum(),y[cont].sum(),sampled_role))
    assert not scored and not missing and cov['replies']==manifest['rollout_records']
    assert len(obs)+cov['structural_na_positions']==cov['positions_located']
    return obs,cov


def metrics(obs):
    n = len(obs)
    if not n:
        return dict(positions=0)
    ids = [r[0] for r in obs]
    q,p,qc,pc = np.array([r[6:10] for r in obs]).T
    positive = (q>0)&(p>0)
    assert np.all(positive==np.array([r[4] for r in obs]))  # no one-sided FP32 underflow
    result = cluster_mean(ids,np.column_stack((q,p,q<p,qc,pc)))
    result['positions'] = result.pop('n')
    result['log_stop_ratio'] = cluster_mean(np.array(ids)[positive],np.log(q[positive]/p[positive])[:,None])
    result['support_counts'] = dict(stop=int(positive.sum()),continuation=sum(r[5] for r in obs),
                                     both=sum(r[4] and r[5] for r in obs))
    return result


def summarize(obs,overall=False):
    cohorts = dict(all=obs,legal=[r for r in obs if r[1]],next_command=[r for r in obs if r[2]],
                   newline_next_command=[r for r in obs if r[2]=='newline'])
    if overall:
        cohorts['legal_without_next_command'] = [r for r in obs if r[1] and not r[2]]
        cohorts['illegal_next_command'] = [r for r in obs if not r[1] and r[2]]
    result = {}
    for cohort,rr in cohorts.items():
        result[cohort] = {}
        for role in ROLES:
            if role.startswith('bracket_sampled_'):
                suffix_role = role.replace('bracket_sampled_', 'after_')
                selected = [r for r in rr if r[3]=='bracket' and r[10]==suffix_role]
            else:
                selected = [r for r in rr if role=='all' or r[3]==role or (role=='following' and r[3]!='bracket')]
            v = metrics(selected)
            if role.startswith('bracket') or role=='following':
                both = [r for r in selected if r[4] and r[5]]
                v['both_candidates_present'] = metrics(both)
            result[cohort][role] = v
    return result


def coverage_dict(counts):
    keys = ('replies','located','no_command','unclosed_first_command','positions_located','legal_replies',
            'next_command_replies','next_command_newline','next_command_same_line','legal_and_next_command',
            'structural_na_replies','structural_na_positions')
    return {key:counts[key] for key in keys}


def run_arm(arm):
    tokens,eos = vocabulary()
    sources,windows = {},[]
    steps = sorted(int(p.parent.parent.name) for p in (FORMAL/'pilot').glob('*/'+arm+'/optimizer.json'))
    assert steps==list(range(1,steps[-1]+1))
    all_obs,all_cov = [],Counter()
    for first in steps[::5]:
        obs,coverage = [],Counter()
        end = min(first+4,steps[-1])
        for k in range(first,end+1):
            records,counts = update(arm,k,tokens,eos,sources)
            obs.extend(records); coverage.update(counts)
        windows.append(dict(arm=arm,start=first,end=end,coverage=coverage_dict(coverage),cohorts=summarize(obs)))
        all_obs.extend(obs); all_cov.update(coverage)
        print(arm,end,len(obs),flush=True)
    return arm,windows,dict(coverage=coverage_dict(all_cov),cohorts=summarize(all_obs,True)),sources


def main():
    _,eos = vocabulary()
    windows,overall,sources = [],{},{}
    with ProcessPoolExecutor(max_workers=2) as pool:
        for arm,w,stats,inputs in pool.map(run_arm,('R-gap','P-gap')):
            windows.extend(w)
            overall[arm]=stats
            sources.update(inputs)
    save(DEST/'section26_action_ending.json',dict(columns=COLUMNS,seed=SEED,task_bootstrap_draws=DRAWS,
         weights='equal positions',eos_ids=sorted(eos),source_sha256=sources,
         tokenizer_sha256={p:digest_file(MODEL/p) for p in ('vocab.json','tokenizer_config.json','generation_config.json')},
         overall=overall,windows=windows))


if __name__=='__main__':
    main()
