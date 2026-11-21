"""Balance unchanged single-record target computations over four GPUs and verify the merge."""
from collections import Counter, defaultdict

import torch

from .d_data import SMOKE
from .d_targets import build
from .e25_layout import Layout
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, read_rows, save

FILES = {'student':'student_sha256','target':'target_sha256','bf16_before':'bf16_before_sha256',
         'candidate_mass':'mass_sha256'}


def assignment(work, slots=4):
    """Keep each task and both replicates together; split contiguous arm ranges by estimated tokens."""
    assigned=[[] for _ in range(slots)]
    total=sum(weight for job,goal,weight in work)
    cumulative=0
    for job,goal,weight in work:
        rank=min(slots-1,int((cumulative+weight/2)*slots/total))
        assigned[rank].append((job,goal))
        cumulative+=weight
    return assigned


def plan(jobs):
    work=[]
    for job in jobs:
        name,k,a=job
        layout=Layout(name)
        path=layout.folder(k,a)
        if (path/'target_manifest.json').exists():
            continue
        source=SMOKE if k==1 else path
        loads=Counter()
        for rep in range(2):
            for row in read_rows(source/f'rollout_{rep}.jsonl'):
                size=sum(len(row['views'][v])+len(row['response']) for v in ('none','full','placeholder'))
                loads[row['goal']]+=max(1,size)
        goals={t['goal_index'] for t in read(path.parent/'tasks.json')}
        assert len(goals)==32 and set(loads)==goals
        work.extend((job,g,loads[g]) for g in sorted(goals))
    return assignment(work) if work else [[] for _ in range(4)]


def score_shard(rank,shards):
    grouped=defaultdict(set)
    for job,goal in shards[rank]:
        grouped[tuple(job)].add(goal)
    for (name,k,arm),goals in grouped.items():
        l=Layout(name)
        dst=l.folder(k,arm)/'target_parts'/str(rank)
        build(k,arm,layout=l,goals=goals,output=dst)
        assert set(read(dst/'target_manifest.json')['goals'])==goals


def key(row):
    return row['goal'],row['rep'],row['step']


def equal(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
    if isinstance(left,list):
        return len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left==right


def merge(jobs,shards):
    for name,k,arm in jobs:
        l=Layout(name);dst=l.folder(k,arm)
        if (dst/'target_manifest.json').exists():
            continue
        data={name:[] for name in FILES}
        metas=[];goals=set()
        for rank,chunk in enumerate(shards):
            expected={g for j,g in chunk if list(j)==[name,k,arm]}
            if not expected:
                continue
            part=dst/'target_parts'/str(rank);m=read(part/'target_manifest.json')
            assert set(m['goals'])==expected and not goals.intersection(expected)
            goals.update(expected);metas.append(m)
            for stem,field in FILES.items():
                f=part/f'{stem}.pt';assert digest_file(f)==m[field]
                data[stem].extend(torch.load(f,weights_only=True,map_location='cpu'))
        assert len(goals)==32 and goals=={t['goal_index'] for t in read(dst.parent/'tasks.json')}
        order=sorted(range(len(data['student'])),key=lambda i:key(data['student'][i]))
        valid=[key(data['student'][i]) for i in order]
        na=sorted((r for m in metas for r in m['na_records']),key=key)
        keys=valid+list(map(key,na))
        assert len(keys)==len(set(keys))==sum(m['rollout_records'] for m in metas)
        checks={}
        for field in ('scorer_sha','projection_age','projector_fit_sha'):
            checks[field]=metas[0][field]
            assert all(m[field]==checks[field] for m in metas)
        exact={}
        for stem,field in FILES.items():
            assert len(data[stem])==len(order)
            sorted_values=[data[stem][i] for i in order]
            if k==1 and arm=='R-matched' and name!='formal_2512':
                reference=Layout('formal').folder(1,'R')
                ref_file=reference/f'{stem}.pt'
                assert digest_file(ref_file)==read(reference/'target_manifest.json')[field]
                before=torch.load(ref_file,weights_only=True,map_location='cpu')
                assert equal(before,sorted_values),(name,k,arm,stem)
                exact[stem]=len(sorted_values)
            tmp=dst/f'{stem}.pending'
            torch.save(sorted_values,tmp);tmp.replace(dst/f'{stem}.pt')
            checks[field]=digest_file(dst/f'{stem}.pt')
        save(dst/'target_manifest.json',dict(**checks,goals=sorted(goals),na_records=na,
             rollout_records=len(keys),identical_records_reused=sum(m['identical_records_reused'] for m in metas),
             duplicate_verified=any(m['duplicate_verified'] for m in metas),section2511_shards=len(metas),
             exact_reference_comparison=exact))
