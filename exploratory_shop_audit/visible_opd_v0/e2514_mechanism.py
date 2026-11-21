"""CPU-only first-search diagnostics using original A.5 tokens and held-out B predictions."""
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer

from .b_collect import PARENT, RUN as STAGE_B
from .e25_paths import PUBLIC
from .parser import categories
from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.grpo_forced_audit.common import MODEL, digest_file, read, read_rows, save

OUT = PUBLIC/'migration_2514'


def teacher(tokenizer):
    goals = read(PARENT/'snapshots/goals_frozen.json')
    episodes, searches = set(), {}
    inputs = {}
    for folder in [*sorted(PARENT.glob('smoke_batch_*')), *sorted(PARENT.glob('formal_batch_*'))]:
        for name in ('steps','episodes'):
            path = folder/(name+'.jsonl')
            inputs[str(path.relative_to(PARENT))] = digest_file(path)
        for row in read_rows(folder/'episodes.jsonl'):
            if row['mode'] in ('full','placeholder','none'):
                key = row['mode'],row['goal'],row['rep']
                assert key not in episodes
                episodes.add(key)
        for row in read_rows(folder/'steps.jsonl'):
            key = row['mode'],row['goal'],row['rep']
            name,arg = parse_action_fixed(row['action'])
            if key not in episodes or key in searches or not row['legal'] or name!='search':continue
            labels = categories(tokenizer,row['response_ids'],{'page_type':row['page']})
            searches[key] = (labels.count('search'),str(goals[key[1]]['asin']).casefold() in arg.casefold())
    stats = []
    for mode in ('full','placeholder','none'):
        for decode in ('sample','greedy'):
            keys = sorted(k for k in episodes if k[0]==mode and (k[2]==2)==(decode=='greedy'))
            assert len(keys)==(200 if decode=='sample' else 100)
            observations = [searches[k] for k in keys if k in searches]
            counts = Counter(n for n,_ in observations)
            hits = sum(hit for _,hit in observations)
            stats.append(dict(mode=mode,decode=decode,episodes=len(keys),with_first_search=len(observations),
                no_legal_search=len(keys)-len(observations),token_histogram=counts,
                mean_tokens=float(np.mean([n for n,_ in observations])) if observations else None,
                gold_in_query=hits,proportion=hits/len(observations) if observations else None))
    save(OUT/'a5_first_query.json',dict(results=stats,inputs=inputs))


def bracket(tokenizer):
    folds=[np.load(STAGE_B/f'fold_{i}.npz') for i in range(4)]
    n=sum(len(f['idx']) for f in folds)
    pred=np.full((n,9),np.nan,dtype=np.float32)
    fold_ids=np.full(n,-1)
    for i,f in enumerate(folds):
        assert np.all(fold_ids[f['idx']]==-1)
        pred[f['idx']]=f['content'];fold_ids[f['idx']]=i
    assert np.isfinite(pred).all()
    bracket_ids={i for tok,i in tokenizer.get_vocab().items() if tok==']'}
    assert len(bracket_ids)==1 and tokenizer.decode(list(bracket_ids))==']'
    bracket_id=next(iter(bracket_ids))
    split=read(PARENT/'asin_split_manifest.json')
    expected={r['goal_index']:split['cal_folds'][r['asin']] for r in split['T_cal']}
    totals=Counter(); values=defaultdict(list); offset=0;tasks=set()
    for path in sorted((STAGE_B/'features').glob('*.pt')):
        data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        goal=data['goal'];tasks.add(goal)
        assert np.all(fold_ids[offset:offset+len(data['classes'])]==expected[goal])
        index=0
        for j,category in enumerate(data['classes']):
            if category!='search':continue
            index+=1;totals[index]+=1
            at=torch.where(data['mask'][j] & (data['ids'][j]==bracket_id))[0]
            assert len(at)<=1
            if len(at):
                col=int(at[0]);values[index].append((goal,float(data['d'][j,col]),float(pred[offset+j,col])))
        offset+=len(data['classes'])
    assert offset==n and tasks==set(expected)
    def stats(rows):
        x=np.array([r[1:] for r in rows])
        return dict(tasks=len(set(r[0] for r in rows)),n=len(rows),mean=x.mean(0).tolist() if len(x) else None,
                    median=np.median(x,axis=0).tolist() if len(x) else None,
                    positive_share=(x>0).mean(0).tolist() if len(x) else None)
    save(OUT/'tcal_closing_bracket.json',dict(candidate=']',token_id=bracket_id,
         columns=['d_content','unshrunk_oof_prediction'],tasks=len(tasks),all_positions=n,
         selected_position='1-based ordinal among actual search argument tokens, exact closing-bracket candidate only',
         fold_sha256=[digest_file(STAGE_B/f'fold_{i}.npz') for i in range(4)],
         overall=stats(sum(values.values(),[])),positions=[dict(position=i,all_search_positions=totals[i],**stats(values[i])) for i in sorted(totals)]))


if __name__=='__main__':
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    teacher(tok)
    bracket(tok)
