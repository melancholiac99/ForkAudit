"""Paired original-vs-replay rates and action diagnostics on §26 evaluation tasks."""
from collections import Counter,defaultdict
import csv
import gzip
import io

import numpy as np

from .acceptance import ALL_REQUIREMENTS
from .first_command_replay import DEST, ROOT, MODELS, POOLS, first_command
from .gap_cpu_actions import TYPES
from .gap_report import contrasts
from .gap_score import RUN, PUBLIC, split_tasks
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save
from exploratory_shop_audit.runtime import parse_generation


def action_counts(events):
    out=Counter()
    saw_cut=False
    for e in events:
        action,cut=first_command(e['text'],parse_generation(e['text']))
        assert (action,cut)==(e['action'],e['truncated'])
        ok=e['legal']
        out.update(['actions'])
        out['legal']+=ok
        if saw_cut:
            out['subsequent']+=1
            out['subsequent_legal']+=ok
        if cut:
            out['cuts']+=1
            out['cut_legal']+=ok
            saw_cut=True
        if not ok:
            label=e['illegal_type']
            assert label in TYPES
            out[label]+=1
            if label=='absent_product_id':
                x=e['detail']
                out[f"absent_gold{int(x['gold'])}_partial{int(x['partial'])}"]+=1
        else:
            assert 'illegal_type' not in e
    out['episodes_with_cut']+=saw_cut
    return out


def ratio(num,den):
    return num/den if den else None


def action_summary(c):
    return dict(actions=c['actions'],legal=c['legal'],legal_rate=ratio(c['legal'],c['actions']),
                truncations=c['cuts'],episodes_with_cut=c['episodes_with_cut'],
                truncated_legal=c['cut_legal'],truncated_legal_rate=ratio(c['cut_legal'],c['cuts']),
                subsequent_actions=c['subsequent'],subsequent_legal=c['subsequent_legal'],
                subsequent_legal_rate=ratio(c['subsequent_legal'],c['subsequent']),
                illegal_types={t:c[t] for t in TYPES},absent_ids={
                    f'gold={g},partial={p}':c[f'absent_gold{g}_partial{p}'] for g in (0,1) for p in (0,1)})


def cell(arm,k,pool,mode,tasks,sources):
    data={}; labels={t['goal_index']:t['asin'] for t in tasks}; excluded=set()
    for rank in range(4):
        suffix=f'{pool}/{mode}/{arm}_{k:02}/rank{rank}'
        old=RUN/'formal/evaluation'/suffix
        new=ROOT/suffix
        m=read(new/'complete.json')
        assert m['archive_sha256']==digest_file(old/'complete.json')
        sources[suffix+'/complete']=digest_file(new/'complete.json')
        assert len(m['parts'])==20
        for name,newhash in m['parts'].items():
            assert digest_file(new/name)==newhash
            sources[suffix+'/'+name]=dict(original=digest_file(old/name),replay=newhash)
            def index(records):
                out={(r['goal'],r['rep']):r for r in records}
                assert len(records)==len(out)
                return out
            orig=index(read(old/name)); replay=index(read(new/name))
            assert set(orig)==set(replay) and not set(data)&set(orig)
            batch_cut=any(e['truncated'] for r in replay.values() for e in r.get('events',[]))
            for key,r in replay.items():
                base=orig[key]; g,rep=key
                assert g in m['goals'] and 0<=rep<=4
                if r.get('no_common_page') or base.get('no_common_page'):
                    assert mode=='page' and r.get('no_common_page')==base.get('no_common_page')==True
                    data[key]=None;excluded.add(g);continue
                ev=r['events']
                anch=RUN/'formal/pages'/pool/f'rank{rank}'/f'{pool}_{g}.json'
                extra=len(read(anch)['actions']) if mode=='page' else 0
                assert len(ev)+extra==r['length'] and r['length']<=35
                before=int(base['reason'] in ALL_REQUIREMENTS)
                after=int(r['reason'] in ALL_REQUIREMENTS)
                counts=action_counts(ev)
                stratum='episode_cut' if counts['cuts'] else 'episode_no_cut'
                for label in [stratum]+([] if batch_cut else ['whole_batch_no_cut']):
                    counts[label]+=1
                    counts[label+'_gain']+=after>before
                    counts[label+'_loss']+=after<before
                data[key]=(before,after,counts)
    assert set(data)=={(g,rep) for g in labels for rep in range(5)}
    result=[]; taskrows=[]
    for decode,reps in [('sample',range(4)),('greedy',[4])]:
        oldp,newp={},{}
        all_counts=Counter(); sampled=0
        for g in sorted(labels):
            if g in excluded:
                assert all(data[g,j] is None for j in range(5))
                continue
            group=[data[g,j] for j in reps]
            a=sum(x[0] for x in group);b=sum(x[1] for x in group)
            c=sum((x[2] for x in group),Counter())
            oldp[g]=a/len(group);newp[g]=b/len(group)
            all_counts.update(c);sampled+=len(group)
            taskrows.append([arm,k,pool,mode,decode,g,labels[g],len(group),a,b,c['actions'],c['legal'],c['cuts'],c['cut_legal'],*[c[t] for t in TYPES]])
        stat=contrasts(newp,oldp,labels)
        assert all_counts['actions']-all_counts['legal']==sum(all_counts[t] for t in TYPES)
        summary=action_summary(all_counts)
        result.append(dict(model=f'{arm}@{k}',dataset=pool,interaction=mode,decode=decode,
                           tasks=stat['tasks'],excluded_no_page=len(excluded),asins=stat['asins'],episodes=sampled,
                           original=float(np.mean(list(oldp.values()))),replay=float(np.mean(list(newp.values()))),
                           paired_difference=stat['difference'],paired_ci95=stat['ci95'],**summary,
                           outcome_audit={s:dict(episodes=all_counts[s],gains=all_counts[s+'_gain'],losses=all_counts[s+'_loss'])
                                          for s in ('episode_cut','episode_no_cut','whole_batch_no_cut')}))
    return result,taskrows


def main():
    out=[];taskrows=[];sources={}
    registration=read(PUBLIC/'endpoint_report.json')
    pools=split_tasks()
    for arm,k in MODELS:
        for pool in POOLS:
            for mode in (('free','page') if k in (0,45,50) else ('free',)):
                stats,rows=cell(arm,k,pool,mode,pools[pool],sources)
                for r in stats:
                    if mode=='free':
                        baseline=registration['behavior'][pool][r['decode']][f'{arm}@{k}']
                    else:
                        baseline=registration['common_pages'][pool][r['decode']]['arms'][arm]
                    assert abs(baseline['rate']-r['original'])<1e-12 and baseline['tasks']==r['tasks']
                out.extend(stats);taskrows.extend(rows)
    DEST.mkdir(parents=True,exist_ok=True)
    f=DEST/'behavior_task_pairs.csv.gz'
    with f.open('wb') as raw,gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as gz,io.TextIOWrapper(gz,newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['arm','checkpoint','dataset','interaction','decode','goal','asin','episodes','original_purchases',
                         'replay_purchases','actions','legal','truncations','truncated_legal',*TYPES])
        writer.writerows(taskrows)
    assert len(out)==32
    setup=[]
    for path in sorted(ROOT.glob('*/*/*/rank*/complete.json')):
        receipt=read(path)
        warm=receipt.get('warm_model',False)
        if warm:
            assert receipt['warm_verified'] and len(receipt['validation_tokens_sha256'])==64
        setup.append(dict(shard=str(path.parent.relative_to(ROOT)),warm=warm,
            environment_workers=receipt.get('environment_workers',1),
            setup_seconds=receipt.get('model_setup_seconds'),total_seconds=receipt.get('total_seconds'),
            validation_tokens_sha256=receipt.get('validation_tokens_sha256')))
    assert len(setup)==64
    save(DEST/'behavior.json',dict(cells=out,sources=sources,model_setup=setup,pairs_sha256=digest_file(f),
        original_sha256=digest_file(PUBLIC/'endpoint_report.json'),bootstrap=dict(cluster='ASIN',paired_by='task',draws=10000,seed=20261123),
        hardware='two A100 80GB; logical ranks 0 and 2 on GPU 0, 1 and 3 on GPU 1, sequential on each',
        seeds=[2201,2202,2203,2204],greedy_seed=2201))


if __name__=='__main__':
    main()
