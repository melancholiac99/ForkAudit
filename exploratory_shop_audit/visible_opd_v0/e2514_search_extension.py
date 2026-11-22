"""Complete verified historical query coverage alongside the sealed primary intervention."""
from copy import deepcopy
import fcntl
import os
import sys

from .e25_paths import PUBLIC, RUN
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, digest_value, read, save

OUT = PUBLIC/'migration_2514'
DATA = RUN/'migration_2514/search'
PASSES = ('handoff','final')


def paths(wave):
    if wave not in PASSES:
        raise ValueError(wave)
    public = OUT/'search_coverage'/wave
    tape = DATA/'coverage'/wave
    prior = OUT if wave=='handoff' else OUT/'search_coverage/handoff'
    return public,tape,prior


def handoff_ready():
    return (OUT/'P-frozen/endpoint.json').is_file() and (OUT/'R-matched/behavior/020.json').is_file()


def sources(wave):
    if wave=='handoff':
        assert handoff_ready()
        endpoints = {'P-frozen':read(OUT/'P-frozen/endpoint.json')['step'],'R-matched':20}
    else:
        endpoints = {arm:read(OUT/arm/'endpoint.json')['step'] for arm in ('P-frozen','R-matched','R')}
    return [(arm,k) for arm,end in endpoints.items() for k in range(10,end+1,10)]


def extend_tasks(original, recovered):
    tasks = deepcopy(original)
    by_task = {(t['split'],t['goal']):t for t in tasks}
    seen = set()
    for row in recovered:
        if not row['in_intervention_source'] or row['first_query'] is None:
            continue
        k,pool,g,rep = row['step'],row['split'],row['goal'],row['rep']
        key = k,pool,g,rep
        task = by_task[pool,g]
        assert k in (20,30) and rep in task['good_reps' if k==20 else 'failed_reps']
        assert key not in seen
        seen.add(key)
        missing = task['missing'][str(k)]
        assert missing['unrecovered']>0
        missing['unrecovered']-=1
        if missing['unrecovered']==0:
            del missing['unrecovered']
        queries = task['queries'][str(k)]
        query = row['first_query']
        queries[query] = queries.get(query,0)+1
    for task in tasks:
        for k,field in (('20','good_reps'),('30','failed_reps')):
            assert sum(task['queries'][k].values())+sum(task['missing'][k].values())==len(task[field])
    return tasks


def query_set(tasks):
    return {(t['split'],t['goal'],q) for t in tasks for group in t['queries'].values() for q in group}


def empty_rank(tape,rank):
    for k in (20,30):
        relative=f'{k:03}/rank{rank}/complete.json'
        target=tape/relative
        expected=dict(checkpoint=read(DATA/'swap'/relative)['checkpoint'],jobs=0,job_sha=digest_value([]))
        if target.exists():assert read(target)==expected
        else:save(target,expected)


def seal(wave):
    from .e2514_queries import crosscheck_evaluations
    public,tape,prior = paths(wave)
    tape.mkdir(parents=True,exist_ok=True)
    with (tape/'plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        output = public/'plan.json'
        if not output.is_file():
            checkpoints = sources(wave)
            archive = crosscheck_evaluations(checkpoints,include_pages=True)
            tasks = extend_tasks(read(OUT/'intervention_tasks.json'),archive['rows'])
            before = query_set(read(prior/'intervention_tasks.json'))
            after = query_set(tasks)
            assert before<=after
            save(public/'archived_queries.json',archive)
            save(public/'intervention_tasks.json',tasks)
            save(output,dict(pass_name=wave,registration_github='c1bf0ac03d51dc15b37c4fc43b02853275fb0aad',source_checkpoints=checkpoints,
                primary_tasks_sha256=archive['sealed_plan']['sha256'],prior_tasks_sha256=digest_file(prior/'intervention_tasks.json'),
                tasks_sha256=digest_file(public/'intervention_tasks.json'),archive_sha256=digest_file(public/'archived_queries.json'),
                new_queries=sorted(after-before),distinct_queries=len(after),recovered_source_trials=archive['intervention_source_rows']))
        plan = read(output)
        assert plan['tasks_sha256']==digest_file(public/'intervention_tasks.json')
        assert plan['archive_sha256']==digest_file(public/'archived_queries.json')
        assert plan['prior_tasks_sha256']==digest_file(prior/'intervention_tasks.json')
        if not plan['new_queries']:
            for rank in range(4):empty_rank(tape/'swap',rank)
        return plan


def run_rank(wave):
    from .e2512_routing import owners
    from .e2514_gate import group_b_devices
    assert os.environ['OPD_MIGRATION_GROUP']=='B' and os.environ['CUDA_VISIBLE_DEVICES']==group_b_devices()
    rank = int(os.environ['LOCAL_RANK'])
    plan = seal(wave)
    tape = paths(wave)[1]/'swap'
    jobs = [(pool,g,q,i) for pool,g,q in plan['new_queries'] if owners()[g]==rank for i in range(4)]
    if not jobs:
        empty_rank(tape,rank)
        return
    from .e2514_swap import collect_jobs
    for k in (30,20):
        collect_jobs(k,rank,jobs,tape)


def finished_sampling(wave):
    tape = paths(wave)[1]/'swap'
    return all((tape/f'{k:03}'/f'rank{r}/complete.json').is_file() for k in (20,30) for r in range(4))


def pages(wave):
    from .e2514_pages import page_judgments
    public,_,prior = paths(wave)
    page_judgments(tasks_file=public/'intervention_tasks.json',output=public/'intervention_result_pages.json',
        prior=prior/'intervention_result_pages.json',recovered=read(public/'archived_queries.json')['rows'])


def finish(wave):
    from .e2514_swap_report import report
    public,_,_ = paths(wave)
    assert finished_sampling(wave)
    plan = seal(wave)
    tapes = [DATA/'swap',paths('handoff')[1]/'swap']
    if wave=='final':
        tapes.append(paths('final')[1]/'swap')
    report(public,tapes)
    result = read(public/'search_swap_report.json')
    assert result['fresh_episodes']==8*plan['distinct_queries']
    if wave=='final':
        from .e2514_queries import export_combined_searches
        export_combined_searches(public/'historical_searches',public/'archived_queries.json')
    save(public/'coverage.json',dict(posthoc=True,sealed_primary='../..',source_checkpoints=plan['source_checkpoints'],
        recovered_original_source_trials=plan['recovered_source_trials'],additional_queries_this_pass=len(plan['new_queries']),
        additional_episodes_this_pass=8*len(plan['new_queries']),fresh_episodes_including_prior=result['fresh_episodes'],
        tasks_sha256=result['plan_sha256'],pages_sha256=result['pages_sha256'],outcomes_sha256=digest_file(public/'search_swap_report.json')))


if __name__=='__main__':
    action,wave=sys.argv[1:]
    {'rank':run_rank,'pages':pages,'report':finish,'seal':seal}[action](wave)
