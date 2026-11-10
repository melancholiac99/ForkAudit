"""Read-only E4e verification, paired exploratory analysis and audit sampling."""
import argparse,collections,hashlib,json,os,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent
sys.path.insert(0,str(H));import four_arm_eval as R,stage_logic as S,funnel as F
from e4_01_rollout import seed_for
RUN=H/'run_20261110';RESULT=H/'results_20261110'

def load(p):return json.loads(Path(p).read_text())
def save(p,x):R.E.write_json(str(p),x)
def sha(p):return R.E.file_sha256(str(p))
def data_root(stage):return H/'data'/f'{stage}_20261110'/'valid_unseen'

def verify(stage):
    plan=load(RUN/'plan.json');games={g['seed']:g for g in load(R.E.GAMES_PATH)['valid_unseen']}
    seeds=plan['pilot_games'] if stage=='pilot' else plan['full_games'];reps=plan['pilot_reps'] if stage=='pilot' else plan['full_reps']
    tags=plan['pilot_policies'] if stage=='pilot' else list(plan['policies'])
    records={};raw={};events=[];counts={};files=[]
    for tag in tags:
        for cond in plan['conditions']:
            episodes=[];count=collections.Counter()
            expected={f's{s}_r{r}.json' for s in seeds for r in range(reps)}
            dest=data_root(stage)/tag/cond
            assert {p.name for p in dest.glob('s*_r*.json')}==expected,(tag,cond,'missing or extra episodes')
            for seed in seeds:
                g=games[seed]
                for rep in range(reps):
                    path=dest/f's{seed}_r{rep}.json';ep=load(path);prov=ep['provenance']
                    assert ep['seed']==seed and ep['rep']==rep and ep['game_file']==g['game_file'] and ep['task_type']==g['task_type']
                    assert ep['condition']==cond and prov['condition']==cond
                    assert prov['model']==plan['model'] and prov['adapter']==plan['policies'][tag]
                    assert prov['execution_plan_sha256']==sha(RUN/'plan.json')
                    assert prov['source_sha256']==plan['source'][str(H/'four_arm_eval.py')]
                    assert prov['games_sha256']==sha(R.E.GAMES_PATH)
                    assert ep['task']==R.common.extract_task(ep['initial_obs'])
                    assert ep['n_turns']==len(ep['steps'])<=R.common.MAX_STEPS
                    carry=S.CarryState();carry.update(ep['initial_obs']);obs=ep['initial_obs']
                    for i,step in enumerate(ep['steps'],1):
                        assert step['turn']==i and step['sample_seed']==seed_for(seed,i,rep)
                        assert step['obs_before']==obs
                        assert step['held_known_before']==carry.known and step['held_before']==(carry.item if carry.known else None)
                        model,valid,fmt=R.common.parse(step['response'])
                        assert model==step['model_action'] and valid==step['model_valid'] and fmt==step['fmt']
                        forced=S.forced_release_action(carry,ep['task'],ep['task_type'],step['admissible']) if cond in ('hard','both') else None
                        assert step['rule_triggered']==(forced is not None)
                        assert step['overridden']==(forced is not None and forced!=model)
                        assert step['action']==(forced or model) and step['valid']==(True if forced is not None else valid)
                        if forced:
                            assert forced in step['admissible']
                            count['triggered']+=1;count['overridden']+=int(step['overridden'])
                            contradiction=not carry.known or not carry.item or S.is_target(carry.item,S.target_word(ep['task']))
                            count['observable_unnecessary_release']+=int(contradiction)
                            assert not contradiction
                            context=[ep['initial_obs']]+[s['obs_after'] for s in ep['steps'][:i-1]]
                            events.append(dict(path=str(path.relative_to(ROOT)),policy=tag,condition=cond,turn=i,
                                task=ep['task'],task_type=ep['task_type'],held=carry.item,prior_feedback=context[-8:],
                                carry_feedback=[o for o in context if S.PICK.search(o) or S.RELEASE.search(o) or S.HELD_TEXT.search(o) or 'hands are empty' in o.lower() or 'not carrying anything' in o.lower()],
                                admissible=step['admissible'],model_action=model,executed_action=step['action'],feedback=step['obs_after'],
                                overridden=step['overridden']))
                        count['steps']+=1;count['known_steps']+=int(carry.known)
                        count['known_non_target_steps']+=int(carry.known and bool(carry.item) and not S.is_target(carry.item,S.target_word(ep['task'])))
                        carry.update(step['obs_after'])
                        assert step['held_known_after']==carry.known and step['held_after']==(carry.item if carry.known else None)
                        if forced:count['empty_after_trigger']+=int(carry.known and carry.item is None)
                        if i<len(ep['steps']):assert not step['won_after']
                        obs=step['obs_after']
                    assert bool(ep['won'])==bool(ep['steps'] and ep['steps'][-1]['won_after'])
                    count['episodes']+=1;count['episodes_with_override']+=int(any(s['overridden'] for s in ep['steps']))
                    episodes.append(ep);raw[str(path.relative_to(ROOT))]=sha(path);files.append(path)
            records[tag,cond]=episodes
            counts[tag+'/'+cond]=dict(count,observable_unnecessary_release_rate=count['observable_unnecessary_release']/count['triggered'] if count['triggered'] else None)
    expected_n=len(seeds)*reps*len(tags)*4;assert len(raw)==expected_n
    stats={}
    for tag in tags:
        stats[tag]={c:dict(success=sum(e['won'] for e in records[tag,c])/len(records[tag,c]),**F.funnel_for(records[tag,c])) for c in plan['conditions']}
        stats[tag]['paired']={f'{b}-{a}':F.paired_ci(records[tag,a],records[tag,b]) for a,b in [('base','state'),('base','hard'),('base','both'),('hard','both'),('state','both')]}
    if stage=='full':
        pooled={c:[e for t in tags if t.startswith('aprime') for e in records[t,c]] for c in plan['conditions']}
        stats['aprime_pooled']={c:dict(success=sum(e['won'] for e in pooled[c])/len(pooled[c]),**F.funnel_for(pooled[c])) for c in plan['conditions']}
        stats['aprime_pooled']['paired']={f'{b}-{a}':F.paired_ci(pooled[a],pooled[b]) for a,b in [('base','state'),('base','hard'),('base','both'),('hard','both'),('state','both')]}
    report=dict(stage=stage,exploratory=True,episodes=len(raw),games=len(seeds),policies=stats,intervention_audit=counts,
        limitations=['Task milestones are observed evidence, not causal recovery budgets.','Observable unnecessary-release flags do not measure whether a legal destination is strategically optimal.',
        'Paired game-cluster bootstrap uses original E4e 2000 resamples and seed 20261110; multiple exploratory contrasts, not preregistered confirmations.',plan['missing_optional_public_rl']])
    save(RESULT/f'{stage}_analysis.json',report);save(RESULT/f'{stage}_raw_manifest.json',raw)
    save(RESULT/f'{stage}_verification.json',dict(passed=True,episodes=len(raw),analysis_sha256=sha(RESULT/f'{stage}_analysis.json'),raw_manifest_sha256=sha(RESULT/f'{stage}_raw_manifest.json'),checks=['fixed cohort/repeats','weights routing and frozen source/plan','sampling seeds','complete feedback chain','all carry states and legal rule applications','raw response versus model/executed actions']))
    samples=[]
    for tag in tags:
        for cond in ['hard','both']:
            subset=[e for e in events if e['policy']==tag and e['condition']==cond]
            subset.sort(key=lambda e:hashlib.sha256((e['path']+':'+str(e['turn'])).encode()).hexdigest())
            for e in subset[:4]:samples.append(dict(id=f'review_{len(samples):03d}',**e))
    save(RESULT/f'{stage}_override_review_request.json',dict(stage=stage,samples=samples,total_triggered=len(events),
        labels=['rule_supported','rule_contradicted','uncertain'],instructions='Read agent-visible previous feedback, held object, task target, admissible command and returned feedback. Label whether the release rule is supported by this evidence. Do not infer causal benefit or optimality from successful release. Outcome hidden in sample payload.'))
    if stage=='full':
        lines=['E4e exploratory four-arm recovery audit',f'{len(raw)} episodes / {len(seeds)} fixed games / 3 rollout repeats / 4 policies','']
        for tag,s in stats.items():
            lines.append(tag+': '+', '.join(f"{c}={100*s[c]['success']:.2f}%" for c in plan['conditions']))
            for k,v in s['paired'].items():lines.append(f"  {k}: {v['difference_pp']:+.4f} pp, 95% game-cluster CI {v['ci95_pp']}")
        lines.extend(['',*report['limitations']]);(RESULT/'E4e_REPORT.txt').write_text('\n'.join(lines)+'\n')
    return files

def replay_audit(stage,files):
    # All pilot episodes; one predeclared hash-selected full episode per model/condition.
    if stage=='pilot':chosen=files
    else:
        groups=collections.defaultdict(list)
        for p in files:groups[str(p.parent)].append(p)
        chosen=[min(ps,key=lambda p:hashlib.sha256(('E4e-replay:'+p.name).encode()).hexdigest()) for _,ps in sorted(groups.items())]
    from envpool import EnvPool
    import alfenv
    pool=EnvPool(8);eps={str(i):load(p) for i,p in enumerate(chosen)};checks=[]
    try:
        opened=pool.open([(k,os.path.join(alfenv.ALFWORLD_DATA,x['game_file']),[],None) for k,x in eps.items()])
        for k,x in eps.items():assert opened[k][0]==x['initial_obs'] and not opened[k][3]
        current={k:v[:2] for k,v in opened.items()}
        for t in range(50):
            active={k:x for k,x in eps.items() if len(x['steps'])>t}
            if not active:break
            for k,x in active.items():
                assert current[k][0]==x['steps'][t]['obs_before']
                assert current[k][1]==x['steps'][t]['admissible']
            response=pool.step({k:x['steps'][t]['action'] for k,x in active.items()})
            for k,x in active.items():
                obs,adm,won,done=response[k];assert obs==x['steps'][t]['obs_after'] and bool(won)==x['steps'][t]['won_after']
                current[k]=(obs,adm)
        pool.close(list(eps));checks=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in chosen]
    finally:pool.shutdown()
    save(RESULT/f'{stage}_environment_replay.json',dict(passed=True,episodes=len(chosen),files=checks))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','full']);args=ap.parse_args()
    RESULT.mkdir(exist_ok=True);files=verify(args.stage);replay_audit(args.stage,files)
    print('Verification and real environment replay passed',args.stage,len(files),flush=True)
