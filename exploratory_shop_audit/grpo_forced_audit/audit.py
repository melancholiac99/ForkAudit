"""Forced single-step arms with independent A/B continuation banks."""
from collections import defaultdict
import argparse
import random
import numpy as np
from .common import *
from .rollout import Runner,jobs
from .analysis import interval
from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.audit_statistics import reliability_components


def semantics(action):
    name,arg=parse_action_fixed(action)
    return name,arg.lower() if isinstance(arg,str) and name=='click' else arg


def candidates(state,original,seed):
    public=state['public'];available=state['available']['clickables']; actions=[original];coverage={}
    def add(action):
        if semantics(action) not in {semantics(a) for a in actions}:actions.append(action)
    if public['page_type']=='search_results':
        for product in public.get('products',[])[:5]:add(f'click[{product["asin"]}]')
    else:
        if state['stop_available']:add('click[buy now]')
        axes={a:[v for v in values if str(v).lower() in available] for a,values in public.get('available_options',{}).items()}
        rng=random.Random(seed);order=sorted(axes);rng.shuffle(order)
        for axis in order:rng.shuffle(axes[axis])
        depth=0
        while len(actions)<6 and any(depth<len(v) for v in axes.values()):
            for axis in order:
                if depth<len(axes[axis]):add(f'click[{axes[axis][depth]}]')
                if len(actions)>=6:break
            depth+=1
        coverage={'all_axes_and_values':public.get('available_options',{}),'selected_before':public.get('selected_options',{}),
                  'included_values':[semantics(a)[1] for a in actions if semantics(a)[0]=='click']}
    return actions[:6],coverage


def run_audit(run,checkpoint,label,count):
    split=read(run/'splits.json');tasks=split['E0'][:count] if label=='base' else split['E1'][:count]
    folder=run/('audit_base' if label=='base' else 'audit_post');folder.mkdir(exist_ok=True)
    runner=Runner(run,checkpoint)
    try:
        runner.trajectories(folder/'natural',jobs(tasks,'audit_'+label,label))
        raw=read_rows(folder/'natural/steps.jsonl');by_id=defaultdict(list)
        for row in raw:by_id[row['trajectory_id']].append(row)
        existing={r['state_id']:r for r in read_rows(folder/'states.jsonl')}
        points=[]
        for episode,rows in by_id.items():
            rows.sort(key=lambda r:r['step'])
            for page in ['search_results','product_detail']:
                step=next((s for s in rows if s['before']['public']['page_type']==page),None)
                if step is None:continue
                sid=episode+':'+page
                if sid in existing:points.append(existing[sid]);continue
                previous=[s for s in rows if s['step']<step['step']]
                point={'state_id':sid,'task_id':step['task_id'],'goal_index':step['goal_index'],'page':page,
                    'prefix':[s['action'] for s in previous],'history':[(s['before']['observation'],s['generation']['text']) for s in previous],
                    'state':step['before'],'input':step['input'],'original_action':step['action'],'original_legal':step['legal'],
                    'checkpoint_sha':runner.checkpoint_hash,'interface_version':INTERFACE}
                point['actions'],point['coverage']=candidates(step['before'],step['action'],stable_seed(SEED,sid,'arms'))
                append(folder/'states.jsonl',point);points.append(point)
        previous_arms={(r['state_id'],r['candidate']):r for r in read_rows(folder/'candidates.jsonl')}
        for point in points:
            for cid,action in enumerate(point['actions']):
                key=(point['state_id'],cid)
                if key in previous_arms:continue
                restored=runner.engine.call(cmd='open',key='arm',goal_index=point['goal_index'],prefix=point['prefix'])
                from exploratory_shop_audit.run_pilot import public_state
                assert public_state(restored)==point['state'],'prefix restore changed observation/state'
                result=runner.engine.call(cmd='step',key='arm',action=action);after=result['state'];upper=None
                # Privileged completion uses only the selected arm's product; fixing
                # a spec arm preserves that axis, other axes may be completed.
                if after['public'].get('product') and not result['status']['done']:
                    fixed=None
                    if point['page']=='product_detail':
                        _,value=semantics(action)
                        fixed={k:v for k,v in after['public'].get('selected_options',{}).items() if v.lower()==value}
                    upper=runner.engine.call(cmd='maximum',goal_index=point['goal_index'],asin=after['public']['product']['asin'],fixed=fixed)
                    if upper['options'] is not None:
                        # Verify the controller is actually reachable within native
                        # counters/remaining steps rather than just static scoring.
                        for axis,value in upper['options'].items():
                            if after['public'].get('selected_options',{}).get(axis)==value:continue
                            if result['status']['done']:break
                            result=runner.engine.call(cmd='step',key='arm',action=f'click[{value}]');after=result['state']
                        if not result['status']['done']:result=runner.engine.call(cmd='step',key='arm',action='click[buy now]')
                        upper['controller_status']=result['status']
                        upper['controller_reward']=result['status']['reward'] if result['status'].get('purchase') and result['status'].get('reward_valid') else None
                # Restore and execute once again to save uncontaminated arm feedback.
                runner.engine.call(cmd='open',key='arm',goal_index=point['goal_index'],prefix=point['prefix'])
                one=runner.engine.call(cmd='step',key='arm',action=action)
                row={'state_id':point['state_id'],'task_id':point['task_id'],'candidate':cid,'action':action,
                    'terminal':one['status']['done'],'one_step':one,'privileged_completion':upper,
                    'no_state_change':one['state']['public']==point['state']['public'],'original_legal':point['original_legal']}
                append(folder/'candidates.jsonl',row);previous_arms[key]=row
                runner.engine.call(cmd='close',key='arm')
        for bank in ['A','B']:
            continuation_jobs=[]
            for point in points:
                for cid,action in enumerate(point['actions']):
                    arm=previous_arms[(point['state_id'],cid)]
                    if arm['terminal']:continue
                    for rep in range(3):
                        continuation_jobs.append({'task_id':point['task_id'],'goal_index':point['goal_index'],'split':'audit_'+label,
                            'trajectory_id':f'{point["state_id"]}:arm{cid}:{bank}:{rep}', 'state_id_origin':point['state_id'],
                            'candidate':cid,'bank':bank,'repeat':rep,'prefix':point['prefix']+[action],
                            'history':point['history']+[(point['state']['observation'],'Action: '+action)]})
            runner.trajectories(folder/f'bank_{bank.lower()}',continuation_jobs)
    finally:runner.close()
    summarize(run,folder)


def reliability(rows,name):
    # Reuse the already reviewed estimator and bootstrap, including raw negative
    # signal estimates and bootstrap nonpositive-S frequency.
    from exploratory_shop_audit.audit_statistics import bootstrap_reliability
    result=bootstrap_reliability(rows,stable_seed(SEED,name))
    result['rho_raw']=result.get('rho')
    result['rho1']=result['rho']['1'] if result.get('identifiable') else None
    result['rho3']=result['rho']['3'] if result.get('identifiable') else None
    result['source']='exploratory_shop_audit/audit_statistics.py'
    return result


def summarize(run,folder):
    points=read_rows(folder/'states.jsonl');arms=read_rows(folder/'candidates.jsonl');samples=defaultdict(list)
    for bank in ['a','b']:
        for r in read_rows(folder/f'bank_{bank}/episodes.jsonl'):
            # Identity is encoded and also recoverable from the job manifest.
            prefix,arm,library,rep=r['trajectory_id'].rsplit(':',3)
            samples[(prefix,int(arm[3:]),library)].append(r)
    outcomes=[];components=defaultdict(list);candidate_records=[];full_components=defaultdict(list)
    for point in points:
        values={};valid=True;nonterminal=[];full=[]
        for cid in range(len(point['actions'])):
            arm=next(r for r in arms if r['state_id']==point['state_id'] and r['candidate']==cid)
            if arm['terminal']:
                values[cid]=[arm['one_step']['status']['reward']]*2;valid &= arm['one_step']['status'].get('reward_valid') is True
                full.append([values[cid][0]]*6)
                invalid_fraction=float(arm['one_step']['status'].get('reward_valid') is not True);length=0;sample_count=1
            else:
                a=samples[(point['state_id'],cid,'A')];b=samples[(point['state_id'],cid,'B')];assert len(a)==len(b)==3
                values[cid]=[float(np.mean([r['reward'] for r in x])) for x in [a,b]]
                nonterminal.append([r['reward'] for r in a+b]);full.append(nonterminal[-1]);valid &= all(r['reward_valid'] is True for r in a+b)
                invalid_fraction=float(np.mean([r['reward_valid'] is not True for r in a+b]));length=float(np.mean([r['continuation_length'] for r in a+b]));sample_count=6
            upper=arm.get('privileged_completion')
            upper_reward=upper.get('controller_reward') if upper else None
            candidate_records.append({'state_id':point['state_id'],'task_id':point['task_id'],'candidate':cid,'means_AB':values[cid],
                'invalid_fraction':invalid_fraction,'actual_independent_executions':sample_count,'continuation_length':length,
                'terminal':arm['terminal'],'controller_reward':upper_reward,'controller_minus_rollout':upper_reward-float(np.mean(values[cid])) if upper_reward is not None else None})
        if len(nonterminal)>=2:components[point['page']].append(reliability_components(nonterminal))
        if len(full)>=2:full_components[point['page']].append(reliability_components(full))
        best_a=max(values,key=lambda i:(values[i][0],-i));best_b=max(values,key=lambda i:(values[i][1],-i))
        outcomes.append({'task_id':point['task_id'],'state_id':point['state_id'],'page':point['page'],
            'A_select_B_evaluate':values[best_a][1]-values[0][1],'B_select_A_evaluate':values[best_b][0]-values[0][0],
            'selected_A':best_a,'selected_B':best_b,'all_valid':bool(valid),'original_legal':point['original_legal'],'values':values})
    summary={}
    tasks={t['task_id']:t for k in ['T','V'] for t in read(run/'splits.json')[k]}
    for page in ['search_results','product_detail']:
        rows=[r for r in outcomes if r['page']==page]
        # One point per task per page; preserve selection/evaluation within cluster.
        summary[page]={'gain':interval([r['A_select_B_evaluate'] for r in rows],str(folder)+page,.975),
            'reverse_gain':interval([r['B_select_A_evaluate'] for r in rows],str(folder)+page+'reverse'),
            'all_valid_gain':interval([r['A_select_B_evaluate'] for r in rows if r['all_valid']],str(folder)+page+'valid',.975),
            'reliability':reliability(components[page],str(folder)+page),'including_deterministic_terminal_reliability':reliability(full_components[page],str(folder)+page+'full'),
            'terminal_note':'Exact terminal mean/zero variance used analytically; copied values are not independent observations.',
            'positive_fraction':float(np.mean([r['A_select_B_evaluate']>0 for r in rows])) if rows else None,
            'negative_fraction':float(np.mean([r['A_select_B_evaluate']<0 for r in rows])) if rows else None,'strata':{}}
        for label in ['original_legal','original_illegal','H_spec','H_click','H_search','H_unknown','train','heldout']:
            train={t['task_id'] for t in read(run/'splits.json')['T']}
            selected=[r for r in rows if (r['original_legal'] if label=='original_legal' else not r['original_legal'] if label=='original_illegal' else r['task_id'] in train if label=='train' else r['task_id'] not in train if label=='heldout' else tasks[r['task_id']]['label']==label)]
            summary[page]['strata'][label]=interval([r['A_select_B_evaluate'] for r in selected],str(folder)+page+label)
    save(folder/'metrics.json',{'pages':summary,'state_records':outcomes,'candidate_records':candidate_records,'visibility':{'status':'not_reproduced','reason':'No exact ALFWorld student-visible-share definition located in available source; no proxy substituted.'}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--checkpoint',default='base');p.add_argument('--label',default='base');p.add_argument('--count',type=int,default=200);a=p.parse_args();run_audit(a.run,a.checkpoint,a.label,a.count)

if __name__=='__main__':main()
