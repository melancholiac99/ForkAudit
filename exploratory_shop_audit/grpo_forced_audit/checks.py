"""Real cross-process interface/restoration checks, never a benchmark claim."""
from copy import deepcopy
import json
from .common import *
from .interface import Engine, REJECTION
from exploratory_shop_audit.runtime import RPC, SYSTEM
from exploratory_shop_audit.env_worker import parse_action_fixed


def environment(run,name):
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
    return RPC([CPU_PYTHON,'-m','exploratory_shop_audit.grpo_forced_audit.worker','--run',run],run/f'{name}.log',env)


def normalized(state):
    state=deepcopy(state)
    # Session IDs are identical between these independent processes.
    return state


def preflight(run):
    setup(run); a=Engine(run); b=environment(run,'preflight_secondary');checks=[]
    try:
        assert a.hashes==b.ready['tables']
        parser=a.parser_audit()
        train=[i for i,g in enumerate(a.server.goals) if a.server.product_item_dict[g['asin']]['tag']=='train']
        # Test only train and never protected final IDs.
        blind=set(read(COMMUNITY/'src/shopping_grpo/resources/blind_final_task_ids.json')['task_ids'])
        train=[i for i in train if i not in blind]
        interesting=[]
        for idx in train:
            product=a.server.product_item_dict[a.server.goals[idx]['asin']]
            values=[v for opts in product['options'].values() for v in opts]
            if any('  ' in v or '[' in v or ']' in v for v in values):interesting.append(idx)
        chosen=list(dict.fromkeys(interesting[:12]+train[:12]))
        for idx in chosen:
            key=f'p0:{idx}';prefix=[]
            state=a.open(key,idx)
            for action in [f'search[{a.server.goals[idx]["instruction_text"]}]']:
                rb=b.call(cmd='open',key=key,goal_index=idx,prefix=prefix)
                assert normalized(rb)==normalized(state)
                ra=a.step(key,action);rb=b.call(cmd='step',key=key,action=action);assert ra==rb
                prefix.append(action);state=ra['state']
            shown=state['public'].get('products',[])
            if not shown: continue
            # First result, one actual specification, invalid feedback, and repeat termination.
            actions=[f'click[{shown[0]["asin"]}]']
            for action in actions:
                ra=a.step(key,action);rb=b.call(cmd='step',key=key,action=action);assert ra==rb;prefix.append(action);state=ra['state']
            product=state['public'].get('product',{})
            options=[v for vals in state['public'].get('available_options',{}).values() for v in vals]
            legal=state['available']['clickables']
            options=[v for v in options if isinstance(v,str) and v.lower() in legal]
            # Engine-native values; public available_options may be structured dicts.
            native=a.server.product_item_dict[shown[0]['asin']]['options']
            options=[v for values in native.values() for v in values if v in legal]
            if options:
                value=next((v for v in options if '  ' in v or '[' in v or ']' in v),options[0])
                footer=json.loads(state['observation'].rsplit('可点击的按钮: ',1)[1]);assert value in footer
                action=f'click[{value}]';assert parse_action_fixed(action)==('click',value)
                ra=a.step(key,action);rb=b.call(cmd='step',key=key,action=action);assert ra==rb;prefix.append(action);state=ra['state']
                assert ra['legal']
            restored=b.call(cmd='open',key=key,goal_index=idx,prefix=prefix);assert state==restored
            score=deepcopy(state['stop_score'])
            if state['stop_available']:
                ra=a.step(key,'click[buy now]');rb=b.call(cmd='step',key=key,action='click[buy now]');assert ra==rb
                assert ra['status']['reward']==score['reward'];assert ra['status']['reward_valid']==score['valid']
                checks.append({'goal_index':idx,'prefix_restored':True,'buy_score_equal':True,'reward':score['reward'],'reward_valid':score['valid']})
            # Separate search/navigation path with exact native buttons.
            state=a.open(key,idx,prefix[:2]);other=shown[1]['asin'] if len(shown)>1 else shown[0]['asin']
            nav=[]
            for action in ['click[< prev]',f'click[{other}]','click[back to search]',f'search[{a.server.goals[idx]["instruction_text"]}]']:
                ra=a.step(key,action);assert ra['legal'] and not ra['status']['done'];nav.append(action)
            state=a.open(key,idx,prefix[:2]);b.call(cmd='open',key=key,goal_index=idx,prefix=prefix[:2])
            for j in range(3):
                ra=a.step(key,'not_an_action');rb=b.call(cmd='step',key=key,action='not_an_action');assert ra==rb
                assert not ra['legal'] and REJECTION in ra['state']['observation']
                assert ra['state']['public']['actions']==state['public']['actions'] or ra['status']['done']
                if ra['status']['done']:break
            a.close(key);b.call(cmd='close',key=key)
        assert len(checks)>=20
        save(run/'preflight_report.json',{'passed':True,'cross_process_checks':checks,'parser':parser,'tables':a.hashes,
            'interface':INTERFACE,'navigation_paths':True,'invalid_feedback_and_terminal_counters':True,'gpu_training_checks':'pending'})
        save(run/'interface_manifest.json',{'old_system':SYSTEM,'new_system':SYSTEM,'system_change':'none: search_products mapping already explicit',
            'old_renderer_sha256':digest_file(COMMUNITY/'src/shopping_grpo/environment/observation.py'),
            'new_interface_sha256':digest_file(HERE/'interface.py'),'changes':['Preserve exact clickable strings in footer','Deterministic rejection feedback; same native transition'],
            'interface':INTERFACE})
    finally:b.close()
