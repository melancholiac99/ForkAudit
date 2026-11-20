"""Isolated unprivileged checkpoint evaluation with common seeds and native entropy."""
import os
import sys
import time
from collections import Counter,defaultdict
from .d_data import RUN, SCRATCH, ARMS, PARENT, state_dir, folder
from .parser import categories
from exploratory_shop_audit.runtime import make_input,parse_generation
from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner
from exploratory_shop_audit.grpo_forced_audit.common import read, save, stable_seed, digest_value, digest_file

def event(state,action,legal):
    if not legal:return 'illegal'
    name,arg=parse_action_fixed(action)
    if name!='click':return 'search' if name=='search' else 'other_legal'
    arg=(arg or '').lower()
    if arg in {str(p['asin']).lower() for p in state.get('products',[])}:return 'open_listed_product'
    if arg in {str(v).lower() for vs in state.get('available_options',{}).values() for v in vs}:return 'select_visible_option'
    return 'buy' if arg=='buy now' else 'other_click'

def episodes(runner,jobs):
    active=[];finished=[]
    for split,g,rep in jobs:
        key=f'{split}_{g}_{rep}'
        state=runner.engine.call(cmd='open',key=key,goal_index=g)
        active.append(dict(key=key,split=split,goal=g,rep=rep,h=[],state=state,events=Counter(),
                           entropy_sum=Counter(),tokens=Counter(),actions=[]))
    while active:
        requests=[]
        for e in active:
            state=e['state']
            ids=make_input(runner.tokenizer,state['task'],e['h'],state['observation'])['token_ids']
            requests.append(dict(token_ids=ids,temperature=0 if e['rep']==4 else .7,
                seed=stable_seed(20261120,'D_eval',e['split'],e['goal'],e['rep'],len(e['h']))))
        keep=[]
        for e,gen in zip(active,runner.model.call(requests),strict=True):
            state=e['state'];pub=state['public']
            cats=categories(runner.tokenizer,gen['token_ids'],pub)
            for c,entropy in zip(cats,gen['entropy'],strict=True):
                if c in ('click','option','search'):
                    e['entropy_sum'][c]+=entropy;e['tokens'][c]+=1
            action=parse_generation(gen['text'])
            out=runner.engine.call(cmd='step',key=e['key'],action=action)
            e['events'][event(pub,action,out['legal'])]+=1
            e['actions'].append(action)
            if out['status']['done']:
                st=out['status'];assert st.get('reward_valid') and len(e['actions'])<=35
                finished.append(dict(split=e['split'],goal=e['goal'],rep=e['rep'],length=len(e['actions']),
                    reason=st['termination_reason'],reward=st['reward'],actions_sha=digest_value(e['actions']),
                    events=e['events'],entropy_sum=e['entropy_sum'],tokens=e['tokens']))
                runner.engine.call(cmd='close',key=e['key'])
            else:
                e['h'].append((state['observation'],gen['text']))
                e['state']=out['state'];keep.append(e)
        active=keep
    assert len(finished)==len(jobs)
    return finished

def run(k,rank):
    assert (RUN/'training_complete.json').exists()
    splits=read(PARENT/'asin_split_manifest.json')
    arm='B' if k==0 else ARMS[rank]
    path=RUN/'evaluation'/arm/f'rank{rank}'
    path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():return
    if not (path/'snapshots').exists():(path/'snapshots').symlink_to(PARENT/'snapshots',target_is_directory=True)
    os.environ['SHOP_GPU_IDS']=str(rank)
    os.environ['OPD_ENTROPY_TAPE']=str(SCRATCH/f'eval_entropy_{rank}.bin')
    checkpoint='base' if k==0 else state_dir(10,arm)/'adapter'
    if k:
        assert digest_file(checkpoint/'adapter_model.safetensors')==read(folder(10,arm)/'optimizer.json')['safetensors_sha256']
    jobs=[]
    for name in ('V','T_opd_eval'):
        for t in splits[name]:
            for rep in range(5):jobs.append((name,t['goal_index'],rep))
    chunks=[j for j in range(0,len(jobs),32) if k or (j//32)%4==rank]
    start=time.monotonic()
    runner=Runner(path,checkpoint)
    try:
        for j in chunks:
            dest=path/f'episodes_{j:04}.json'
            if not dest.exists():save(dest,episodes(runner,jobs[j:j+32]))
    finally:runner.close()
    save(path/'complete.json',dict(arm=arm,k=k,rank=rank,count=sum(len(jobs[j:j+32]) for j in chunks),seconds=time.monotonic()-start))

if __name__=='__main__':
    os.umask(0o077)
    run(int(sys.argv[1]),int(os.environ['LOCAL_RANK']))
