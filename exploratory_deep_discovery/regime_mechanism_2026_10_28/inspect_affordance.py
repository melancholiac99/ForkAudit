"""CPU-only inspection of a specific reversible pickup constraint.

Select from metadata: held object differs from clue object, using source arm for
old Dead and held-object arms for old Fork. This does not run an LLM or establish
that a repaired agent would finish the full task.
"""
from pathlib import Path
import re
from protocol import HERE,ROOT,read,write,post_facts,admissible,digest


def held(facts):
    return [m.group(1) for f in facts
            if (m:=re.match(r'^holds\([^,]+,\s*([a-z]+ \d+)\s*:\s*object\)$',f))]


def main():
    import alfenv
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler,AlfredInfos
    d=read(HERE/'design.json');selected=[];all_arm_metadata=[]
    for p in d['points']:
        pk=read(ROOT/p['peek_path'])
        for i,a in enumerate(pk['peeks']):
            objects=held(post_facts(pk,a))
            all_arm_metadata.append(dict(point_id=p['id'],old_regime=p['old_regime'],arm=f'a{i}',
                                         action=a['action'],held_objects=objects))
            source=a['action']==p['source_action']
            eligible=(p['old_regime']=='Dead' and source) or p['old_regime']=='Fork'
            if eligible and objects and p['clue']['object'] not in objects:
                selected.append((p,f'a{i}',objects[0]))
    records=[];env=None
    try:
        for p,arm,obj in selected:
            game=str(Path(alfenv.ALFWORLD_DATA)/p['game_file'])
            if env is None:
                env=textworld.start(game,request_infos=textworld.EnvInfos(won=True,facts=True,admissible_commands=True),
                                    wrappers=[AlfredDemangler(shuffle=False),AlfredInfos])
            else:
                env.load(game)
            state=env.reset()
            for action,expected in zip(p['prefix'],p['expected_prefix_observations']):
                state,_,_=env.step(action)
                assert state.feedback==expected
            state,_,done=env.step(p['arms'][arm]['action'])
            assert state.feedback==p['arms'][arm]['observation_after'] and not done
            navigation=f"go to {p['clue']['location']}"
            navigation_executed=navigation in state['admissible_commands']
            if navigation_executed:
                state,_,_=env.step(navigation)
            before=admissible(state['admissible_commands'])
            take=f"take {p['clue']['object']} from {p['clue']['location']}"
            drop=f"move {obj} to {p['clue']['location']}"
            row=dict(point_id=p['id'],old_regime=p['old_regime'],arm=arm,
                forced_action=p['arms'][arm]['action'],held=obj,clue=p['clue']['text'],
                navigation=navigation if navigation_executed else None,
                observation_at_target=state.feedback,target_take=take,
                target_take_admissible_before=take in before,drop_action=drop,drop_admissible=drop in before)
            if drop in before:
                state,_,done=env.step(drop)
                row.update(drop_response=state.feedback,target_take_admissible_after=take in state['admissible_commands'],
                           held_after=held({str(f) for f in state['facts']}))
            records.append(row)
            print(row,flush=True)
    finally:
        if env is not None:
            env.close()
    write(HERE/'affordance_checks.json',dict(status='exploratory CPU environment diagnostics',
        design_sha256=d['design_sha256'],selection=__doc__,records=records,all_arm_metadata=all_arm_metadata,
        limitation='A legal release restores the local pickup affordance, not proven full-task recoverability. '
                   'No new LLM continuation or post-repair success measurement.'))


if __name__=='__main__':
    main()
