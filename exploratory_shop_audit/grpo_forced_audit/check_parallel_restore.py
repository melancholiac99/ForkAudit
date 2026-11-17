"""Recheck saved forced-arm prefixes in two independent CPU engines."""
from .common import DEFAULT_RUN, read_rows, save, utc
from .checks import environment
from exploratory_shop_audit.run_pilot import public_state


def main():
    run=DEFAULT_RUN.resolve()
    points=read_rows(run/'audit_base/states.jsonl')
    arms={(r['state_id'],r['candidate']):r for r in read_rows(run/'audit_base/candidates.jsonl')}
    first=environment(run,'route_restore_check_a')
    second=environment(run,'route_restore_check_b')
    records=[]
    try:
        assert first.ready['tables']==second.ready['tables']
        for point in points:
            for cid,action in enumerate(point['actions']):
                saved=arms[(point['state_id'],cid)]
                if saved['terminal']:continue
                key=f'{point["state_id"]}:arm{cid}:A:0'
                request={'cmd':'open','key':key,'goal_index':point['goal_index'],'prefix':point['prefix']+[action]}
                a,b=first.call(**request),second.call(**request)
                assert a==b,'Cross-process prefix state changed'
                assert public_state(a)==public_state(saved['one_step']['state']),'Saved arm observation changed'
                assert a['stop_score']==saved['one_step']['state']['stop_score']
                record={'state_id':point['state_id'],'candidate':cid,'observation_equal':True,'score_equal':True,'buy_checked':False}
                if a['stop_available']:
                    buy={'cmd':'step','key':key,'action':'click[buy now]'}
                    x,y=first.call(**buy),second.call(**buy)
                    assert x==y and x['status']['reward']==a['stop_score']['reward']
                    assert x['status']['reward_valid']==a['stop_score']['valid']
                    record['buy_checked']=True
                for engine in [first,second]:engine.call(cmd='close',key=key)
                records.append(record)
        assert records
        save(run/'route_restore_checks.json',{'time':utc(),'passed':True,'checks':records,'tables':first.ready['tables']})
        print(f'{len(records)} saved arms restored identically across two processes',flush=True)
    finally:
        first.close();second.close()


if __name__=='__main__':main()
