"""Trace diagnostics: navigation, actual pickup, outcome. Never infer intention."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from metrics import bootstrap
from protocol import ARMS,CONDITIONS,read,write
from run import result_key,validate_record
from analyze import csv_write


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run_dir',type=Path)
    args=ap.parse_args();m=read(args.run_dir/'run_manifest.json')
    points=[p for p in m['design']['points'] if p['id'] in m['config']['point_ids']]
    detailed=[];point_rows=[]
    for p in points:
        for arm in ARMS:
            for cond in CONDITIONS:
                cell=[]
                for rep in range(m['config']['reps']):
                    key=result_key(p,arm,cond,rep)
                    r=read(args.run_dir/'results'/f'{key}.json');validate_record(r,m['run_fingerprint'],key)
                    ts=[t for t in r['transitions'] if not t['forced']]
                    obj,loc=p['clue']['object'],p['clue']['location']
                    take=f'take {obj} from '
                    pickup=f'You pick up the {obj} from '
                    row=dict(point_id=p['id'],old_regime=p['old_regime'],arm=arm,condition=cond,rep=rep,
                        next_go_to_clue=int(bool(ts) and ts[0]['action']==f'go to {loc}'),
                        attempt_take_clue3=int(any(t['action'].startswith(take) for t in ts[:3])),
                        actual_take_clue3=int(any(t['action'].startswith(take) and
                            t['observation_after'].startswith(pickup) for t in ts[:3])),
                        actual_take_clue_any=int(any(t['action'].startswith(take) and
                            t['observation_after'].startswith(pickup) for t in ts)),
                        success=int(r['won']),generated_steps=r['generated_steps'])
                    detailed.append(row);cell.append(row)
                names=['next_go_to_clue','attempt_take_clue3','actual_take_clue3','actual_take_clue_any',
                       'success','generated_steps']
                point_rows.append(dict(point_id=p['id'],old_regime=p['old_regime'],arm=arm,condition=cond,
                    target_word_native=p['arms'][arm]['target_word_in_feedback'],
                    **{k:float(np.mean([r[k] for r in cell])) for k in names}))
    # Average all five arms per point first; one independent task per point.
    summary={}
    for group in ['all','Dead','Indifferent','Fork']:
        summary[group]={}
        ids=[p['id'] for p in points if group=='all' or p['old_regime']==group]
        for metric in names:
            lifts=[];native=[];completed=[]
            for pid in ids:
                means={cond:np.mean([r[metric] for r in point_rows if r['point_id']==pid and r['condition']==cond])
                       for cond in CONDITIONS}
                native.append(means['native']);completed.append(means['completed'])
                lifts.append(means['completed']-means['native'])
            summary[group][metric]=dict(native=bootstrap(native),completed=bootstrap(completed),lift=bootstrap(lifts))
    write(args.run_dir/'behavior_summary.json',dict(status='exploratory trace diagnostics',
        definitions='actual_take requires both a take command and the environment pickup response; '
                    'first three generated decisions exclude forced action. These are behavior events, not inferred intent.',
        groups=summary))
    csv_write(args.run_dir/'behavior_records.csv',detailed);csv_write(args.run_dir/'behavior_cells.csv',point_rows)
    print({g:{k:summary[g][k]['lift'] for k in ['actual_take_clue3','actual_take_clue_any','success']} for g in summary})


if __name__=='__main__':
    main()
