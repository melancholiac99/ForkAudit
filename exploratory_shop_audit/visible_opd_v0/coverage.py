"""Freeze class minima from paired T_cal score tapes and evaluate ten OPD updates.

The input JSONL has one row per response: goal, rep, step, page, classes and
an aligned boolean valid list. Each manifest episode is required to contribute
at least one response record; a mask of all false is a measured zero.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .section16 import DECISION_CLASSES, TASKS_PER_UPDATE, ROLLOUTS_PER_TASK, freeze_p2_minima, scoreable, p2_coverage
from exploratory_shop_audit.data import digest_file, write_once

def count_file(path, selected, expected_reps):
    counts = Counter()
    covered = defaultdict(set)
    records = set()
    reps = defaultdict(set)
    if selected not in ('full','product','evidence'):
        raise ValueError('A selected teacher packet is required')
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            goal, rep, step=row['goal'],row['rep'],row['step']
            key=goal,rep,step
            if key in records or not isinstance(step,int) or step<0:
                raise ValueError('Repeated or invalid position')
            records.add(key)
            reps[goal].add(rep)
            for label,valid in zip(row['classes'],row['valid'],strict=True):
                if type(valid) is not bool:
                    raise ValueError('Expected a boolean structural mask')
                if label in DECISION_CLASSES and scoreable(selected,row['page'],valid):
                    counts[label]+=1
                    covered[label].add(goal)
    if not reps or any(v != set(range(expected_reps)) for v in reps.values()):
        raise ValueError('Unexpected rollout count')
    return counts,covered,set(reps)

def freeze(cal_file,run,out):
    split=json.loads((run/'asin_split_manifest.json').read_text())
    selection=run/'formal_report/packet_selection.json'
    mode=json.loads(selection.read_text())['selected']
    counts,_,goals=count_file(cal_file,mode,1)
    if goals!={r['goal_index'] for r in split['T_cal']}:
        raise ValueError('Calibration file does not match the frozen T_cal task pool')
    write_once(out,dict(selected=mode,calibration_sha256=digest_file(cal_file),
                        packet_selection_sha256=digest_file(selection),tasks=len(goals),counts=counts,
                        tasks_per_update=TASKS_PER_UPDATE,rollouts_per_task=ROLLOUTS_PER_TASK,
                        minima=freeze_p2_minima(counts,len(goals))))

def pilot(files,frozen,out):
    if len(files)!=10 or len({p.resolve() for p in files})!=10:
        raise ValueError('Ten distinct per-update score tapes are required')
    definition=json.loads(frozen.read_text())
    counts=Counter()
    covered=defaultdict(set)
    for path in files:
        new,groups,goals=count_file(path,definition['selected'],ROLLOUTS_PER_TASK)
        if len(goals)!=TASKS_PER_UPDATE:
            raise ValueError('Expected 32 tasks × 2 trajectories')
        counts.update(new)
        for label,group in groups.items():
            covered[label].update(group)
    distinct={k:len(covered[k]) for k in DECISION_CLASSES}
    by_class=p2_coverage(definition['minima'],counts,distinct)
    write_once(out,dict(counts=counts,distinct_tasks=distinct,minima=definition['minima'],
                        meets_by_class=by_class,passes=all(by_class.values()),freeze_sha256=digest_file(frozen),
                        updates=[digest_file(f) for f in files]))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    stage=sub.add_parser('freeze')
    for k in ('cal_file','run','out'):stage.add_argument(k,type=Path)
    stage=sub.add_parser('pilot')
    for k in ('frozen','out'):stage.add_argument(k,type=Path)
    stage.add_argument('files',nargs=10,type=Path)
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.cal_file,args.run,args.out)
    else:pilot(args.files,args.frozen,args.out)
