"""Read saved experiment metrics into TensorBoard without touching the trainer."""
import argparse
from collections import defaultdict
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

from .common import DEFAULT_RUN, read, save
from .budget import occupancy


def rows(path):
    if not path.exists():
        return []
    # A concurrent append may leave the final line incomplete for one poll.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


def metric_points(run):
    """Yield only observed values; absent updates and NA metrics remain absent."""
    for row in rows(run/'training_metrics.jsonl'):
        step=int(row['update'])
        for tag,value in row['metrics'].items():
            yield 'train/'+tag,step,value
        episodes=rows(run/'rollouts'/f'train_{step-1:04d}_sample_episodes.jsonl')
        if episodes:
            grouped=defaultdict(list)
            for ep in episodes:
                grouped[ep['task_id']].append(ep)
            for field in ['reward','strict_success','purchase','length']:
                values=[sum(float(r[field]) for r in group)/len(group) for group in grouped.values()]
                yield 'train/rollout/'+field,step,sum(values)/len(values)
    for row in rows(run/'training_signal.jsonl'):
        step=int(row['update'])
        yield 'signal/same_score_group_fraction',step,row['same_score_groups']/row['tasks']
        yield 'signal/mean_group_reward_std',step,sum(row['group_std'])/len(row['group_std'])
        yield 'signal/episodes',step,row['episodes']
    attempts=defaultdict(int)
    for row in rows(run/'advantage_checks.jsonl'):
        step=int(row['update']);attempts[step]+=1
        # A resumed, uncommitted update may use different native DP padding.
        # Keep the old diagnostic immutable and expose each retry separately.
        prefix='signal/' if attempts[step]==1 else f'signal/attempt_{attempts[step]}/'
        yield prefix+'nonzero_advantage_fraction',step,row['nonzero_fraction']
    for path in sorted((run/'evaluation').glob('checkpoint_*.json')):
        _,step,decode=path.stem.split('_')
        summary=read(path)
        for group in ['easy','hard','H_spec','H_click','H_search','H_unknown']:
            for metric,stats in summary.get(group,{}).get('metrics',{}).items():
                yield f'eval/{decode}/{group}/{metric}',int(step),stats.get('estimate')
        for metric,stats in summary.get('weighted_all',{}).items():
            yield f'eval/{decode}/population_weighted/{metric}',int(step),stats.get('estimate') if stats else None


class Exporter:
    def __init__(self, run, logdir):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        self.run=run
        self.state_path=logdir/'export_state.json'
        logdir.mkdir(parents=True,exist_ok=True)
        self.lock=(logdir/'export.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.state=read(self.state_path) if self.state_path.exists() else {'seen':{},'runtime_step':0,'started_unix':time.time()}
        self.train=EventFileWriter(str(logdir/'training'),flush_secs=5)
        self.runtime=EventFileWriter(str(logdir/'runtime_seconds'),flush_secs=5)

    @staticmethod
    def scalar(writer, tag, step, value):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        writer.add_event(Event(wall_time=time.time(),step=step,
            summary=Summary(value=[Summary.Value(tag=tag,simple_value=float(value))])))

    def poll(self):
        for tag,step,value in metric_points(self.run):
            if value is None or not math.isfinite(float(value)):
                continue
            key=f'{tag}@{step}'
            if key in self.state['seen']:
                if self.state['seen'][key]!=float(value):
                    raise ValueError(f'Saved scalar changed at {key}; inspect original records')
                continue
            self.scalar(self.train,tag,step,value)
            self.state['seen'][key]=float(value)
        tick=max(self.state['runtime_step']+1,int(time.time()-self.state['started_unix']))
        self.state['runtime_step']=tick
        import psutil
        mem=psutil.virtual_memory()
        values={'host/available_memory_gib':mem.available/2**30,
                'host/used_memory_gib':mem.used/2**30,
                'budget/cumulative_gpu_hours':occupancy(self.run),
                'progress/completed_optimizer_updates':len(rows(self.run/'weight_updates.jsonl'))}
        live=read(self.run/'live_status.json') if (self.run/'live_status.json').exists() else {}
        values['progress/trainer_process_alive']=int(psutil.pid_exists(live.get('pid',-1)))
        result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        if result.returncode==0:
            for line in result.stdout.splitlines():
                gpu,util,memory=[part.strip() for part in line.split(',')]
                values[f'gpu{gpu}/utilization_percent']=float(util)
                values[f'gpu{gpu}/memory_gib']=float(memory)/1024
        for tag,value in values.items():
            self.scalar(self.runtime,tag,tick,value)
        self.train.flush();self.runtime.flush()
        save(self.state_path,self.state)

    def close(self):
        self.train.close();self.runtime.close();self.lock.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--logdir',type=Path)
    parser.add_argument('--interval',type=float,default=15)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    exporter=Exporter(args.run,args.logdir or args.run/'tensorboard')
    try:
        while True:
            try:
                exporter.poll()
            except (FileNotFoundError,json.JSONDecodeError):
                # Atomic replacement or session completion may race a poll.
                traceback.print_exc()
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        exporter.close()


if __name__=='__main__':
    main()
