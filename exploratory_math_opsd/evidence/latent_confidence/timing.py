from experiment import *
from datetime import datetime
def main():
    out=RAW/'pilot';m=read(RAW/'manifest.json')
    run={};total=0.
    for label in ('extract',*[f'{s}-{arm}' for arm in ARMS for s in ('fixed','math')]):
        fs=list((out/label).glob('*.npz'))
        expected=len(set(i for i,_,_ in m['train']) & set(m['pilot_extract'])) if label=='extract' else 33
        assert len(fs)==expected,(label,len(fs),expected)
        timings=[]
        for f in fs:
            with np.load(f) as a:
                if 'elapsed' in a:timings.append(float(a['elapsed']))
        if not timings and label=='extract':
            # First handoff retained already scored questions. Time the same extraction again if needed.
            raise RuntimeError('missing pilot extraction timings')
        target=len(set(i for i,_,_ in m['train'])) if label=='extract' else (750 if label.startswith('fixed') else 500)
        sec=float(np.mean(timings))*target;total+=sec
        run[label]=dict(pilot_questions=len(fs),timed_questions=len(timings),measured_seconds=sum(timings),estimated_full_gpu_seconds=sec)
    cpu=read(RAW/'cpu_timing.json') if (RAW/'cpu_timing.json').exists() else {}
    save(HERE/'timing.json',dict(unique_pilot_questions=100,cpu_benchmark=cpu,estimated_cpu_seconds=sum(x['estimated_seconds'] for x in cpu.values()) if cpu else None,work=run,full_gpu_hours=total/3600,four_gpu_hours_wall_ideal=total/14400,
         note='Active question processing only. Hardware contention, loading, layer selection, CPU bootstrap and queue waits are additional.'))
    print((HERE/'timing.json').read_text(),flush=True)
if __name__=='__main__': main()
