"""Run once between trainers, with timeout and an independent orphan recovery guard."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
H=Path(__file__).resolve().parent;ROOT=H.parent
GUARD=ROOT/'exploratory_E4_extra_scoring_speed/hook.py'
spec=importlib.util.spec_from_file_location('recovery',GUARD);G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)


def save(x):
    p=H/'trial_hook_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
    print(json.dumps(x),flush=True)


def main(parent,child):
    assert os.getpgrp()==os.getpid()
    p,c=G.proc(parent),G.proc(child);assert p and c and c['parent']==parent
    assert b'run_e4b.sh' in Path(f'/proc/{parent}/cmdline').read_bytes()
    assert b'Azero' in Path(f'/proc/{child}/cmdline').read_bytes()
    guard=subprocess.Popen([sys.executable,str(GUARD),'--guard',str(os.getpid()),G.proc(os.getpid())['identity'],str(parent),p['identity']],
                           start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    benchmark_start=None
    try:
        os.kill(parent,signal.SIGSTOP)
        save(dict(stage='waiting_current_training',parent=parent,child=child,guard=guard.pid))
        deadline=time.monotonic()+1200
        while G.alive(child,c['identity']):
            assert time.monotonic()<deadline;time.sleep(3)
        completed=ROOT/'exploratory_E4b/data/adapters/Azero_s0'
        log=json.loads((completed/'train_log.json').read_text());assert log['steps']==120
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        benchmark_start=time.monotonic();save(dict(stage='benchmark_running',parent=parent))
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1')
        with (H/'trial_gpu.log').open('w') as out:
            subprocess.run([sys.executable,'-u',str(H/'trial.py'),'run'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((H/'trial_result.json').read_text());delay=time.monotonic()-benchmark_start
        net=result.get('projected_remaining_training_gross_saving_minutes',0)-delay/60
        useful=result.get('comparison',{}).get('passed',False) and result.get('speedup',0)>=1.25 and net>=30
        result.update(benchmark_delay_seconds=delay,projected_net_saving_minutes=net,large_enough=useful,adopt=False,
                      decision='Eligible for deployment review' if useful else 'Not worth adopting; retain frozen implementation')
        (H/'trial_result.json').write_text(json.dumps(result,indent=2)+'\n')
        save(dict(stage='complete',large_enough=useful,decision=result['decision'],delay_seconds=delay))
    except BaseException as exc:
        save(dict(stage='failed',error=str(exc),adopt=False));traceback.print_exc()
    finally:
        G.resume(parent,p['identity']);print('Formal E4b queue resumed unchanged',flush=True)


if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))
