"""Bounded exclusive timing of checkpoint disabling; preserve resident formal trainer."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run():
    import math
    import statistics
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    import benchmark as B
    control = json.loads((HERE/'numerical_control.json').read_text())
    assert control['summary']['original']['all_bitwise_equal']
    assert control['summary']['no_checkpoint']['all_bitwise_equal']
    torch.cuda.set_per_process_memory_fraction(.5)
    assert torch.cuda.mem_get_info()[0] > 45*1024**3
    model = AutoModelForCausalLM.from_pretrained(B.BASE, local_files_only=True, dtype=torch.bfloat16,
                                                attn_implementation='sdpa').to('cuda')
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, B.ADAPTER, is_trainable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    cases = json.loads((HERE/'inputs.json').read_text())['cases']
    results = []
    for i, job in enumerate(cases):
        refs = B.references(model, job)
        for variant in ['original', 'no_checkpoint']:
            B.trial(model, job, variant, refs, 82000+i, collect_grad=False)
        record = {k:job[k] for k in ['kind', 'arm', 'row', 'total_tokens']}
        record['trials'] = []
        for repeat in range(4):
            for variant in (['original', 'no_checkpoint'] if repeat%2 == 0 else ['no_checkpoint', 'original']):
                actual, _ = B.trial(model, job, variant, refs, 82000+i, collect_grad=False)
                record['trials'].append(dict(variant=variant, repeat=repeat, **actual))
        medians = {v:statistics.median(t['seconds'] for t in record['trials'] if t['variant']==v)
                   for v in ['original', 'no_checkpoint']}
        record['median_seconds'] = medians
        record['speedup'] = medians['original']/medians['no_checkpoint']
        results.append(record)
        print(json.dumps(record), flush=True)
    out = dict(complete=True, scope='Exclusive GPU forward/backward timing; formal trainer resident but stopped; no optimizer or reference preprocessing timed',
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), torch_threads=torch.get_num_threads(),
               per_case_speedups=[r['speedup'] for r in results],
               geometric_mean_speedup=math.exp(statistics.mean(math.log(r['speedup']) for r in results)),
               peak_allocated_GiB=max(t['peak_allocated_bytes'] for r in results for t in r['trials'] if t['variant']=='no_checkpoint')/1024**3,
               cases=results, formal_training_implementation_changed=False)
    B.save('checkpoint_speed.json', out)
    print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2),flush=True)


def supervise(pid):
    import fcntl
    assert os.getpgrp() == os.getpid()
    with (HERE/'checkpoint_speed.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard_path = ROOT/'exploratory_E4_extra_scoring_speed/hook.py'
        spec = importlib.util.spec_from_file_location('recovery',guard_path)
        recovery = importlib.util.module_from_spec(spec); spec.loader.exec_module(recovery)
        target = recovery.proc(pid); assert target and target['state'] not in ['T','t','Z']
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert b'e4_06_train.py' in cmd and cmd[cmd.index(b'--lr')+1] == b'2e-05'
        subprocess.Popen([sys.executable,str(guard_path),'--guard',str(os.getpid()),recovery.proc(os.getpid())['identity'],
                          str(pid),target['identity']],start_new_session=True,stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        start = time.monotonic();started = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        status = dict(training_pid=pid, training_identity=target['identity'], started_utc=started,
                      reason='Short exclusive timing test; optimizer/model/sample state stays in resident process',
                      effect='Formal train_log wall seconds include this interruption; no annotation cost is affected')
        try:
            os.kill(pid,signal.SIGSTOP)
            status['stage']='timing_in_progress'
            (HERE/'training_pause.json').write_text(json.dumps(status,indent=2)+'\n')
            time.sleep(3)  # Allow already submitted formal CUDA kernels to drain.
            assert recovery.proc(pid)['state'] in ['T','t']
            env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='25',OPENBLAS_NUM_THREADS='1')
            with (HERE/'checkpoint_speed.log').open('w') as output:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--run'],cwd=ROOT,
                               stdout=output,stderr=subprocess.STDOUT,env=env,check=True,timeout=120)
            status['stage']='timing_complete'
        except BaseException as exc:
            status.update(stage='timing_failed',error_type=type(exc).__name__,message=str(exc))
            raise
        finally:
            recovery.resume(pid,target['identity'])
            status.update(pause_seconds=time.monotonic()-start,resumed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            (HERE/'training_pause.json').write_text(json.dumps(status,indent=2)+'\n')
            print(json.dumps(status),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--training-pid',type=int)
    args=parser.parse_args()
    if args.run:run()
    else:
        assert args.training_pid
        supervise(args.training_pid)
