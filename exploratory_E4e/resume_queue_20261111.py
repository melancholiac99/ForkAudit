"""Registered same-model GPU replacement, strict historical pilot gate, then resume."""
import argparse,fcntl,json,os,sys,time,traceback
from pathlib import Path
import run_with_gigpo as W
C=W.C;V=W.V;H=W.H;D=H/'resume_20261111'
old_env=C.env;old_invoke=C.invoke;old_publish=C.publish;old_notify=C.notify

def env():
    p=V.load(D/'registration.json');return dict(old_env(),CUDA_VISIBLE_DEVICES=p['new_gpu_uuid'])
C.env=env

def notify(message):
    message=message.replace('恢复入口 exploratory_E4e/run_with_gigpo.py','恢复入口 exploratory_E4e/resume_queue_20261111.py')
    old_notify(message)
C.notify=notify

def invoke(key,args,cpu=False):
    args=list(args)
    if str(H/'server_eval.py') in args:args[args.index(str(H/'server_eval.py'))]=str(H/'resume_eval_20261111.py')
    return old_invoke(key,args,cpu=cpu)
C.invoke=invoke

def audit_preservation():
    reg=V.load(D/'registration.json')
    for path,sha in V.load(D/'preserved_manifest.json').items():assert V.sha(path)==sha,path
    expected=dict(registration_sha256=V.sha(D/'registration.json'),gpu_inventory=reg['new_gpu_inventory'],runner_sha256=V.sha(H/'resume_eval_20261111.py'),original_scientific_source_sha256=V.sha(H/'four_arm_eval.py'))
    old=V.load(D/'preserved_manifest.json');new={}
    for group in [H/'data/full_20261110',H/'data/rollout',D/'pilot']:
        for p in group.rglob('s*_r*.json'):
            if str(p) in old:continue
            e=V.load(p);assert e['hardware_resume']==expected,str(p)
            new[str(p)]=V.sha(p)
    V.save(D/'preservation_verification.json',dict(passed=True,old_records=len(old),new_records=len(new),new_manifest=new,registration_sha256=V.sha(D/'registration.json')))

def publish():
    audit_preservation()
    return old_publish()
C.publish=publish

def compare_pilot():
    checks=[];mismatches=[]
    plan=V.load(C.PLAN)
    for tag in plan['pilot_policies']:
        for cond in plan['conditions']:
            for seed in plan['pilot_games']:
                rel=Path('valid_unseen')/tag/cond/f's{seed}_r0.json'
                old=H/'data/pilot_20261110'/rel;new=D/'pilot'/rel
                a=V.load(old);b=V.load(new)
                for x in (a,b):
                    x.pop('provenance',None);x.pop('hardware_resume',None)
                if a!=b:mismatches.append(str(rel))
                checks.append(dict(path=str(rel),old_sha256=V.sha(old),new_sha256=V.sha(new),exact=a==b))
    result=dict(passed=not mismatches,episodes=len(checks),mismatches=mismatches,checks=checks,registration_sha256=V.sha(D/'registration.json'),criterion='Exact all stored episode fields except provenance/hardware receipt, including every generated response, action, feedback, won and step count. No tolerance relaxation.')
    V.save(D/'compatibility.json',result)
    assert result['passed'] and len(checks)==48,'Historical pilot differs; do not resume/mix full outputs'

def main():
    with (D/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            W.check_extension();C.state('hardware_resume_preflight');C.preflight(deep=True)
            import resume_eval_20261111 as E
            E.check_registration();audit_preservation()
            for tag in V.load(C.PLAN)['pilot_policies']:
                args=C.eval_args('pilot',tag);args[args.index('--out')+1]=str(D/'pilot')
                if not all((D/'pilot/valid_unseen'/tag/c/'summary.json').exists() for c in V.load(C.PLAN)['conditions']):
                    C.invoke('migration_pilot_'+tag,args)
            compare_pilot();C.state('hardware_resume_gate_passed')
            try:C.notify(f'【E4e换卡复现通过】48回合完整回复、动作及环境反馈逐项一致，记录 {D}/compatibility.json。自动续跑剩余E4e，然后GiGPO；全部核验上传credit-grok后关机。')
            except Exception:(D/'notification_error.log').write_text(traceback.format_exc())
            C.main()
        except Exception:C.recover(traceback.format_exc())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--watch',type=int);a=ap.parse_args()
    if a.watch:C.watch(a.watch)
    else:main()
