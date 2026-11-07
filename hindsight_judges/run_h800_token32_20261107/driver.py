"""Run the frozen 32B protocol on H800 in an isolated result directory."""
from pathlib import Path
import fcntl, os, subprocess, sys, time, traceback
import numpy as np
R=Path(__file__).resolve().parent;H=R.parent;ROOT=H.parent
sys.path.insert(0,str(H))
import exp2_common as C
import exp5_token_slot as S

def state(stage,**kw):
    C.save_json(R/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))
    print(stage,kw,flush=True)

def wake(message):
    thread=os.environ.get('CODEX_THREAD_ID')
    if not thread:return
    with (R/'wake.log').open('a') as log:
        try:subprocess.run(['codex','queue','--thread',thread,'--resume','--message',message],stdout=log,stderr=subprocess.STDOUT,timeout=45)
        except Exception:traceback.print_exc(file=log)

def timing(plan,final=False):
    out=Path(plan['output']);old=Path(plan['old_output'])/'32b/first32';new=out/'32b/first32'
    groups,_=C.decode_rows(out/'groups.jsonl.gz');pairs=[]
    for g in groups[:32]:
        p=new/'records'/(g['id']+'.json.gz')
        if not p.exists():continue
        a=S.read(old/'records'/p.name);b=S.read(p)
        assert a['group_sha256']==b['group_sha256'] and a['model_calls']==b['model_calls']
        pairs.append(dict(id=g['id'],old_seconds=a['seconds'],new_seconds=b['seconds'],model_calls=b['model_calls']))
    if len(pairs)<2:return False
    old_seconds=sum(p['old_seconds'] for p in pairs);new_seconds=sum(p['new_seconds'] for p in pairs)
    audit=[]
    for p in sorted((new/'vocab_audits').glob('*.npz')):
        q=old/'vocab_audits'/p.name
        if not q.exists():continue
        with np.load(q) as aa,np.load(p) as bb:
            assert int(aa['y'])==int(bb['y'])
            for d in ('AF','AM','EF','EM'):
                tv=float(np.abs(np.exp(aa[d].astype(np.float64))-np.exp(bb[d].astype(np.float64))).sum()/2)
                audit.append(dict(file=p.name,distribution=d,tv=tv,below_original_0p005=tv<.005))
    result=dict(points=len(pairs),paired_records=pairs,old_seconds=old_seconds,h800_seconds=new_seconds,
        matched_speedup=old_seconds/new_seconds,seconds_per_point=new_seconds/len(pairs),
        estimated_386_score_hours=new_seconds/len(pairs)*386/3600,
        final_pilot=final,cross_hardware_full_vocab_comparisons=audit,
        cross_hardware_max_tv=max((x['tv'] for x in audit),default=None),
        caveat='Scoring time excludes model loading and CPU analysis. Early points need not represent longer contexts. Old Teacher scores are comparison-only, never merged into the H800 run. Frozen 7B Student reference is reused explicitly.')
    C.save_json(R/('pilot_timing.json' if final else 'early_timing.json'),result)
    return True

def check(plan):
    assert S.fingerprint()==plan['source']
    assert C.sha_file(Path(__file__))==plan['driver_sha256']
    for p,h in plan['frozen_metadata'].items():assert C.sha_file(Path(p))==h,p
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    assert inventory==plan['gpu_inventory']

def execute(plan,key,args,gpu):
    check(plan)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu_uuid'] if gpu else '',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    with (R/(key+'.log')).open('a') as log:
        p=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    state('running',task=key,child_pid=p.pid,output=plan['output'])
    announced=(R/'early_timing.json').exists()
    try:
        while p.poll() is None:
            if key=='pilot_score' and not announced and timing(plan):
                announced=True
                wake('【H800 32B实测测速】已完成至少两个与6000D相同的真实点，请读 '+str(R/'early_timing.json')+' 更新倍率和预计时长。当前只在新目录运行H800 Teacher，冻结7B参考复用；不要混入旧32B分数。按用户授权继续32点核验后386点全量；credit-grok尚待换卡续跑，不得提前关机。')
            time.sleep(10)
        assert p.returncode==0,f'{key} failed: {p.returncode}'
    finally:
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass

def main():
    with (R/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=C.read_json(R/'plan.json')
        try:
            check(plan)
            for limit,stage in ((32,'pilot'),(0,'full')):
                for command in ('score','analyze','verify'):
                    execute(plan,stage+'_'+command,[H/'exp5_token_slot.py',command,'--name','32b','--model',plan['model'],'--out',plan['output'],'--limit',limit],command=='score')
                folder=Path(plan['output'])/'32b'/('first32' if limit else 'full')
                v=C.read_json(folder/'verification.json');assert v['passed'] and v['points']==(32 if limit else 386)
                assert v['analysis_sha256']==C.sha_file(folder/'analysis.json')
                run=C.read_json(folder/'run.json');assert run['complete'] and run['fingerprint']['gpu_uuid']==plan['gpu_uuid']
                old=C.read_json(Path(plan['old_output'])/'32b/first32/run.json')['fingerprint']
                for k in ('source','model','groups_hash','name','cache_policy','memo_limit','workers'):
                    assert old[k]==run['fingerprint'][k],k
                if limit:
                    timing(plan,True)
                    wake('【H800 32B pilot核验通过】请读 '+str(R/'pilot_timing.json')+'。按授权继续全量386点，旧6000D分数不混入；评分配置及TV门槛未改。credit-grok仍待H800迁移，不能只因32B完成就关机。')
            execute(plan,'supplement',[H/'supplement_slot_analysis_20261106.py','token','--model','32b','--cohort','full','--token-root',plan['output'],'--output',plan['supplement_output']],False)
            report=Path(plan['supplement_output'])/'token_32b_full.json';d=C.read_json(report)
            full=Path(plan['output'])/'32b/full'
            assert d['original_verification_sha256']==C.sha_file(full/'verification.json')
            assert d['original_analysis_sha256']==C.sha_file(full/'analysis.json')
            check(plan)
            state('completed_verified',output=plan['output'],supplement=str(report),no_shutdown_until_all_tasks_done=True)
            wake('【H800 32B全量完成】请核验 '+plan['output']+' 与 '+plan['supplement_output']+'，并列汇报legacy/supplement gate及跨硬件对照；仅推research/hindsight-judges。然后接管credit-grok的单H800迁移与剩余实验，保留旧结果、冻结队列和门槛。两分支各自核验上传完成后才关机，不得现在提前关机。')
        except Exception:
            error=traceback.format_exc();(R/'failure.log').write_text(error);state('failed',error=error)
            wake('【H800 32B异常】请读 '+str(R/'status.json')+'、failure.log和对应日志，尽力修复续跑。不得混入旧32B分数、放宽TV/统计门槛或覆盖旧结果；credit-grok仍有未完任务，不能因本阶段失败立即关机。')
            raise

if __name__=='__main__':main()
