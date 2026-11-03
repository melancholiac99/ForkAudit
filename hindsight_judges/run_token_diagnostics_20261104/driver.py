"""Append full-response diagnostics behind existing jobs. Never train HLN or shut down."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
from pathlib import Path
import fcntl,json,subprocess,sys,time,traceback

RUN=Path(__file__).resolve().parent;H=RUN.parent;ROOT=H.parent
OLD=H/'run_diagnostics_20261104'
sys.path.insert(0,str(H));import exp2_common as C
import exp3_token_diagnostic as T

TASKS={
 'cache_p4':dict(card=0,command='cache',cohort='p4',name='7b',deps=[]),
 'cache_e4':dict(card=0,command='cache',cohort='e4',name='7b',deps=['cache_p4']),
 'p4_7b':dict(card=0,command='score',cohort='p4',name='7b',deps=['cache_e4']),
 'p4_32b':dict(card=0,command='score',cohort='p4',name='32b',deps=['p4_7b']),
 'e4_32b':dict(card=1,command='score',cohort='e4',name='32b',deps=['cache_e4']),
}

def cmd(*a):return subprocess.check_output(list(map(str,a)),cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def state(stage,**kw):
    C.save_json(RUN/'status.json',dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
    print(stage,kw,flush=True)
def wake(message):
    if os.environ.get('CODEX_THREAD_ID'):
        with (RUN/'wake.log').open('a') as f:
            subprocess.run(['codex','queue','--thread',os.environ['CODEX_THREAD_ID'],'--resume','--message',message],stdout=f,stderr=subprocess.STDOUT,timeout=60)
def old_card_done(card):
    jobs=C.read_json(OLD/'jobs.json')
    # Original scheduler may assign Think localization to its first freed card.
    # Waiting for that job to finish prevents claiming a briefly idle pending slot.
    if 'localization' not in jobs or jobs['localization'].get('returncode')!=0:return False
    assigned=[j for j in jobs.values() if j['gpu_index']==card]
    return bool(assigned) and all(j.get('returncode')==0 for j in assigned)
def gpu_idle(uuid):
    lines=cmd('nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader').splitlines()
    return not any(line.split(',')[0].strip()==uuid for line in lines)
def progress(task):
    out=H/('results/token_diagnostic_'+task['cohort'])
    folder=out/'student_cache' if task['command']=='cache' else out/task['name']/'full'
    p=folder/'progress.json'
    return C.read_json(p) if p.exists() else None

def main():
    with (RUN/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=C.read_json(RUN/'execution.json');source=T.source_fingerprint()
            assert source==plan['source']
            for c in ['p4','e4']:
                groups,meta=T.prepared(H/('results/token_diagnostic_'+c))
                assert meta['groups_hash']==plan['datasets'][c]['groups_hash']
            inventory=cmd('nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader').splitlines()
            assert inventory==plan['gpus']
            uuids=[x.split(',')[1].strip() for x in inventory]
            pending=dict(TASKS);active={};done={};jobs={};benchmark_sent=False
            state('waiting_for_existing_diagnostics',plan=TASKS)
            while pending or active:
                oldstatus=C.read_json(OLD/'status.json')
                if oldstatus['stage'] in ['failed','scoring_with_failures']:
                    raise RuntimeError('Existing diagnostic queue needs repair first; preserving its running jobs')
                for key,(proc,log) in list(active.items()):
                    ret=proc.poll()
                    if ret is None:continue
                    log.close();del active[key];done[key]=ret
                    jobs[key].update(returncode=ret,ended=time.time());C.save_json(RUN/'jobs.json',jobs)
                    if ret:
                        wake('【完整回复诊断异常】'+key+' 失败。请读 '+str(RUN)+'/status.json、jobs.json 和对应日志。用户确认 M/L/LN/F，保留全部学生Think，E4必须冻结7B Student同前缀logprob，对照独立回放Q；只诊断不跑HLN/OPD，不关机。保留其他GPU任务，尽力修复；不放宽TV<0.005或混用配置。')
                for key,task in list(pending.items()):
                    if any(dep in done and done[dep]!=0 for dep in task['deps']):
                        done[key]=-1;del pending[key];continue
                    if not all(done.get(dep)==0 for dep in task['deps']):continue
                    card=task['card']
                    if any(TASKS[k]['card']==card for k in active):continue
                    if not old_card_done(card) or not gpu_idle(uuids[card]):continue
                    assert T.source_fingerprint()==source,'Source changed while queued; explicit review required'
                    out=H/('results/token_diagnostic_'+task['cohort'])
                    args=[sys.executable,'-u',str(H/'exp3_token_diagnostic.py'),task['command'],
                          '--cohort',task['cohort'],'--out',str(out),'--name',task['name']]
                    log=(RUN/(key+'.log')).open('a')
                    proc=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                          env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuids[card]))
                    jobs[key]=dict(pid=proc.pid,started=time.time(),gpu_uuid=uuids[card],command=args)
                    active[key]=(proc,log);del pending[key];C.save_json(RUN/'jobs.json',jobs)
                current={k:progress(TASKS[k]) for k in active}
                state('running' if active else 'waiting_for_existing_diagnostics',pending=list(pending),active=list(active),done=done,progress=current)
                e4=current.get('e4_32b')
                if e4 and e4['points']>=10 and not benchmark_sent:
                    benchmark_sent=True
                    C.save_json(RUN/'e4_initial_throughput.json',e4)
                    wake('【完整回复诊断吞吐已测得】E4 32B已完成至少10个真实点，请读 '+str(RUN/'status.json')+' 和 results/token_diagnostic_e4/32b/full/progress.json，按实测速率更新用户的剩余时间估计。初始点不一定代表后续长上下文。继续任务，不运行HLN或关机。')
                if pending or active:time.sleep(30)
            assert all(v==0 for v in done.values()) and len(done)==len(TASKS),str(done)
            state('analyzing_and_verifying')
            outputs=[]
            for cohort,name in [('p4','7b'),('p4','32b'),('e4','32b')]:
                out=H/('results/token_diagnostic_'+cohort)
                for stage in ['analyze','verify']:
                    with (RUN/(cohort+'_'+name+'_'+stage+'.log')).open('a') as f:
                        subprocess.run([sys.executable,'-u',str(H/'exp3_token_diagnostic.py'),stage,'--cohort',cohort,
                            '--out',str(out),'--name',name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
                outputs.append(str(out/name/'full/analysis.json'))
            for path,digest in plan['protected_old_results'].items():assert C.sha_file(path)==digest,path
            state('completed_verified',analyses=outputs,hln=False,shutdown=False);os.sync()
            wake('【完整回复诊断完成】P4 7B/32B、E4 32B Teacher与冻结7B Student logprob、双向全词表KL已完成分析核验。请读 '+str(RUN/'status.json')+' 及三个 analysis.json，结合独立回放Q汇报；不能从排序断言有效OPD梯度。本任务未运行HLN、训练或关机。')
        except BaseException as exc:
            state('failed',error=str(exc));traceback.print_exc();os.sync()
            wake('【完整回复诊断队列异常】请读 '+str(RUN/'status.json')+' 和 driver.log，检查GPU现有任务后尽力修复。用户只授权诊断，不跑HLN/OPD，不关机；不放宽门槛，不混用运行。')
            raise

if __name__=='__main__':main()
