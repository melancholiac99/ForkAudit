"""Monitor frozen file availability; act only on the single registered look."""
from pathlib import Path
import fcntl,os,signal,subprocess,sys,time,traceback
import analyze_interim as A
C=A.C;F=A.F;R=F.parent;H=A.H;ROOT=H.parent

def state(stage,**kw):
    C.save_json(F/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))

def identity(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return None
    parts=p.read_text().split(') ',1)[1].split()
    return None if parts[0]=='Z' else parts[19]

def argv(pid):return [s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s]

def matches_score(words,output):
    return (str(H/'exp5_token_slot.py') in words and 'score' in words and
        '--out' in words and words[words.index('--out')+1]==output and
        '--name' in words and words[words.index('--name')+1]=='32b' and
        '--limit' in words and words[words.index('--limit')+1]=='0')

def wake(message):
    thread=os.environ.get('CODEX_THREAD_ID')
    if not thread:return
    with (F/'wake.log').open('a') as log:
        try:subprocess.run(['codex','queue','--thread',thread,'--resume','--message',message],stdout=log,stderr=subprocess.STDOUT,timeout=45)
        except Exception:traceback.print_exc(file=log)

def check_receipt():
    reg=A.checked_registration();receipt=C.read_json(F/'registration_receipt.json')
    assert receipt['registration_sha256']==C.sha_file(F/'registration.json')
    assert receipt['remote_verified_before_monitor_start'] and receipt['full_records_at_registration']==0
    for name in ('registration.json','PREREG_FUTILITY_20261107.txt','analyze_interim.py','monitor.py','test_futility.py'):
        p=F/name;rel=str(p.relative_to(ROOT))
        committed=subprocess.check_output(['git','show',receipt['commit']+':'+rel],cwd=ROOT)
        assert committed==p.read_bytes(),name
    return reg

def stop_own_queue(reg,d):
    assert d['decision']=='stop_for_futility'
    existing=F/'results/stop_receipt.json'
    if existing.exists():return C.read_json(existing)
    s=C.read_json(R/'status.json');assert s['stage']=='running' and s['task']=='full_score',s
    driver=s['pid'];child=s['child_pid'];driver_id=identity(driver);child_id=identity(child)
    assert driver_id and child_id and str(R/'driver.py') in argv(driver)
    assert matches_score(argv(child),reg['output'])
    intent=dict(driver_pid=driver,driver_identity=driver_id,child_pid=child,child_identity=child_id,decision_sha256=C.sha_file(F/'results/decision.json'),time=time.time())
    C.save_json(F/'results/stop_intent.json',intent)
    # Freeze the scheduler before terminating its current scoring child, so it
    # cannot misclassify the planned stop as an unexpected worker failure.
    os.kill(driver,signal.SIGSTOP)
    try:
        fresh=C.read_json(R/'status.json');assert fresh['task']=='full_score' and fresh['child_pid']==child
        assert identity(child)==child_id and matches_score(argv(child),reg['output'])
        os.kill(child,signal.SIGTERM)
        deadline=time.monotonic()+30
        while identity(child)==child_id and time.monotonic()<deadline:time.sleep(.2)
        if identity(child)==child_id:
            os.kill(child,signal.SIGKILL)
            for _ in range(50):
                if identity(child)!=child_id:break
                time.sleep(.1)
        assert identity(child)!=child_id,'Scoring process did not terminate'
        os.kill(driver,signal.SIGTERM);os.kill(driver,signal.SIGCONT)
        for _ in range(50):
            if identity(driver)!=driver_id:break
            time.sleep(.1)
        assert identity(driver)!=driver_id,'Driver did not terminate'
    except Exception:
        # Keep the registered futility decision explicit for recovery; do not
        # resume a scheduler which could silently restart stopped work.
        C.save_json(F/'results/stop_error.json',dict(intent=intent,error=traceback.format_exc(),driver_may_be_paused=True))
        raise
    full=Path(reg['output'])/'32b/full';files={str(p):C.sha_file(p) for p in (full/'records').glob('*.json.gz')}
    assert all(files.get(p)==h for p,h in d['record_hashes'].items())
    receipt=dict(**intent,status='stopped_early_for_futility',reason_zh='因无效提前停止',
        complete_records_preserved=files,decision_record_count=len(d['record_hashes']),
        additional_records_not_used_for_decision=sorted(set(files)-set(d['record_hashes'])),
        temporary_files_preserved=[str(p) for p in (full/'records').glob('*.tmp')],
        original_full_run_not_marked_complete=True,no_other_processes_signalled=True)
    C.save_json(existing,receipt)
    C.save_json(R/'status.json',dict(stage='stopped_early_for_futility',time=time.time(),decision=str(F/'results/decision.json'),stop_receipt=str(existing),no_shutdown_until_credit_complete=True))
    os.sync();return receipt

def partial_diagnostics(reg,d):
    import token_slot_analysis as T
    records,manifest=A.load_snapshot(reg);assert manifest==d['record_hashes']
    result=T.calculate(records)
    assert result['paired']['pre_k_T2_minus_T1_b1_value_pp']==d['pre_k_T2_minus_T1_value_pp']
    result.update(status='partial_after_futility_stop_exploratory',complete_full_experiment=False,
        registration_sha256=C.sha_file(F/'registration.json'),record_hashes=manifest,
        commitment_curve_by_point=[dict(id=r['id'],cluster=r['cluster'],k=r['k'],k_char=r['k_char'],curve=r['commitment_curve']) for r in records],
        interpretation='Stopped cohort: commitment alignment and native-token traces are descriptive. No early success claim; fixed-horizon full-cohort inference is unavailable.')
    C.save_json(F/'results/partial_diagnostics.json',result)
    assert all(C.sha_file(Path(p))==h for p,h in manifest.items())
    C.save_json(F/'results/partial_verification.json',dict(passed=True,raw_records_unchanged=len(records),
        decision_primary_matches_original_calculate=True,analysis_sha256=C.sha_file(F/'results/partial_diagnostics.json'),
        decision_sha256=C.sha_file(F/'results/decision.json'),full_run_complete=False))

def main():
    with (F/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            reg=check_receipt();folder=Path(reg['output'])/'32b/full/records'
            while True:
                present=sum((folder/(i+'.json.gz')).exists() for i in reg['checkpoint_ids'])
                state('waiting_for_registered_prefix',complete_prefix_records=present,required=reg['checkpoint_points'])
                if present==reg['checkpoint_points']:break
                s=C.read_json(R/'status.json')
                if s['stage']=='failed':raise RuntimeError('Scoring queue failed; futility not evaluated')
                if s['stage']=='running' and not identity(s['pid']):raise RuntimeError('Scoring driver exited; repair required')
                time.sleep(10)
            check_receipt();d=A.analyze();state('decision_recorded',decision=d['decision'])
            if d['decision']=='stop_for_futility':
                stop_own_queue(reg,d);partial_diagnostics(reg,d)
                state('stopped_early_for_futility',decision=str(F/'results/decision.json'))
                wake('【32B预登记无效判停】已在固定204点/30轨迹的一次检查触发CI上界<+3pp，按用户授权仅停止H800 32B队列。请读 '+str(F/'results')+'，核验部分诊断及停止凭据，仅上传research/hindsight-judges；不得重启此全量来追求结论。随后继续credit-grok的单H800迁移和剩余任务，两分支核验上传完成前不要关机。')
            else:
                state('one_look_complete_continue_to_full',decision=str(F/'results/decision.json'))
                wake('【32B中期检查继续全量】固定204点/30轨迹的一次无效检查未触发停止；请读 '+str(F/'results/decision.json')+'。原队列继续386点，不作提前成功判断，不再重复中期检查。')
        except Exception:
            error=traceback.format_exc();(F/'failure.log').write_text(error);state('failed',error=error)
            wake('【32B中期监控异常】请读 '+str(F/'failure.log')+' 和 results/stop_intent.json（如存在）。这是运行核验问题，不得当作无效结论；尽力修复，保留一次检查规则和原始数据。若已冻结driver须检查停止进度，不得盲目重启全量。')
            raise

if __name__=='__main__':main()
