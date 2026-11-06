"""Adopt live jobs; finish 7B only, then use both GPUs for frozen math work."""
from pathlib import Path
import fcntl,os,signal,subprocess,sys,time,traceback
R=Path(__file__).resolve().parent;H=R.parent;ROOT=H.parent
sys.path.insert(0,str(H))
import exp2_common as C,exp5_token_slot as S,exp6_math_slot as E
import driver as D,math_two_gpu_worker as W
A=W.A

def identity(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return None
    fields=p.read_text().split(') ',1)[1].split()
    return None if fields[0]=='Z' else fields[19]

def active(job):return identity(job['pid'])==job['identity']

def stop(job):
    if not active(job):return
    os.kill(job['pid'],signal.SIGTERM)
    for _ in range(60):
        if not active(job):return
        time.sleep(1)
    raise RuntimeError('Process did not terminate cleanly: '+str(job))

def check_sources(plan):
    assert S.fingerprint()==plan['token_source'] and E.fingerprint()==plan['math_source']
    assert all(C.sha_file(Path(p))==s for p,s in plan['orchestration_source'].items())

def dual(stage,rows,plan):
    if not rows:return
    path=A/(stage+'_assignment.json')
    if path.exists():assignment=C.read_json(path)
    else:
        partial=[r['id'] for r in rows if any((E.OUT/'screen_generations'/r['id']).glob('*.json'))] if stage=='screen' else []
        assignment=dict(all_ids=[r['id'] for r in rows],parts=W.split_questions(rows,partial),
                        rule='Canonical question order, balanced by pending question count; partial screening stays original GPU',
                        math_source=plan['math_source'],time=time.time())
        C.save_json(path,assignment)
    assert assignment['all_ids']==[r['id'] for r in rows]
    assert sorted(sum(assignment['parts'].values(),[]))==sorted(assignment['all_ids'])
    children={}
    try:
        attempt=str(time.time_ns())
        for role,gpu in plan['gpus'].items():
            folder=A/stage/role;folder.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,MATH_DUAL_ATTEMPT=attempt,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
            with (folder/'worker.log').open('a') as log:
                children[role]=subprocess.Popen([sys.executable,'-u',str(R/'math_two_gpu_worker.py'),stage,role],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        C.save_json(A/(stage+'_launch.json'),dict(time=time.time(),pids={k:p.pid for k,p in children.items()}))
        D.state('math','dual_gpu_'+stage,children={k:p.pid for k,p in children.items()})
        while any(p.poll() is None for p in children.values()):
            assert all(p.poll() in (None,0) for p in children.values()),'Dual math worker failed: '+stage
            states={role:C.read_json(A/stage/role/'status.json') if (A/stage/role/'status.json').exists() else {'stage':'loading_or_cross_card_check'} for role in children}
            C.save_json(E.OUT/'status.json',dict(stage='dual_gpu_'+stage,time=time.time(),pid=os.getpid(),workers=states))
            time.sleep(10)
        assert all(p.returncode==0 for p in children.values())
        for role in children:
            done=C.read_json(A/stage/role/'completion.json');assert done['passed']
            assert all(C.sha_file(Path(p))==s for p,s in done['outputs'].items())
            assert C.read_json(A/stage/role/'cross_card_check.json')['passed']
    finally:
        for p in children.values():
            if p.poll() is None:p.terminate()
        for p in children.values():
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:pass

def main():
    locks=[]
    for path in (R/'token_driver.lock',R/'math_driver.lock',E.OUT/'queue.lock',A/'controller.lock'):
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    plan=C.read_json(A/'plan.json');check_sources(plan)
    C.save_json(A/'controller_status.json',dict(stage='waiting_for_current_7b',pid=os.getpid(),time=time.time()))
    D.state('token','running',task='token_7b_score_adopted_no_32b_followup',child_pid=plan['token_child']['pid'])
    D.state('math','running',task='math_screen_adopted_until_7b_finishes',child_pid=plan['math_child']['pid'])
    try:
        while active(plan['token_child']):
            if not active(plan['math_child']):
                _,rows=E.prepared()
                assert all((E.OUT/'screen'/(r['id']+'.json')).exists() for r in rows if C.read_json(E.OUT/'student'/(r['id']+'.json'))['eligible']), 'Original math screening exited early'
            time.sleep(15)
        check_sources(plan)
        run=C.read_json(S.OUT/'7b/full/run.json');assert run['complete'],'7B scoring exited without complete checkpoint'
        assert len(C.read_json(S.OUT/'7b/full/manifest.json'))==386
        for command in ('analyze','verify'):
            D.execute('token','token_7b_full_'+command+'_final_scope',[H/'exp5_token_slot.py',command,'--name','7b','--limit',0])
        assert C.read_json(S.OUT/'7b/full/verification.json')['passed']
        D.state('token','completed_7b_verified_32b_full_cancelled',models_completed=['7b'],cancelled=['32b_full'])
        D.wake('【7B已完成，转双卡数学】逐token 7B全量已核验；按用户要求取消32B全量，保留已有32B pilot。现在接管数学剩余工作，双卡数值检查通过才生成；无关机授权。')
        # Only an uncommitted in-flight response can be lost; saved JSON files are atomic.
        stop(plan['math_child'])
        if not (A/'preserved_math_hashes.json').exists():
            files=list((E.OUT/'screen_generations').glob('*/*.json'))+list((E.OUT/'screen').glob('*.json'))
            C.save_json(A/'preserved_math_hashes.json',{str(p):C.sha_file(p) for p in files})
        _,rows=E.prepared()
        eligible=[r for r in rows if C.read_json(E.OUT/'student'/(r['id']+'.json'))['eligible']]
        pending=[r for r in eligible if not (E.OUT/'screen'/(r['id']+'.json')).exists()]
        dual('screen',pending,plan)
        assert all((E.OUT/'screen'/(r['id']+'.json')).exists() for r in eligible)
        assert all(C.sha_file(Path(p))==s for p,s in C.read_json(A/'preserved_math_hashes.json').items())
        target=plan['target'];E.request_review('full',target)
        D.state('math','awaiting_full_cut_review')
        annotations=E.wait_file(E.OUT/'full/cut_annotations.json',
            '【数学全量截断待核验】请读 '+str(E.OUT/'full/cut_review_request.json')+'；按冻结预算逐题审查，原32题复用pilot标注，新题保留原边界。两卡将按题分配四臂，评分前完成标注及补充盲审准备。无关机授权。',lambda:E.cut_annotations('full',target))
        original=C.read_json(E.OUT/'first32/cut_annotations.json')['questions']
        frozen=C.read_json(E.OUT/'full_budget_inputs.json')['frozen_cut_proposals']
        for qid,a in annotations['questions'].items():
            assert a['prefix_end']==frozen[qid]['prefix_end']
            assert C.sha_file(E.OUT/'student'/(qid+'.json'))==frozen[qid]['student_sha256']
            if qid in original:assert a==original[qid]
        dual('evaluate',E.selected('full',target),plan)
        D.execute('math','math_full_analyze_dual',[H/'exp6_math_slot.py','analyze','--stage','full','--target',target])
        D.state('math','awaiting_full_outcome_review')
        E.wait_file(E.OUT/'full/outcome_review.json','【数学全量输出待核验】请按 '+str(E.OUT/'full/outcome_review_request.json')+' 完成必需抽样/格式核验，保留原始响应与自动判定。逐token仅7B全量，32B已取消；无关机授权。',lambda:C.read_json(E.OUT/'full/outcome_review.json'))
        for command in ('analyze','verify'):
            D.execute('math','math_full_'+command+'_dual_final',[H/'exp6_math_slot.py',command,'--stage','full','--target',target])
        assert C.read_json(E.OUT/'full/verification.json')['passed']
        check_sources(plan)
        D.state('math','completed_verified')
        C.save_json(A/'controller_status.json',dict(stage='completed_verified',time=time.time(),no_shutdown=True))
        D.wake('【7B及双卡数学完成】请核验并发布 research/hindsight-judges。逐token32B全量按用户要求未运行，保留pilot；数学按题双卡且数值门槛通过。检查 '+str(A)+'。无关机授权。')
    except Exception:
        error=traceback.format_exc();(A/'failure.log').write_text(error)
        C.save_json(A/'controller_status.json',dict(stage='failed',time=time.time(),error=error))
        D.wake('【7B转双卡数学异常】请读 '+str(A/'failure.log')+' 并尽力修复。不得恢复已取消的逐token32B全量，不改原结果和门槛，无关机授权。');raise

if __name__=='__main__':main()
