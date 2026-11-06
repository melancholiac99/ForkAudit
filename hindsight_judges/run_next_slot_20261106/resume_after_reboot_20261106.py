"""Resume verified pilots at full stages after user-authorized reboot.

Only orchestration changes: use original score/screen/evaluate/analyze/verify
commands with their unchanged numerical policy, resume guards and review gates.
Do not reload models merely to visit already verified Student/pilot stages.
"""
from pathlib import Path
import argparse,fcntl,os,sys,time,traceback,subprocess
R=Path(__file__).resolve().parent;H=R.parent;ROOT=H.parent
sys.path.insert(0,str(H))
import exp2_common as C,exp5_token_slot as S,exp6_math_slot as E
import token_slot_backend as K,math_slot_core as M
import driver as D
AUDIT=R/'reboot_resume_20261106'

def verify_token_pilot(name):
    p=S.OUT/name/'first32';v=C.read_json(p/'verification.json');run=C.read_json(p/'run.json')
    assert v['passed'] and v['source']==S.fingerprint() and run['complete']
    assert v['analysis_sha256']==C.sha_file(p/'analysis.json')
    assert run['fingerprint']['source']==S.fingerprint()
    assert run['manifest_sha256']==C.sha_file(p/'manifest.json')
    manifest=C.read_json(p/'manifest.json');assert len(manifest)==32
    assert all(C.sha_file(p/'records'/(qid+'.json.gz'))==sha for qid,sha in manifest.items())

def preflight():
    plan=C.read_json(R/'execution.json');assert plan['mode']=='parallel' and plan['user_parallel_authorized']
    assert not (R/'USER_PAUSED.json').exists()
    assert plan['source']==S.fingerprint()
    groups,meta=S.prepared(S.OUT,0);by={g['id']:g for g in groups}
    for name in ('7b','32b'):verify_token_pilot(name)
    folder=S.OUT/'7b/full';run=C.read_json(folder/'run.json')
    assert run['fingerprint']['source']==S.fingerprint()
    token_records={}
    for f in sorted((folder/'records').glob('*.json.gz')):
        x=S.read(f);assert x['id']==f.name[:-8] and x['group_sha256']==by[x['id']]['sha256']
        assert x['fingerprint_sha256']==K.digest(run['fingerprint'])
        token_records[x['id']]=C.sha_file(f)
    prep,qs=E.prepared();rows={q['id']:q for q in qs}
    ready=C.read_json(R/'math_ready.json');assert ready['cpu_verified']
    assert all(C.sha_file(Path(p))==sha for p,sha in ready['source'].items())
    pilot=E.OUT/'first32';v=C.read_json(pilot/'verification.json')
    assert v['passed'] and v['source']==E.fingerprint()
    for field,file in [('analysis_sha','analysis.json'),('review_sha','outcome_review.json'),('annotations_sha','cut_annotations.json')]:
        assert v[field]==C.sha_file(pilot/file)
    a=C.read_json(pilot/'analysis.json')
    assert all(C.sha_file(Path(p))==sha for p,sha in a['raw_results_hashes'].items())
    budget=C.read_json(E.OUT/'full_budget.json');bv=C.read_json(E.OUT/'full_budget_verification.json')
    assert bv['passed'] and bv['budget_sha256']==C.sha_file(E.OUT/'full_budget.json')
    assert budget['pilot_verification_sha']==C.sha_file(pilot/'verification.json')
    assert budget['math_source']==E.fingerprint() and budget['basis']=='timing_and_cut_quality'
    assert budget['inputs_sha256']==C.sha_file(E.OUT/'full_budget_inputs.json')
    screening={};complete_screen={}
    for f in sorted((E.OUT/'screen_generations').glob('*/*.json')):
        qid=f.parent.name;i=int(f.stem);assert qid in rows and i in range(4)
        x=C.read_json(f);g=x['generation']
        assert g['seed']==M.seed('screen',qid,i) and g['generation']==E.CONFIG
        assert len(g['ids'])==len(g['logp']) and g['stop'] in ('eos','length')
        assert C.read_json(E.OUT/'student'/(qid+'.json'))['eligible']
        screening[str(f.relative_to(E.OUT))]=C.sha_file(f)
    for f in sorted((E.OUT/'screen').glob('*.json')):
        x=C.read_json(f);assert x['id']==f.stem
        for i,sha in x['records'].items():assert C.sha_file(E.OUT/'screen_generations'/f.stem/(i+'.json'))==sha
        complete_screen[f.stem]=C.sha_file(f)
    result=dict(passed=True,time=time.time(),gpu_assignments=plan['gpu_assignments'],
        source=S.fingerprint(),math_source=E.fingerprint(),full_budget_sha256=C.sha_file(E.OUT/'full_budget.json'),
        token_full_saved_points=len(token_records),token_record_hashes=token_records,
        math_screened_questions=len(complete_screen),math_screen_record_hashes=complete_screen,
        math_saved_screen_responses=len(screening),math_screen_response_hashes=screening,
        pilot_verifications={n:C.sha_file(S.OUT/n/'first32/verification.json') for n in ('7b','32b')},
        math_pilot_verification_sha256=C.sha_file(pilot/'verification.json'),
        no_scientific_code_or_configuration_changes=True,no_shutdown=True)
    AUDIT.mkdir(exist_ok=True);C.save_json(AUDIT/'preflight.json',result)
    print('PREFLIGHT',result['token_full_saved_points'],'token points;',len(complete_screen),'screened questions;',len(screening),'screen responses',flush=True)

def gpu_check(name):
    import torch,exp1c_precision,token_diagnostic_backend as B
    plan=C.read_json(R/'execution.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==plan['gpu_assignments'][name]
    assert torch.cuda.device_count()==1
    exp1c_precision.set_flags(torch)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    runtime=B.runtime()
    expected=(C.read_json(S.OUT/'7b/full/run.json')['fingerprint']['runtime'] if name=='token'
              else C.read_json(E.OUT/'stages/screen/model.json')['runtime'])
    assert runtime==expected, 'Runtime mismatch: do not mix reboot configuration'
    with torch.inference_mode():
        z=torch.empty(16*1024*1024,device='cuda',dtype=torch.float32);z.fill_(1);z.add_(1)
        assert float(z.sum())==2*len(z)
        x=torch.ones((512,512),device='cuda');y=x@x
        assert bool((y==512).all());torch.cuda.synchronize()
    C.save_json(AUDIT/(name+'_gpu_check.json'),dict(passed=True,runtime=runtime,matched_prior_runtime=True,time=time.time()))
    print('GPU_CHECK',name,'passed, exact runtime match',flush=True)

def full_math():
    with (E.OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        E.prepared();b=C.read_json(E.OUT/'full_budget.json');target=b['target']
        assert b['basis']=='timing_and_cut_quality' and 1<=target<=200
        assert b['pilot_verification_sha']==C.sha_file(E.OUT/'first32/verification.json')
        D.execute('math','math_full_screen_resume',[H/'exp6_math_slot.py','screen','--target',target])
        E.request_review('full',target)
        full_annotations=E.wait_file(E.OUT/'full/cut_annotations.json',
            '【数学全量截断待核验】请核验 '+str(E.OUT/'full/cut_review_request.json')+'。遵守 full_budget.json：原32题逐项复用 pilot 标注，新题保留原提议边界；只凭截断前已写推导区分 error_observed/error_unknown，四臂生成前保存 full/cut_annotations.json。另一GPU继续，无关机授权。',
            lambda:E.cut_annotations('full',target))
        pilot_annotations=C.read_json(E.OUT/'first32/cut_annotations.json')['questions']
        frozen_cuts=C.read_json(E.OUT/'full_budget_inputs.json')['frozen_cut_proposals']
        for qid,a in full_annotations['questions'].items():
            assert a['prefix_end']==frozen_cuts[qid]['prefix_end']
            assert C.sha_file(E.OUT/'student'/(qid+'.json'))==frozen_cuts[qid]['student_sha256']
            if qid in pilot_annotations:assert a==pilot_annotations[qid], 'Pilot annotation must be reused unchanged'
        for command in ('evaluate','analyze'):
            D.execute('math','math_full_'+command+'_resume',[H/'exp6_math_slot.py',command,'--stage','full','--target',target])
        E.wait_file(E.OUT/'full/outcome_review.json',
            '【数学全量输出待核验】请检查 '+str(E.OUT/'full/outcome_review_request.json')+' 的必需抽样/异常格式，保存 outcome_review.json；保留自动评分、原始响应和门槛。另一GPU继续，无关机授权。',
            lambda:C.read_json(E.OUT/'full/outcome_review.json'))
        for command in ('analyze','verify'):
            D.execute('math','math_full_'+command+'_resume',[H/'exp6_math_slot.py',command,'--stage','full','--target',target])
        assert C.read_json(E.OUT/'full/verification.json')['passed']

def worker(name):
    with (R/(name+'_driver.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=C.read_json(R/'execution.json');audit=C.read_json(AUDIT/'preflight.json')
        assert audit['passed'] and audit['source']==S.fingerprint() and audit['math_source']==E.fingerprint()
        assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_assignments'][name]
        assert C.read_json(AUDIT/(name+'_gpu_check.json'))['passed']
        try:
            if name=='token':
                for model in ('7b','32b'):
                    for command in ('score','analyze','verify'):
                        assert plan['source']==S.fingerprint()
                        D.execute(name,f'token_{model}_0_{command}_resume',[H/'exp5_token_slot.py',command,'--name',model,'--limit',0,'--model',f'/root/autodl-tmp/models/Qwen2.5-{model.upper()}-Instruct'])
                    assert C.read_json(S.OUT/model/'full/verification.json')['passed']
            else:full_math()
            D.state(name,'completed_verified')
            D.wake('【'+name+' 全量实验完成】重启续跑完成并核验。请检查 '+str(R)+'/'+name+'_status.json，汇报并发布 research/hindsight-judges；另一实验继续，无关机授权。')
        except Exception:
            error=traceback.format_exc();(R/(name+'_failure.log')).write_text(error)
            D.state(name,'failed',error=error)
            D.wake('【重启续跑异常 '+name+'】请读 '+str(R)+'/'+name+'_failure.log 并修复。保护另一正常GPU队列、原数据和配置，不放宽门槛，无关机授权。')
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--gpu-check',choices=('token','math'));p.add_argument('--worker',choices=('token','math'));a=p.parse_args()
    if a.preflight:preflight()
    elif a.gpu_check:gpu_check(a.gpu_check)
    elif a.worker:worker(a.worker)
    else:p.error('Choose --preflight, --gpu-check or --worker')
