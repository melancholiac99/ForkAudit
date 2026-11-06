"""Read-only CPU follow-up; never launch, stop, or alter the two GPU workers."""
from pathlib import Path
import fcntl,os,subprocess,sys,time,traceback
RUN=Path(__file__).resolve().parent;H=RUN.parent;ROOT=H.parent
sys.path.insert(0,str(H));import exp2_common as C;import supplement_slot_analysis_20261106 as S
SCRIPT=H/'supplement_slot_analysis_20261106.py';OUT=H/'results/next_slot_supplement_20261106'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def wake(message):
 if os.environ.get('CODEX_THREAD_ID'):
  with (RUN/'supplement_wake.log').open('a') as f:
   try:subprocess.run(['codex','queue','--thread',os.environ['CODEX_THREAD_ID'],'--resume','--message',message],stdout=f,stderr=subprocess.STDOUT,timeout=45)
   except Exception:traceback.print_exc(file=f)
def ready(path):
 v=path/'verification.json'
 return v.exists() and C.read_json(v).get('passed') and (not (path/'run.json').exists() or C.read_json(path/'run.json').get('complete'))
def execute(args):
 with (RUN/'supplement_analysis.log').open('a') as f:
  p=subprocess.run([sys.executable,str(SCRIPT),*args],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,'Supplement failed: '+str(args)
def publish_status(stage,**kwargs):
 C.save_json(RUN/'supplement_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kwargs))
def main():
 with (RUN/'supplement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  source={str(p):C.sha_file(p) for p in (SCRIPT,H/'AMENDMENT_TOKEN_MATH_20261106.md',H/'test_supplement_slot_20261106.py',Path(__file__))}
  done={};announced=set()
  try:
   while True:
    assert all(C.sha_file(Path(p))==s for p,s in source.items()),'Supplement implementation changed while scheduled'
    for stage in ('first32','full'):
     request=H/'results/math_slot_20261106'/stage/'cut_review_request.json'
     selection=OUT/f'math_{stage}_blind_sample_selection.json'
     if request.exists() and not selection.exists():
      selected=S.continuity_sample(C.read_json(request)['questions'])
      C.save_json(selection,dict(ids=[q['id'] for q in selected],request_sha256=C.sha_file(request),
          rule='First24 by SHA256(20261106-continuity:ID); no outcome access',time=time.time()))
    fulls=[H/'results/token_slot_20261106'/m/'full' for m in ('7b','32b')]+[H/'results/math_slot_20261106/full']
    all_full_ready=all(ready(p) for p in fulls)
    jobs=[('token',m,'first32',H/'results/token_slot_20261106'/m/'first32') for m in ('7b','32b')]
    jobs += [('math',None,'first32',H/'results/math_slot_20261106/first32')]
    if all_full_ready:jobs += [('token',m,'full',fulls[i]) for i,m in enumerate(('7b','32b'))]+[('math',None,'full',fulls[2])]
    for kind,model,stage,original in jobs:
     key=f'token_{model}_{stage}' if kind=='token' else f'math_{stage}'
     if not ready(original):continue
     labels=OUT/f'math_{stage}_blind_labels.json'
     signature={'original_verification_sha':C.sha_file(original/'verification.json'),'labels_sha':C.sha_file(labels) if kind=='math' and labels.exists() else None}
     if done.get(key)==signature:continue
     args=['token','--model',model,'--cohort',stage] if kind=='token' else ['math','--stage',stage]
     if kind=='math' and labels.exists():args += ['--blind-labels',str(labels)]
     execute(args);result=C.read_json(OUT/(key+'.json'))
     assert result['supplement_script_sha256']==source[str(SCRIPT)] and result['original_verification_sha256']==signature['original_verification_sha']
     if kind=='math':assert result['blind_review']['sample_ids']==C.read_json(OUT/f'math_{stage}_blind_sample_selection.json')['ids']
     done[key]=signature
     C.save_json(OUT/(key+'_execution_check.json'),dict(passed=True,output_sha256=C.sha_file(OUT/(key+'.json')),source=source,original=signature,pilot_only=stage=='first32',blind_review_status=result.get('blind_review',{}).get('status')))
     if kind=='math' and result['blind_review']['status']!='complete' and key not in announced:
      announced.add(key);wake('【数学续写盲审】补充分析已生成 '+str(OUT/f'math_{stage}_blind_review.json')+'。先只读该文件，不查看正确率或条件映射；逐题给X/Y genuine_continuation / restart_or_rewrite / ambiguous标签，写同目录 math_'+stage+'_blind_labels.json，格式含 blind_payload_sha256 及 samples:[id,X,Y,reviewer,reason]。补充监控会汇总。两张GPU继续正常队列，无关机授权。')
    publish_status('waiting_for_verified_inputs_or_blind_review',completed=done,all_full_ready=all_full_ready)
    required=('token_7b_full','token_32b_full','math_full')
    if all(k in done for k in required) and C.read_json(OUT/'math_full.json')['blind_review']['status']=='complete':
     publish_status('completed',completed=done)
     wake('【三项补充分析完成】请核验 '+str(OUT)+' 的 full 报告，并列汇报legacy与supplement gate、动作区分span敏感性和数学盲审覆盖。结果仅发布 research/hindsight-judges，无关机授权。');return
    time.sleep(30)
  except Exception:
   error=traceback.format_exc();(RUN/'supplement_failure.log').write_text(error);publish_status('failed',error=error)
   wake('【补充分析异常】请读 '+str(RUN)+'/supplement_failure.log；只修独立补充代码，不修改或重启正常的两路GPU实验，不改原始结果和门槛。');raise
if __name__=='__main__':main()
