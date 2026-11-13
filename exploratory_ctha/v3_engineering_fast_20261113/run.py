"""Engineering-only handoff; immutable parent implementation and precision."""
import sys,os,json,time,shutil,fcntl,traceback,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v3_fp32_fast'))
from util import *
import validate_pipeline as V
from precision import check_environment
PREVIOUS=Path('/root/autodl-tmp/ctha_v3_fp32_20261113')
OWN=Path(__file__).resolve().parent

def own_hashes():return {p.name:sha(p) for p in sorted(OWN.iterdir()) if p.suffix in ('.py','.json','.md')}
def live(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
original_job=V.job
def job(name,args):
 assert own_hashes()==load(DATA/'registration.json')['engineering_source_sha']
 return original_job(name,args)
V.job=job

def main():
 DATA.mkdir(exist_ok=True)
 with (DATA/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   check_environment();assert DATA!=PREVIOUS
   assert not (DATA/'registration.json').exists(),'No duplicate restart'
   previous=load(PREVIOUS/'registration.json');cpu=load(DATA/'cpu_validation.json')
   assert source_hashes()==cpu['source_sha']
   assert all(sha(ROOT/k)==v for k,v in previous['source_sha'].items()), 'Original baseline sources changed'
   bridge=load(OWN/'speed_bridge.json');assert bridge['exact_token_ids'] and bridge['max_behavior_logprob_difference']==0 and bridge['passed_original_parity_gate']
   assert cpu['legacy_tests_passed'] and cpu['cold_start_gate']['passed']
   immutable(DATA/'registration.json',{'time':time.time(),'source_sha':source_hashes(),'engineering_source_sha':own_hashes(),'plan':load(OWN/'PLAN.json'),'parent_plan':load(Path(V.HERE)/'PLAN.json'),'previous_registration_sha':sha(PREVIOUS/'registration.json'),'cross_kernel_bridge':bridge,'environment':{k:os.getenv(k) for k in [*previous['environment'],'CTHA_RUN_ROOT']},'gpu':previous['gpu']})
   save(DATA/'speed_bridge.json',bridge)
   for f in ['train_games.json','repeat_games.json']:shutil.copy2(PREVIOUS/f,DATA/f)
   adopted=load(PREVIOUS/'repeat_a_launch.json')['pid'];V.state('waiting_for_adopted_repeat_a',child=adopted)
   while live(adopted):time.sleep(10)
   raw=PREVIOUS/'repeat_a';summary=load(raw/'summary.json');prov=summary['provenance']
   expected=load(DATA/'repeat_games.json')
   assert summary['n_episodes']==8 and prov['source_sha256']==sha(Path(V.HERE)/'collect.py')
   assert prov['schema']=='ctha-v3-fp32' and prov['precision']=='float32_tf32x3_stages1' and prov['adapter_sha256'] is None
   assert prov['selected_game_ids']==[g['game_file'] for g in expected]
   assert len(list(raw.glob('s*_r*.json')))==8
   save(DATA/'reuse_manifest.json',{str(p):sha(p) for p in raw.glob('*.json')});shutil.copytree(raw,DATA/'repeat_a')
   V.collect('repeat_b',DATA/'repeat_games.json',DATA/'repeat_b',0,reps=2)
   V.compare_repeat(DATA/'repeat_a',DATA/'repeat_b')
   V.collect('initial_train',DATA/'train_games.json',DATA/'initial_train',0)
   V.compare_repeat(DATA/'repeat_a',DATA/'initial_train','cross_batch_verification.json')
   V.prepare('initial_prepare',DATA/'initial_train',DATA/'initial_prepared','grpo')
   V.train('direction_probe',DATA/'initial_train',DATA/'initial_prepared',DATA/'direction_probe',0,probe=True)
   job('checkpoint_verify',[OWN/'check_checkpoint.py'])
   job('adapter_parity_sample',[Path(V.HERE)/'adapter_parity.py','sample'])
   job('adapter_parity_score',[Path(V.HERE)/'adapter_parity.py','score'])
   job('independent_verify',[OWN/'verify.py','--data',DATA])
   V.state('engineering_complete',engineering='PASS',learning='NOT_RUN',generalization='NOT_RUN',publication='pending')
   V.notify(f'【CTHA工程验收完成：上传后关机】读取 {DATA}/verification.json、REPORT.txt及检查证据，独立复核后仅上传credit-grok并核对远端，再执行 /bin/bash /usr/bin/shutdown。用户只要工程验收，不运行四轮学习验证、完整P1/P2或A100实验。发布说明不得声称算法有效。发布工具入口 {OWN}/publish.py；旧结果不可覆盖。')
  except Exception:
   error=traceback.format_exc();save(DATA/'failure.json',{'error':error});(DATA/'failure.log').write_text(error);V.state('failed',error=error);os.sync()
   V.notify(f'【CTHA工程验收异常】读取 {DATA}/status.json、failure.log及对应日志；尽力修复，不放宽门槛、修订新目录。只做工程验收，不跑四轮学习验证。用户要求核验上传credit-grok后关机；先检查健康进程，不提前关闭仍在处理的验收。')
   raise
if __name__=='__main__':main()
