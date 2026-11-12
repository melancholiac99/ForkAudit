"""Nonzero-update engineering check, then unchanged README v2 P1/P2 seed-0 pilots."""
import argparse,json,math,os,shutil,sys,time,traceback,fcntl
from pathlib import Path
import server_queue_20261112 as C
H=C.H;D=H/'p1_server_20261112';C.D=D;DATA=C.DATA;MODEL=C.MODEL
ARMS=[('grpo','grpo','grpo','habit'),('gigpo','grpo','gigpo','habit'),('fixed','fixed','gigpo','habit'),('ctha','ctha','gigpo','habit'),('progress','ctha','gigpo','progress'),('ungated','ctha_nogate','gigpo','habit'),('both','ctha','gigpo','both'),('no_center','ctha_nocenter','gigpo','habit')]
def fail(error):
 (D/'failure.log').write_text(error);C.state('failed',error=error);os.sync()
 C.notify(f'【CTHA P1/P2异常】读取 {D}/status.json、failure.log和当前日志；写recovery_claim.json后尽力修复。入口 {H}/p1_queue_20261112.py。保留P0、原smoke及所有训练记录，不放宽门槛、不静默改指纹，代码改动需新目录。检查健康子进程避免重复训练；无关机授权。')
def preserve():
 for p,h in json.loads((D/'preserved_manifest.json').read_text()).items():assert C.sha(p)==h,p

def verify_update():
 import torch
 from safetensors.torch import load_file
 a=DATA/'smoke_update/round_000/adapter';j=json.loads((a/'train_log.json').read_text())
 assert j['optimizer_steps']>0 and j['n_nonzero_advantage_steps']>0 and all(math.isfinite(x['loss']) for x in j['last_losses'])
 assert C.sha(a/'adapter_model.safetensors')==j['weights_sha256']
 tensors=load_file(str(a/'adapter_model.safetensors'),device='cpu')
 assert all(torch.isfinite(v).all().item() for v in tensors.values())
 b={k:float(v.abs().max()) for k,v in tensors.items() if 'lora_B' in k};assert b and any(x>0 for x in b.values())
 ev=json.loads((DATA/'smoke_update/round_000/eval_tune/summary.json').read_text());assert ev['n_episodes']==12 and ev['split']=='tune' and ev['provenance']['adapter_sha256']==j['weights_sha256']
 C.save(D/'nonzero_smoke_verification.json',dict(passed=True,optimizer_steps=j['optimizer_steps'],nonzero_samples=j['n_nonzero_advantage_steps'],finite_tensors=True,lora_B_nonzero_count=sum(x>0 for x in b.values()),lora_B_count=len(b),tune_episodes=12,weights_sha256=j['weights_sha256'],engineering_only=True))
def main():
 with (D/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   C.preflight();preserve();C.state('supplemental_update_check')
   p=DATA/'smoke_update/round_000';src=DATA/'p0_base/round_000/rollouts'
   C.job('nonzero_train',[H/'train.py','--data',p/'prepared/samples.jsonl','--advantage-report',p/'prepared/advantage_report.json','--rollout-summary',src/'summary.json','--model',MODEL,'--out',p/'adapter','--seed','0'])
   C.job('nonzero_eval',[H/'rollout.py','--split','tune','--games',DATA/'smoke/eval_tune_games.json','--iteration','1','--model',MODEL,'--adapter',p/'adapter','--reps','2','--out',p/'eval_tune','--workers','8','--chunk-games','4'])
   verify_update();preserve();C.state('nonzero_smoke_passed')
   C.notify(f'【CTHA非零更新试跑通过】{D}/nonzero_smoke_verification.json 已确认实际优化器更新、非零LoRA B权重及新适配器tune评测。队列继续README_V2原参数的P1五组和P2三组，各12轮、seed0；无关机动作。')
   for name,mode,base,family in ARMS:
    assert shutil.disk_usage(DATA).free>12*1024**3,'Less than 12 GiB disk free; preserve and request recovery'
    C.job(name,[H/'run.py','--run-dir',DATA/(name+'_s0'),'--model',MODEL,'--rounds','12','--seed','0','--tasks-per-round','16','--reps','4','--feature-window','8','--train-horizon','0','--aux-horizon','8','--eval-every','2','--eval-games','40','--mode',mode,'--base-advantage',base,'--family',family])
    r=DATA/(name+'_s0')
    for i in range(12):
     rnd=r/f'round_{i:03d}';a=json.loads((rnd/'prepared/advantage_report.json').read_text());t=json.loads((rnd/'adapter/train_log.json').read_text());s=json.loads((rnd/'rollouts/summary.json').read_text())
     assert a['full_episode_policy_updates'] and a['train_horizon']==0 and a['train_only']
     assert s['n_episodes']==64 and s['split']=='train' and C.sha(rnd/'adapter/adapter_model.safetensors')==t['weights_sha256']
     if i%2:assert json.loads((rnd/'eval_tune/summary.json').read_text())['n_episodes']==80
    C.save(D/(name+'_completion.json'),dict(passed=True,rounds=12,run_dir=str(r)))
   args=[H/'summarize.py','--out',D/'learning_curve.csv','--threshold','0.80']
   for name,*_ in ARMS:args+=['--run',name+'='+str(DATA/(name+'_s0'))]
   C.job('summarize',args);preserve();C.state('completed_pending_independent_review');os.sync()
   C.notify(f'【CTHA P1/P2单seed完成】请独立核验 {D}/learning_curve.json 及八个完成凭据、/root/autodl-tmp/ctha_v2各run。报告env steps到80% tune门槛、曲线、GPU耗时及辅助信号是否激活；单seed不能作多seed确认性结论。核验后只上传credit-grok，保留旧实验，无关机授权。')
  except Exception:fail(traceback.format_exc())
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--watch',type=int);a=ap.parse_args()
 if a.watch:
  while C.live(a.watch):time.sleep(20)
  s=json.loads((D/'status.json').read_text())
  if s['stage'] not in ('failed','completed_pending_independent_review'):fail('Controller exited unexpectedly: '+json.dumps(s))
 else:main()
