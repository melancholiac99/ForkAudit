"""User-requested P1-only verification/publication/shutdown; never starts P2."""
import argparse,fcntl,hashlib,json,math,os,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import server_queue_20261112 as C
H=C.H;ROOT=C.ROOT;D=H/'finalize_p1_20261113';OUT=H/'p1_results_20261113';DATA=Path('/root/autodl-tmp/ctha_v2_tokenfix');ARMS=['grpo','gigpo','fixed','ctha','progress'];C.D=D
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def cmd(args,timeout=300):
 p=subprocess.run(list(map(str,args)),cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
 if p.returncode:raise RuntimeError(str(args)+'\n'+p.stdout)
 return p.stdout.strip()
def state(stage,**kw):C.state(stage,**kw)
def verify():
 for n in ('ungated_s0','both_s0','no_center_s0'):assert not (DATA/n).exists(),f'Unexpected P2 directory: {n}'
 games=load(ROOT/'exploratory_E4/games.json');train={x['game_file'] for x in games['train']};tune={x['game_file'] for x in games['tune']};assert not train&tune
 old=load(H/'p1_server_20261112_tokenfix/preserved_manifest.json')
 for p,h in old.items():assert sha(p)==h,p
 summary={};weights={}
 modes={'grpo':('grpo','grpo','habit'),'gigpo':('grpo','gigpo','habit'),'fixed':('fixed','gigpo','habit'),'ctha':('ctha','gigpo','habit'),'progress':('ctha','gigpo','progress')}
 for name in ARMS:
  r=DATA/(name+'_s0');cfg=load(r/'config.json');assert (cfg['mode'],cfg['base_advantage'],cfg['family'])==modes[name]
  assert cfg['seed']==0 and cfg['train_horizon']==0 and cfg['reps']==4 and cfg['tasks_per_round']==16 and cfg['eval_every']==2 and cfg['eval_games']==40
  assert cfg['min_pairs']==200 and cfg['min_mixed']==20 and cfg['feature_window']==8 and cfg['aux_horizon']==8
  for f,h in cfg['code_sha'].items():assert sha(H/f)==h,f
  previous=None;steps=updates=aux=0
  for i in range(12):
   d=r/f'round_{i:03d}';s=load(d/'rollouts/summary.json');a=load(d/'prepared/advantage_report.json');t=load(d/'adapter/train_log.json')
   assert s['split']=='train' and s['n_episodes']==64 and s['n_games']==16 and not s['provenance']['dry']
   assert s['provenance']['adapter_sha256']==previous and t['adapter_sha256']==previous
   assert a['train_only'] and a['full_episode_policy_updates'] and a['train_horizon']==0
   assert sha(d/'prepared/samples.jsonl')==a['samples_sha256']==t['data_sha256']
   files=sorted((d/'rollouts').glob('s*_r*.json'));assert len(files)==64
   ids=set();ns=nwon=0
   for p in files:
    e=load(p);assert e['game_file'] in train and e['provenance']==s['provenance']
    assert e['n_turns']==len(e['steps'])<=50 and all(st['response_token_ids'] for st in e['steps'])
    key=(e['game_file'],e['rep']);assert key not in ids;ids.add(key);ns+=e['n_turns'];nwon+=int(e['won'])
    assert sha(p)==a['rollouts_sha256'][p.name]
   assert ns==s['env_steps'] and abs(nwon/64-s['success_rate'])<1e-12
   rows=[json.loads(l) for l in (d/'prepared/samples.jsonl').read_text().splitlines() if l.strip()]
   assert len(rows)==ns==t['n_source_steps'];assert sum(abs(x['advantage'])>=1e-12 for x in rows)==t['n_nonzero_advantage_steps']
   assert all(math.isfinite(x['loss']) for x in t['last_losses'])
   wf=d/'adapter/adapter_model.safetensors';previous=sha(wf);assert previous==t['weights_sha256'];weights[str(wf)]=dict(sha256=previous,bytes=wf.stat().st_size)
   steps+=ns;updates+=t['optimizer_steps'];aux+=a['nonzero_aux_samples']
   if i%2:
    e=load(d/'eval_tune/summary.json');assert e['split']=='tune' and e['n_episodes']==80 and e['n_games']==40 and e['provenance']['adapter_sha256']==previous
    es=[load(p) for p in (d/'eval_tune').glob('s*_r*.json')];assert len(es)==80 and all(x['game_file'] in tune for x in es)
    assert abs(sum(x['won'] for x in es)/80-e['success_rate'])<1e-12
  summary[name]=dict(rounds=12,train_episodes=768,tune_episodes=480,train_env_steps=steps,optimizer_steps=updates,aux_samples=aux,last_tune_success=e['success_rate'])
 OUT.mkdir(exist_ok=True);C.save(OUT/'verification.json',dict(passed=True,arms=summary,old_files_preserved=len(old),p2_not_run=True,single_seed_pilot=True));C.save(OUT/'checkpoint_manifest.json',weights)
 return summary
class Parts:
 def __init__(self,folder):self.folder=folder;self.file=None;self.size=0;self.paths=[]
 def write(self,b):
  total=len(b)
  while b:
   if self.file is None or self.size==80*1024**2:
    if self.file:self.file.close()
    p=self.folder/f'raw.tar.gz.part{len(self.paths):03d}';self.paths.append(p);self.file=p.open('wb');self.size=0
   n=min(len(b),80*1024**2-self.size);self.file.write(b[:n]);self.size+=n;b=b[n:]
  return total
 def close(self):
  if self.file:self.file.close()
def archive():
 target=OUT/('archive_'+str(time.time_ns()));target.mkdir();writer=Parts(target);manifest={}
 roots=[(DATA/(n+'_s0'),n+'_s0') for n in ARMS]+[(DATA/'smoke_update','smoke_update'),(Path('/root/autodl-tmp/ctha_v2/p0_base'),'p0_base'),(Path('/root/autodl-tmp/ctha_v2/smoke'),'smoke_original')]
 with tarfile.open(fileobj=writer,mode='w|gz') as tar:
  for root,label in roots:
   for p in sorted(root.rglob('*')):
    if not p.is_file() or p.is_symlink() or p.suffix not in ('.json','.jsonl','.log'):continue
    name=label+'/'+str(p.relative_to(root));manifest[name]=dict(sha256=sha(p),bytes=p.stat().st_size);tar.add(p,arcname=name,recursive=False)
 writer.close()
 from exploratory_credit_projection.audit import JoinedParts
 stream=JoinedParts(writer.paths);seen=set()
 try:
  with tarfile.open(fileobj=stream,mode='r|gz') as tar:
   for member in tar:
    if not member.isfile():continue
    assert member.name in manifest and member.name not in seen;seen.add(member.name)
    f=tar.extractfile(member);assert hashlib.sha256(f.read()).hexdigest()==manifest[member.name]['sha256']
 finally:stream.close()
 assert seen==set(manifest)
 C.save(OUT/'archive_manifest.json',dict(verified=True,files=manifest,parts=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in writer.paths],excluded='Model tensor checkpoints retained on server; hashes in checkpoint_manifest.json. All raw episode records are included.'))
def publish():
 assert cmd(['git','branch','--show-current'])=='credit-grok'
 assert cmd(['git','remote','get-url','origin'])=='https://github.com/melancholiac99/ForkAudit.git'
 assert not cmd(['git','diff','--cached','--name-only']),'Unrelated staged changes'
 files=sorted(p for p in OUT.rglob('*') if p.is_file())
 for p in files:assert p.stat().st_size<90*1024**2
 cmd(['git','add','--',str(OUT)])
 if cmd(['git','diff','--cached','--name-only']):cmd(['git','commit','-m','Publish verified CTHA P0 and P1 seed-0 results; P2 cancelled by user'])
 errors=[]
 for attempt in range(3):
  try:
   cmd(['git','-c','http.version=HTTP/1.1','push','origin','HEAD:credit-grok'],timeout=1200)
   cmd(['git','fetch','--no-tags','origin','credit-grok']);remote=cmd(['git','rev-parse','origin/credit-grok'])
   tree={l.split('\t')[1]:l.split('\t')[0].split()[2] for l in cmd(['git','ls-tree','-r',remote]).splitlines()}
   for p in files:assert tree[str(p.relative_to(ROOT))]==cmd(['git','hash-object',str(p)])
   C.save(D/'publication.json',dict(verified=True,commit=remote,files=len(files),branch='credit-grok'));return
  except Exception:
   errors.append(traceback.format_exc());C.save(D/'publication_errors.json',errors)
   if attempt<2:time.sleep(10)
 raise RuntimeError('Publication failed after three attempts; no force push. '+errors[-1])
def shutdown(outcome):
 assert load(D/'handoff.json')['shutdown_authorized']
 assert not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']),'GPU still active'
 state('shutting_down',outcome=outcome);os.sync()
 with (D/'shutdown.log').open('a') as f:p=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdout=f,stderr=subprocess.STDOUT,timeout=60)
 state('shutdown_command_returned',returncode=p.returncode,outcome=outcome);os.sync();assert p.returncode==0

def recover(error):
 (D/'failure.log').write_text(error);state('requesting_recovery',error=error);os.sync()
 C.notify(f'【CTHA P1收尾异常】用户要求P1结束核验上传后关机，P2取消。读取 {D}/status.json、failure.log；接管先写 {D}/recovery_claim.json，尽力修复，保留所有结果，不改门槛、不强推。入口 {H}/finalize_p1_20261113.py。30分钟无人接管且GPU空闲时保存日志sync后关机；修复成功确认远端上传再关机。')
 start=time.monotonic()
 while not (D/'recovery_claim.json').exists():
  if time.monotonic()-start>1800 and not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):shutdown('unrecoverable_logs_and_results_retained');return
  time.sleep(20)
 state('recovery_handoff')
def main():
 with (D/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   hand=load(D/'handoff.json');pid=hand['child'];state('waiting_p1',child=pid)
   while C.live(pid):
    assert Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]==hand['child_start_ticks']
    time.sleep(15)
   state('verifying');summary=verify()
   args=[sys.executable,str(H/'summarize.py'),'--out',str(OUT/'learning_curve.csv'),'--threshold','0.80']
   for n in ARMS:args+=['--run',n+'='+str(DATA/(n+'_s0'))]
   cmd(args)
   for src,name in [(H/'p1_server_20261112_tokenfix/p0_report.json','p0_report.json'),(H/'p1_server_20261112_tokenfix/nonzero_smoke_verification.json','nonzero_smoke_verification.json'),(D/'handoff.json','p1_only_authorization.json')]:shutil.copyfile(src,OUT/name)
   text=['CTHA P1 seed-0 pilot: 5 arms x 12 rounds. P2 cancelled by user.','Tune is development evaluation; no valid_unseen evaluation or multi-seed confirmation.']
   for n,v in summary.items():text.append(f"{n}: final tune {v['last_tune_success']:.2%}; train env steps {v['train_env_steps']}; auxiliary samples {v['aux_samples']}")
   curve=load(OUT/'learning_curve.json');text.append('Train environment steps to tune success >=80%: '+json.dumps(curve['env_steps_to_threshold'])+' (null = not reached within budget).')
   text.append('Raw P0/P1 episodes included in verified split gzip archive. Model tensors retained locally, checksums published. No thresholds changed.')
   (OUT/'REPORT.txt').write_text('\n'.join(text)+'\n');state('archiving');archive();state('publishing');publish()
   C.notify(f'【CTHA P1完成并上传】P2按用户要求未运行。读取 {OUT}/REPORT.txt、verification.json、learning_curve.json与 {D}/publication.json。P0/P1原始轨迹已归档上传，模型权重本地保留并公布哈希。单seed开发结果，控制器按用户要求关机。')
   shutdown('P1_verified_uploaded_P2_cancelled')
  except Exception:recover(traceback.format_exc())
if __name__=='__main__':
 sys.path.insert(0,str(ROOT))
 ap=argparse.ArgumentParser();ap.add_argument('--watch',type=int);a=ap.parse_args()
 if a.watch:
  while C.live(a.watch):time.sleep(20)
  if load(D/'status.json')['stage'] not in ('shutting_down','shutdown_command_returned','recovery_handoff','requesting_recovery'):recover('P1 finalizer exited unexpectedly')
 else:main()
