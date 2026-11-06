"""Sequential MATH screening and four-arm generation, with pre-outcome review gates."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,'/root/autodl-tmp/math-slot-deps')
import exp2_common as C
import math_slot_core as M
OUT=HERE/'results/math_slot_20261106';RUN=HERE/'run_next_slot_20261106'
MODEL_ROOT=Path('/root/autodl-tmp/models')
CONFIG=dict(max_new_tokens=2048,temperature=.7,top_p=1.,top_k=0)
GPU='GPU-2d103f24-c0fc-18b0-c2f3-9edd2710692a'

def fingerprint():
 files=['exp6_math_slot.py','math_slot_core.py','test_math_slot.py','MATH_SLOT_EXECUTION_20261106.txt','PROTOCOL_NEXT_TOKEN_SLOT_AND_MATH_20261106.md','token_diagnostic_backend.py','exp1c_backend.py','exp1c_precision.py','exp2_common.py','slot_model_loader.py']
 return {f:C.sha_file(HERE/f) for f in files}
def dep_digest():
 d=hashlib.sha256()
 for p in sorted(M.DEPS.rglob('*')):
  if p.is_file() and p.suffix in ('.py','.json') or p.is_file() and p.name=='METADATA':
   d.update(str(p.relative_to(M.DEPS)).encode());d.update(p.read_bytes())
 return d.hexdigest()
def jwrite(p,x):C.save_json(p,x)
def jread(p):return C.read_json(p)
def sha(x):return hashlib.sha256(C.json_bytes(x)).hexdigest()
def status(stage,**kw):jwrite(OUT/'status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw));print(stage,kw,flush=True)
def wake(text):
 if not os.environ.get('CODEX_THREAD_ID'):return
 with (RUN/'math_wake.log').open('a') as f:
  try:subprocess.run(['codex','queue','--thread',os.environ['CODEX_THREAD_ID'],'--resume','--message',text],stdout=f,stderr=subprocess.STDOUT,timeout=45)
  except Exception:traceback.print_exc(file=f)
def prepare():
 import pyarrow.parquet as pq
 api=jread(Path('/root/autodl-tmp/math-dataset-api.json'));data=Path('/root/autodl-tmp/math-slot-data')
 files={};rows=[]
 for item in api['siblings']:
  name=item['rfilename']
  if '/test-' not in name:continue
  p=data/name.replace('/','__');assert p.exists(),str(p);files[name]=C.sha_file(p)
  subject=name.split('/')[0]
  for i,r in enumerate(pq.read_table(p).to_pylist()):
   if r['level'] not in ('Level 3','Level 4','Level 5'):continue
   qid=f'{subject}:{i}';rows.append(dict(id=qid,problem=r['problem'],gold=r['solution'],level=r['level'],subject=subject,queue_key=hashlib.sha256(('20261106:'+qid).encode()).hexdigest()))
 assert len(files)==7 and len(rows)>512
 rows=sorted(rows,key=lambda r:r['queue_key'])[:512]
 for r in rows:
  parsed,_,_=M.parse_answer(r['gold']);assert parsed,r['id']
 OUT.mkdir(exist_ok=True,parents=True)
 meta=dict(dataset='EleutherAI/hendrycks_math',revision=api['sha'],split='test',transport='hf-mirror.com exact revision',files=files,candidate_count=len(rows),candidate_sha=sha(rows),source=fingerprint(),dependencies_sha=dep_digest(),generation=CONFIG)
 if (OUT/'prepare.json').exists():assert jread(OUT/'prepare.json')==meta;assert jread(OUT/'candidates.json')==rows
 else:jwrite(OUT/'candidates.json',rows);jwrite(OUT/'prepare.json',meta)
 print('PREPARED',len(rows),'frozen MATH candidates',api['sha'])
def prepared():
 p=jread(OUT/'prepare.json');rows=jread(OUT/'candidates.json');assert p['source']==fingerprint();assert sha(rows)==p['candidate_sha'];assert p['dependencies_sha']==dep_digest();return p,rows

def load(name,stage):
 import torch,token_diagnostic_backend as B
 from transformers import AutoTokenizer
 prepared();assert torch.cuda.device_count()==1
 assert GPU in subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True)
 path=MODEL_ROOT/f'Qwen2.5-{name.upper()}-Instruct'
 import slot_model_loader as L
 scorer,meta=L.load_model(name,path,ROOT);tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
 meta.update(runtime=B.runtime(),generation=CONFIG,source=fingerprint(),dependencies_sha=dep_digest())
 target=OUT/'stages'/stage;target.mkdir(exist_ok=True,parents=True)
 if (target/'model.json').exists():assert jread(target/'model.json')==meta
 else:jwrite(target/'model.json',meta)
 eos=jread(path/'generation_config.json')['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
 return scorer,tok,eos,target

def gate(s,tok,requests,path):
 import torch
 checks=[]
 with torch.inference_mode():
  for req in requests:
   ids=req['input_ids'];assert len(ids)>1
   kv,h=s._feed(ids,len(ids)-1);del kv
   a=torch.log_softmax(s.model.lm_head(h[-1].float()).float(),-1);del h
   kv,_=s._feed(ids[:-1],len(ids)-2)
   z=s.model.model(input_ids=s._ids(ids[-1:]),past_key_values=kv,use_cache=True)
   b=torch.log_softmax(s.model.lm_head(z.last_hidden_state[0,-1].float()).float(),-1)
   tv=float((a.exp()-b.exp()).abs().sum()/2);assert torch.isfinite(a).all() and torch.isfinite(b).all()
   checks.append(dict(tokens=len(ids),tv=tv,passed=tv<.005,kind='chunked_vs_incremental_full_vocab'));del kv,z,a,b
 jwrite(path,checks);assert all(c['passed'] for c in checks)

def generate(s,tok,eos,req,rng_seed):
 import torch
 ids=req['input_ids'];assert len(ids)+CONFIG['max_new_tokens']<=32768
 start=time.time();output=[];lps=[];generator=torch.Generator(device=s.device).manual_seed(rng_seed)
 with torch.inference_mode():
  kv,h=s._feed(ids,len(ids)-1);last=h[-1];del h
  for _ in range(CONFIG['max_new_tokens']):
   logits=s.model.lm_head(last.float()).float();assert torch.isfinite(logits).all()
   lp=torch.log_softmax(logits,-1);p=torch.softmax(logits/CONFIG['temperature'],-1)
   y=int(torch.multinomial(p,1,generator=generator));output.append(y);lps.append(float(lp[y]))
   if y in eos:break
   z=s.model.model(input_ids=s._ids([y]),past_key_values=kv,use_cache=True);kv=z.past_key_values;last=z.last_hidden_state[0,-1];del z
  del kv,last
 text=tok.decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 return dict(request=req,seed=rng_seed,ids=output,text=text,logp=lps,seconds=time.time()-start,stop='eos' if output[-1] in eos else 'length',generation=CONFIG)

def student():
 _,rows=prepared();s,tok,eos,d=load('7b','student')
 requests=[M.prompt(tok,r['problem'],'INDEPENDENT') for r in rows]
 gate(s,tok,[min(requests,key=lambda x:len(x['input_ids'])),max(requests,key=lambda x:len(x['input_ids']))],d/'gate.json')
 for i,(r,req) in enumerate(zip(rows,requests)):
  out=OUT/'student'/(r['id']+'.json')
  if out.exists():continue
  g=generate(s,tok,eos,req,M.seed('student',r['id']));grade=M.grade(r['gold'],g['text']);proposal=M.propose(g['text'])
  eligible=grade['parseable'] and not grade['correct'] and g['stop']=='eos' and proposal is not None
  jwrite(out,dict(id=r['id'],generation=g,grade=grade,cut_proposal=proposal,eligible=eligible))
  status('student',completed=i+1,total=len(rows),seconds=g['seconds'])
  if i==7:wake('【数学学生生成测速】已生成8题，请读 '+str(OUT)+'/status.json 和 student/ 估时；固定队列512题，不改筛选标准。')

def screened(rows):
 selected=[]
 for r in rows:
  p=OUT/'screen'/(r['id']+'.json')
  if p.exists() and jread(p)['qualified']:selected.append(r)
 return selected

def screen(target):
 _,rows=prepared()
 if len(screened(rows))>=target:return
 candidates=[r for r in rows if jread(OUT/'student'/(r['id']+'.json'))['eligible']]
 pending=[r for r in candidates if not (OUT/'screen'/(r['id']+'.json')).exists()]
 if not pending:return
 s,tok,eos,d=load('32b','screen')
 requests=[M.prompt(tok,r['problem'],'INDEPENDENT') for r in pending]
 gate(s,tok,[min(requests,key=lambda x:len(x['input_ids'])),max(requests,key=lambda x:len(x['input_ids']))],d/('gate_'+str(target)+'.json'))
 for r,req in zip(pending,requests):
  runs=[]
  for i in range(4):
   p=OUT/'screen_generations'/r['id']/(str(i)+'.json')
   if p.exists():g=jread(p)
   else:
    out=generate(s,tok,eos,req,M.seed('screen',r['id'],i));gr=M.grade(r['gold'],out['text']);gr['correct']=gr['correct'] and out['stop']=='eos'
    g=dict(generation=out,grade=gr);jwrite(p,g)
   runs.append(g)
  count=sum(x['grade']['correct'] for x in runs)
  jwrite(OUT/'screen'/(r['id']+'.json'),dict(id=r['id'],correct_count=count,qualified=count>=3,records={str(i):C.sha_file(OUT/'screen_generations'/r['id']/(str(i)+'.json')) for i in range(4)}))
  count=len(screened(rows));status('teacher_screen',qualified=count,target=target,last_question=r['id'])
  if count>=target:break

def selected(stage,target):
 _,rows=prepared();chosen=screened(rows)[:target];return chosen

def request_review(stage,target):
 rows=selected(stage,target);req=[]
 for r in rows:
  student=jread(OUT/'student'/(r['id']+'.json'))
  req.append(dict(id=r['id'],problem=r['problem'],gold_for_offline_review_only=r['gold'],student=student,student_sha=C.sha_file(OUT/'student'/(r['id']+'.json'))))
 path=OUT/stage/'cut_review_request.json';request=dict(stage=stage,questions=req,selected_ids=[r['id'] for r in rows],created_before_evaluation=True)
 if path.exists():assert jread(path)==request
 else:jwrite(path,request)
 return request

def cut_annotations(stage,target):
 req=request_review(stage,target);p=OUT/stage/'cut_annotations.json'
 if not p.exists():return None
 a=jread(p);assert a['request_sha']==C.sha_file(OUT/stage/'cut_review_request.json');assert set(a['questions'])==set(req['selected_ids'])
 for r in req['questions']:M.validate_annotation(r['student']['generation']['text'],a['questions'][r['id']])
 return a

def evaluate(stage,target):
 a=cut_annotations(stage,target);assert a,'Need prospective cut review'
 rows=selected(stage,target);s,tok,eos,d=load('32b',stage)
 requests=[]
 for r in rows:
  st=jread(OUT/'student'/(r['id']+'.json'));prefix=M.validate_annotation(st['generation']['text'],a['questions'][r['id']])
  requests.extend([M.prompt(tok,r['problem'],arm,prefix) for arm in M.ARMS])
 if requests:gate(s,tok,[min(requests,key=lambda x:len(x['input_ids'])),max(requests,key=lambda x:len(x['input_ids']))],d/'gate.json')
 for i,r in enumerate(rows):
  st=jread(OUT/'student'/(r['id']+'.json'));prefix=M.validate_annotation(st['generation']['text'],a['questions'][r['id']]);annsha=C.sha_file(OUT/stage/'cut_annotations.json')
  for arm in M.ARMS:
   p=OUT/stage/'responses'/r['id']/(arm+'.json')
   if p.exists():assert jread(p)['annotation_sha']==annsha;continue
   req=M.prompt(tok,r['problem'],arm,prefix);g=generate(s,tok,eos,req,M.seed('eval_'+stage,r['id']))
   # ACTIVE outputs only suffix; complete answer may span the truncation point.
   complete=prefix+g['text'] if arm=='ACTIVE' else g['text']
   grade=M.grade(r['gold'],complete,st['generation']['text']);grade['correct']=grade['correct'] and g['stop']=='eos'
   jwrite(p,dict(id=r['id'],arm=arm,annotation_sha=annsha,generation=g,graded_text=complete,grade=grade))
  status('evaluation_'+stage,completed=i+1,total=len(rows))

def analyze(stage,target):
 a=cut_annotations(stage,target);assert a
 rows=selected(stage,target);report=[];audits=[]
 for r in rows:
  grades={}
  for arm in M.ARMS:
   p=OUT/stage/'responses'/r['id']/(arm+'.json');g=jread(p);grades[arm]=g['grade']
   audits.append(dict(id=r['id'],arm=arm,path=str(p),sha=C.sha_file(p),needs_review=g['grade']['format']!='boxed' or not g['grade']['parseable']))
  report.append(dict(id=r['id'],cohort=a['questions'][r['id']]['cohort'],grades=grades))
 review_path=OUT/stage/'outcome_review.json'
 if review_path.exists():
  review=jread(review_path);assert review['request_sha']==C.sha_file(OUT/stage/'outcome_review_request.json')
  for item in review['items']:
   match=next(x for x in audits if x['id']==item['id'] and x['arm']==item['arm']);assert item['response_sha']==match['sha'] and item['reason'] and item['reviewer']
   if 'correct' in item:
    row=next(x for x in report if x['id']==item['id']);row['grades'][item['arm']]=dict(row['grades'][item['arm']],correct=bool(item['correct']),retains_student_answer=bool(item['retains_student_answer']))
 else:
  for arm in M.ARMS:
   subset=sorted([x for x in audits if x['arm']==arm],key=lambda x:sha([x['id'],arm]))[:8]
   for x in subset:x['needs_review']=True
  jwrite(OUT/stage/'outcome_review_request.json',dict(records=audits,required=[dict(id=x['id'],arm=x['arm'],response_sha=x['sha']) for x in audits if x['needs_review']]))
 _,pool=prepared()
 student_rows=[jread(OUT/'student'/(x['id']+'.json')) for x in pool]
 screens=[jread(OUT/'screen'/(x['id']+'.json')) for x in pool if (OUT/'screen'/(x['id']+'.json')).exists()]
 timing={arm:sum(jread(OUT/stage/'responses'/r['id']/(arm+'.json'))['generation']['seconds'] for r in rows) for arm in M.ARMS}
 analysis=M.analysis(report);analysis.update(screening=dict(fixed_candidates=len(pool),student_eligible=sum(x['eligible'] for x in student_rows),student_correct=sum(x['grade']['correct'] for x in student_rows),student_unparseable=sum(not x['grade']['parseable'] for x in student_rows),student_truncated=sum(x['generation']['stop']=='length' for x in student_rows),student_no_cut=sum(x['cut_proposal'] is None for x in student_rows),teacher_screened=len(screens),teacher_qualified=sum(x['qualified'] for x in screens),requested=target,selected=len(rows)),generation_seconds_by_arm=timing,seconds_per_question=sum(timing.values())/len(rows) if rows else None,grade_issues={arm:dict(unparseable=sum(not jread(OUT/stage/'responses'/r['id']/(arm+'.json'))['grade']['parseable'] for r in rows),truncated=sum(jread(OUT/stage/'responses'/r['id']/(arm+'.json'))['generation']['stop']=='length' for r in rows)) for arm in M.ARMS});analysis.update(stage=stage,questions=len(rows),source=fingerprint(),records_sha=sha(report),pilot_only=stage=='first32',raw_results_hashes={x['path']:x['sha'] for x in audits},reviewed=review_path.exists())
 jwrite(OUT/stage/'aggregate_rows.json',report);jwrite(OUT/stage/'analysis.json',analysis)
 jwrite(OUT/stage/'REPORT.txt',json.dumps(analysis['cohorts'],ensure_ascii=False,indent=2)+'\n'+analysis['interpretation'])

def verify(stage,target):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(MODEL_ROOT/'Qwen2.5-7B-Instruct'),local_files_only=True)
 a=cut_annotations(stage,target);rows=selected(stage,target);analysis=jread(OUT/stage/'analysis.json');agg=jread(OUT/stage/'aggregate_rows.json')
 assert analysis['source']==fingerprint() and analysis['records_sha']==sha(agg) and analysis['reviewed']
 review=jread(OUT/stage/'outcome_review.json');request=jread(OUT/stage/'outcome_review_request.json')
 required={(x['id'],x['arm']) for x in request['required']};done={(x['id'],x['arm']) for x in review['items']};assert required<=done
 for r in rows:
  st=jread(OUT/'student'/(r['id']+'.json'));sc=jread(OUT/'screen'/(r['id']+'.json'));assert st['eligible'] and sc['correct_count']>=3
  prefix=M.validate_annotation(st['generation']['text'],a['questions'][r['id']])
  screening_seeds=[]
  for i in range(4):
   p=OUT/'screen_generations'/r['id']/(str(i)+'.json');g=jread(p);assert C.sha_file(p)==sc['records'][str(i)];screening_seeds.append(g['generation']['seed'])
   assert g['generation']['request']==M.prompt(tok,r['problem'],'INDEPENDENT')
  for arm in M.ARMS:
   p=OUT/stage/'responses'/r['id']/(arm+'.json');g=jread(p);assert C.sha_file(p)==analysis['raw_results_hashes'][str(p)]
   assert g['annotation_sha']==C.sha_file(OUT/stage/'cut_annotations.json')
   assert g['generation']['request']==M.prompt(tok,r['problem'],arm,prefix)
   assert g['generation']['seed']==M.seed('eval_'+stage,r['id']) and g['generation']['seed'] not in screening_seeds
   assert g['generation']['generation']==CONFIG and len(g['generation']['ids'])==len(g['generation']['logp'])
   again=M.grade(r['gold'],g['graded_text'],st['generation']['text']);again['correct']=again['correct'] and g['generation']['stop']=='eos';assert again==g['grade']
 computed=M.analysis(agg);assert computed['cohorts']==analysis['cohorts']
 # Independently compute primary paired point means; bootstrap function separately checked on CPU.
 for cohort in ('error_observed','error_unknown'):
  subset=[x for x in agg if x['cohort']==cohort]
  if subset:
   mean=100*sum(int(x['grades']['QUOTE-CONTINUE']['correct'])-int(x['grades']['ACTIVE']['correct']) for x in subset)/len(subset)
   assert abs(mean-analysis['cohorts'][cohort]['quote_continue_minus_active_pp']['correct']['mean'])<1e-10
 checks=list((OUT/'stages').glob('*/gate*.json'));assert checks and all(z['passed'] and z['tv']<.005 for p in checks for z in jread(p))
 jwrite(OUT/stage/'verification.json',dict(passed=True,questions=len(rows),analysis_sha=C.sha_file(OUT/stage/'analysis.json'),review_sha=C.sha_file(OUT/stage/'outcome_review.json'),annotations_sha=C.sha_file(OUT/stage/'cut_annotations.json'),checked_prompts_no_gold=True,fresh_eval_seeds=True,raw_grades_recomputed=True,source=fingerprint()))

def command(args):
 p=subprocess.run([sys.executable,'-u',str(Path(__file__)),*args],cwd=ROOT)
 assert p.returncode==0,'MATH stage failed: '+str(args)

def wait_file(path,message,validator):
 if not path.exists():status('awaiting_review',path=str(path));wake(message)
 while not path.exists():time.sleep(30)
 return validator()

def run():
 with (OUT/'queue.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name in ('7b','32b'):assert jread(HERE/'results/token_slot_20261106'/name/'full/verification.json')['passed']
  prepared();command(['student']);command(['screen','--target','32'])
  for stage in ('first32','full'):
   target=32
   if stage=='full':
    budget=wait_file(OUT/'full_budget.json','【数学pilot已核验】请根据实际吞吐、截断质量和固定候选队列冻结全量预算到 '+str(OUT)+'/full_budget.json；不得依据条件效果挑规模。',lambda:jread(OUT/'full_budget.json'))
    assert 1<=budget['target']<=200 and budget['basis']=='timing_and_cut_quality' and budget['reviewer']
    assert budget['pilot_verification_sha']==C.sha_file(OUT/'first32/verification.json');target=budget['target'];command(['screen','--target',str(target)])
   request_review(stage,target)
   wait_file(OUT/stage/'cut_annotations.json','【数学截断待核验】请读 '+str(OUT/stage/'cut_review_request.json')+'，在四臂生成前逐题检查错误是否已经发生，保存 cut_annotations.json（明确未知队列）；不要把最终答错直接当成前缀错误。',lambda:cut_annotations(stage,target))
   command(['evaluate','--stage',stage,'--target',str(target)]);command(['analyze','--stage',stage,'--target',str(target)])
   wait_file(OUT/stage/'outcome_review.json','【数学输出待核验】请检查 '+str(OUT/stage/'outcome_review_request.json')+' 中必需抽样/异常格式，记录 outcome_review.json，保留原始自动判断和响应。',lambda:jread(OUT/stage/'outcome_review.json'))
   command(['analyze','--stage',stage,'--target',str(target)]);command(['verify','--stage',stage,'--target',str(target)])
  status('completed_verified');wake('【数学槽位实验已核验】请结合实验1汇报并上传 research/hindsight-judges，不修改 main 或旧结果；新任务无自动关机授权。')

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','student','screen','evaluate','analyze','verify','run']);p.add_argument('--stage',choices=['first32','full'],default='first32');p.add_argument('--target',type=int,default=32);a=p.parse_args()
 if a.command in ('evaluate','analyze','verify'):globals()[a.command](a.stage,a.target)
 elif a.command=='screen':screen(a.target)
 else:globals()[a.command]()
if __name__=='__main__':main()
