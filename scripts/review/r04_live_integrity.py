import os,sys,json,glob,hashlib,collections,time
from pathlib import Path
sys.path.insert(0,'scripts');import common
root=Path.cwd();hashes={};errors=[];summary={};base={};trs={};conts=seeds=0

def digest(p):
 p=str(p)
 if p not in hashes:hashes[p]=common.file_sha256(p)
 return hashes[p]
def check(cond,msg):
 if not cond:raise ValueError(msg)
for p in sorted(glob.glob('data/rollout/*.json')):
 tr=json.load(open(p));trs[tr['traj_id']]=tr
 tasks={t['seed']:t for t in json.load(open('data/task_list.json'))['tasks']}
 check(tr['task_id']==tasks[tr['seed']]['task_id'],'task list mismatch')
 check(len(tr['steps'])==tr['n_turns']<=50,'rollout length')
 for i,s in enumerate(tr['steps']):
  check(s['turn']==i+1,'rollout turn')
  check((root/s['prompt_path']).is_file(),'missing prompt')
  if i:check(s['obs_before']==tr['steps'][i-1]['obs_after'],'broken trajectory observation chain')
for folder in sorted(glob.glob('data/replay*')):
 bank=0 if folder=='data/replay' else int(folder.rsplit('b',1)[1]);paths=sorted(glob.glob(folder+'/*/*.json'));seen=set();checked=0;divs=0
 for p in paths:
  try:
   r=json.load(open(p));tr=trs[r['traj_id']];turn=r['turn'];s=tr['steps'][turn-1];key=(r['traj_id'],turn)
   check((r['seed'],r['task_id'],r['task_type'])==(tr['seed'],tr['task_id'],tr['task_type']),'record identity')
   check(r['factual_action']==s['action'],'factual mismatch')
   alts=r['alternatives'];defined=r['cf_defined'];check(len(alts)==len(set(alts))<=4,'alternative uniqueness')
   check(all(a in s['admissible'] and a!=s['action'] for a in alts),'alternative support')
   check(defined==(len(alts)==4),'defined flag')
   check(r['exclude_reason']==(None if defined else 'lt4_alternatives'),'exclusion reason')
   source_path='data/rollout/'+tr['traj_id']+'.json'
   check(r['provenance']['inputs'][source_path]==digest(source_path),'rollout source hash')
   if bank==0:
    check(len(r['drafts'])==r['n_draws'] and 32<=r['n_draws']<=300,'draft count')
    check(defined or r['n_draws']==300,'early exclusion')
    check(set(alts)<={d['action'] for d in r['drafts'] if d['valid'] and d['admissible']},'draft support')
    base[key]=r
   else:
    b=base[key];check(b['cf_defined'],'bank contains excluded point')
    check(alts==b['alternatives'],'alternatives changed between banks')
    src=r['alternatives_source'];check(src['sha256']==digest(src['path']),'alternative source hash')
    check(src['path']==f'data/replay/{tr["traj_id"]}/t{turn:02d}.json','wrong alternative source')
    check(r['n_draws']==0 and r['drafts']==[],'unexpected resampling')
    check(r['provenance']['bank']==bank and r['provenance']['seed_offset']==bank*10**11,'bank offset')
   roles=list(range(1,6)) if defined else [1]
   cs=r['continuations'];check(len(cs)==len(roles)*3,'replica count')
   check({(c['role'],c['rep']) for c in cs}=={(role,rep) for role in roles for rep in range(3)},'replica identities')
   check(len(r['factual_outcomes'])==3 and len(r['alt_outcomes'])==len(alts),'outcome shape')
   check(all(len(a)==(3 if defined else 0) for a in r['alt_outcomes']),'alternative outcome shape')
   for c in cs:
    n=len(c['actions']);check(n>=1 and len(c['valids'])==n and c['len']==turn-1+n<=50,'step cap/length')
    check(c['won'] or c['len']==50,'early unfinished failure')
    first=s['action'] if c['role']==1 else alts[c['role']-2]
    check(c['first_action']==c['actions'][0]==first,'first action')
    outs=r['factual_outcomes'] if c['role']==1 else r['alt_outcomes'][c['role']-2]
    check(all(v in (0,1) for v in outs) and outs[c['rep']]==int(c['won']),'outcome mismatch')
    check(c['refeed_div']==0,'refeed divergence');divs+=c['refeed_div'];conts+=1
    if 'seeds' in c:
     expected=[((r['seed']*100+turn)*10+c['role'])*100000+c['rep']*100+k+bank*10**11 for k in range(turn,turn+n-1)]
     check(c['seeds']==expected,'request seed formula')
     for sd in c['seeds']:
      check(sd not in seen,'duplicate request seed');seen.add(sd)
     seeds+=len(expected)
   checked+=1
  except Exception as e:errors.append({'path':p,'error':str(e)})
 summary[folder]={'files_snapshot':len(paths),'passed':checked,'refeed_divergences':divs}
expected_base={(t['traj_id'],s['turn']) for t in trs.values() for s in t['steps']}
check(set(base)==expected_base,'bank 1 missing or extra points')
expected_D={k for k,r in base.items() if r['cf_defined']}
b2={}
for p in glob.glob('data/replay_b1/*/*.json'):
 r=json.load(open(p));b2[(r['traj_id'],r['turn'])]=r
check(set(b2)==expected_D,'bank 2 missing or extra points')
live=set()
for k in expected_D:
 outs=[]
 for r in (base[k],b2[k]):outs+=r['factual_outcomes']+[x for a in r['alt_outcomes'] for x in a]
 if 0<sum(outs)<len(outs):live.add(k)
for folder in summary:
 if folder in ('data/replay','data/replay_b1'):continue
 for p in glob.glob(folder+'/*/*.json'):
  r=json.load(open(p));check((r['traj_id'],r['turn']) in live,'non-live point included')
report={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'trajectories':len(trs),'total_points':len(base),'defined':len(expected_D),'live_target':len(live),'banks':summary,'continuations_checked':conts,'request_seeds_checked':seeds,'errors':errors,'note':'Snapshot of complete JSON files; active bank still in progress. No feature/target associations computed.'}
print(json.dumps(report,indent=2));common.write_json('eval/phase1c_live_integrity.json',report)
check(not errors,'record validation errors')
