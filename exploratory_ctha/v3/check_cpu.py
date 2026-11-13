"""Regression checks and train-only cold-start feasibility, no GPU execution."""
import collections,contextlib,json,math,subprocess,tempfile
from unittest.mock import patch
import torch,numpy as np
from util import *
from trainer import batches,k3
from action_mask import action_mask
from train import clipped_token_loss,selected_logprobs
from habits import load_episodes
from habits_v2 import make_batch,oof_audit

def checks():
 class CharacterTokenizer:
  def decode(self,ids,**kwargs):return ''.join(map(chr,ids))
 tok=CharacterTokenizer()
 for text in ['<think>reason</think><action>go to table 1</action>','<think>reason</think>[action] go to table 1 [/action]']:
  ids=list(map(ord,text));mask=action_mask(tok,ids)
  assert ''.join(chr(t) for t,m in zip(ids,mask) if m)=='go to table 1'
 assert action_mask(tok,list(map(ord,'no parseable action'))) is None
 # Actual batch planner retains zero-advantage positions and a fixed denominator.
 blocks=batches(131,64,0);assert sorted(sum(blocks,[]))==list(range(131));assert list(map(len,blocks))==[64,64,3]
 theta=torch.tensor(0.,requires_grad=True);old=torch.zeros(1)
 losses=[clipped_token_loss(theta.reshape(1),old,a,[True],.2) for a in [1.]+[0.]*7]
 (sum(losses)/len(losses)).backward();assert theta.grad.item()==-.125
 x=torch.nn.Parameter(torch.tensor(0.));opt=torch.optim.AdamW([x],lr=.01,weight_decay=0.)
 for g in [1.,-2.,3.]:opt.zero_grad();x.grad=torch.tensor(g);opt.step()
 with tempfile.TemporaryDirectory() as tmp:
  f=Path(tmp)/'state.pt';torch.save({'parameter':x.detach().clone(),'optimizer':opt.state_dict()},f)
  state=torch.load(f,weights_only=True);y=torch.nn.Parameter(state['parameter']);other=torch.optim.AdamW([y],lr=.01,weight_decay=0.);other.load_state_dict(state['optimizer'])
  for p,o in [(x,opt),(y,other)]:o.zero_grad();p.grad=torch.tensor(-1.);o.step()
  assert torch.equal(x,y)
  assert all(torch.equal(opt.state[x][key],other.state[y][key]) for key in ['step','exp_avg','exp_avg_sq'])
 for advantage,ratio,want in [(1,1,-1),(-1,1,1),(1,1.3,0),(-1,.7,0)]:
  p=torch.tensor(math.log(ratio),requires_grad=True);loss=clipped_token_loss(p.reshape(1),old,advantage,[True],.2);loss.backward();assert np.sign(p.grad.item())==want
 assert torch.allclose(k3(torch.tensor([-.5]),torch.tensor([-.5])),torch.zeros(1))
 return {'action_tags_and_fail_closed_mask':True,'batch_population_retained':True,'zero_rows_denominator':True,'optimizer_resume_exact':True,'surrogate_direction_clipping':True,'reference_penalty_zero_at_reference':True}

if __name__=='__main__':
 from pathlib import Path
 DATA.mkdir(exist_ok=True)
 results=checks();pool=[]
 for i,d in enumerate(sorted(Path('/root/autodl-tmp/ctha_v2/p0_base').glob('round_*'))):
  pool+=load_episodes(d/'rollouts',-1000+i)
 train_games={x['game_file'] for x in load(ROOT/'exploratory_E4/games.json')['train']}
 assert len(pool)==768 and all(e['game_file'] in train_games and e['provenance']['adapter_sha256'] is None for e in pool)
 gate=oof_audit(pool,family='habit',min_mixed=20,min_pairs=200,permutations=12,bootstraps=300)
 cur=load_episodes(Path('/root/autodl-tmp/ctha_v2_tokenfix/ctha_s0/round_000/rollouts'),0)
 rows,report=make_batch(cur,pool,mode='ctha',base='gigpo',family='habit',min_mixed=20,min_pairs=200,permutations=12,bootstraps=300,feature_window=8,train_horizon=0,aux_horizon=8,sign_fraction=.75)
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(PARENT),'-p','test_ctha*.py','-q'],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (DATA/'legacy_tests.log').write_text(p.stdout);assert p.returncode==0
 out={'checks':results,'legacy_tests_passed':True,'cold_start_gate':gate,'offline_nonzero_aux_samples':report['nonzero_aux_samples'],'offline_combined_pool_gate':report['gate'],'source_sha':source_hashes(),'interpretation':'Known P0 data reused for engineering feasibility, not independent evidence or a policy gain.'}
 save(DATA/'cpu_validation.json',out);print(json.dumps(out,indent=2))
 if not gate['passed'] or report['nonzero_aux_samples']==0:raise RuntimeError('No actual cold-start signal; do not run CTHA training')
