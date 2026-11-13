"""Independent CPU checks of evidence; never infer learning from engineering."""
import argparse,json,hashlib,math
from pathlib import Path

def read(p):return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def verify(d):
 reg=read(d/'registration.json');plan=reg['parent_plan'];pg=plan['parity_gate'];sg=plan['stability'];checks={}
 assert read(d/'cpu_validation.json')['source_sha']==reg['source_sha']
 for directory,count in [('repeat_a',8),('repeat_b',8),('initial_train',64)]:
  files=sorted((d/directory).glob('s*_r*.json'));summary=read(d/directory/'summary.json')
  assert len(files)==count==summary['n_episodes']
  for f in files:
   e=read(f);assert e['provenance']['fingerprint']==summary['provenance']['fingerprint']
   assert e['n_turns']==len(e['steps']) and 0<len(e['steps'])<=50
   for s in e['steps']:
    assert len(s['response_token_ids'])==len(s['behavior_logprobs'])>0
    assert all(math.isfinite(x) for x in s['behavior_logprobs'])
  checks[directory+'_complete']=True
 fields=['response_token_ids','prompt_token_ids','sample_seed','action','obs_before','obs_after','won_after']
 for other in ['repeat_b','initial_train']:
  for f in (d/'repeat_a').glob('s*_r*.json'):
   a=read(f);b=read(d/other/f.name);assert a['won']==b['won'] and len(a['steps'])==len(b['steps'])
   assert all(all(x[k]==y[k] for k in fields) for x,y in zip(a['steps'],b['steps']))
  checks['exact_comparison_'+other]=True
 for label,path in [('base_parity',d/'direction_probe/parity.json'),('updated_adapter_parity',d/'adapter_parity/verification.json')]:
  p=read(path);assert p['passed'] and p['tokens']>0
  assert p['mean_abs']<=pg['mean_absolute_sampled_logprob_error_max'] and p['p99_abs']<=pg['p99_absolute_sampled_logprob_error_max'] and p['max_abs']<=pg['max_absolute_sampled_logprob_error_max']
  checks[label]=True
 t=read(d/'direction_probe/train_log.json');assert t['complete'] and t['updates_this_round']==t['total_updates']==4 and t['n_trained_rows']==8
 assert digest(d/'direction_probe/adapter_model.safetensors')==t['weights_sha']
 assert digest(d/'direction_probe/optimizer.pt')==t['optimizer_sha']
 post=read(d/'direction_probe/post_update.json');direction=read(d/'direction_probe/direction_check.json');assert direction['passed']
 for field in ['delta_mean_response_logprob','delta_mean_action_logprob']:
  pos=[r[field] for r in post['direction'] if r['advantage']>0];neg=[r[field] for r in post['direction'] if r['advantage']<0]
  assert len(pos)==len(neg)==4 and sum(pos)/4>0 and sum(neg)/4<0
 for m in [*t['logs'],post['stability']]:
  assert m['sampled_behavior_k3']<=sg['sampled_behavior_k3_stop'] and m['clip_fraction']<=sg['clip_fraction_stop']
 checks['direction_and_stability']=True
 c=read(d/'checkpoint_verification.json');assert c['passed'] and c['moments_equal'] and c['next_step_equal'] and c['restored_updates']==4 and c['next_test_updates']==5
 assert c['optimizer_sha256']==t['optimizer_sha'] and c['weights_sha256']==t['weights_sha']
 checks['actual_checkpoint_restore']=True
 from safetensors.torch import load_file
 weights=load_file(str(d/'direction_probe/adapter_model.safetensors'))
 assert any('lora_B' in k and bool(v.count_nonzero()) for k,v in weights.items());checks['nonzero_lora_B']=True
 assert not any((d/k).exists() for k in ['baseline_tune','grpo','gigpo'])
 result={'passed':True,'checks':checks,'engineering':'PASS','learning':'NOT_RUN','generalization':'NOT_RUN','ctha_online_training':'NOT_RUN','online_second_round_resume':'NOT_TESTED','checkpoint_restore_scope':'Actual probe checkpoint CPU reload, moments/counters and deterministic next-step equality; updated policy GPU parity tested separately','weights_sha256':t['weights_sha'],'optimizer_sha256':t['optimizer_sha']}
 (d/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
 (d/'REPORT.txt').write_text('CTHA v3 ENGINEERING ACCEPTANCE ONLY\nEngineering: PASS\nLearning/generalization: NOT RUN\nNo full P1/P2; no A100 validation.\n\nPassed frozen-source CPU checks; 8 repeated trajectories and cross-batch matching against 64 trajectories; original base and updated-adapter probability gates; balanced positive/negative update direction; original stability gates; nonzero LoRA B; actual checkpoint CPU reload and next-step equality.\n\nOnly the four-step eight-response engineering probe updated model parameters. Optimizer reload equality used fixed engineering gradients on CPU copies; an online second training round was not tested. No policy improvement or CTHA efficacy claim follows from this report.\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();print(json.dumps(verify(a.data),indent=2))
