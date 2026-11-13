"""v3 trainer: persistent Adam, actual behavior logprobs, fixed all-row batches."""
import argparse,contextlib,json,math,os,random,time
from pathlib import Path
import numpy as np
from util import *
from train import clipped_token_loss
from precision import selected_logprobs,check_environment
from action_mask import action_mask
from checkpoint_live import verify_live_checkpoint
from direction_policy import evaluate as evaluate_direction

def batches(n,size,seed):
 order=list(range(n));random.Random(seed).shuffle(order)
 return [order[i:i+size] for i in range(0,n,size)]
def k3(logp,ref):
 import torch
 d=ref-logp
 return torch.exp(d.clamp(-20,20))-d-1

def train(a):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel,LoraConfig,get_peft_model
 check_environment()
 plan=load(HERE/'PLAN.json');spec=plan['optimizer'];pg=plan['parity_gate'];sg=plan['stability']
 if a.out.exists():raise ValueError('Fresh output required; no silent checkpoint reuse')
 report=load(a.data.parent/'advantage_report.json');summary=load(a.summary);prov=summary['provenance']
 assert report['samples_sha256']==sha(a.data) and prov['split']=='train' and prov['schema']=='ctha-v3-fp32'
 assert prov['sampling_logprobs']=='processed_logprobs' and prov['temperature']==.7
 assert prov['adapter_sha256']==(sha(a.adapter/'adapter_model.safetensors') if a.adapter else None)
 rows=[json.loads(l) for l in a.data.read_text().splitlines()]
 assert rows and all(math.isfinite(r['advantage']) for r in rows)
 config=dict(schema='ctha-v3-fp32',data_sha=sha(a.data),source_sha=source_hashes(),adapter_sha=prov['adapter_sha256'],optimizer=spec,seed=a.seed,probe=a.probe)
 if a.adapter:config['optimizer_state_sha']=sha(a.adapter/'optimizer.pt')
 a.out.mkdir(parents=True);save(a.out/'config.json',config)
 torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
 data=[]
 for r in rows:
  pi=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,return_dict=False)
  assert pi==r['prompt_token_ids']
  ids=r['response_token_ids'];lp=r['behavior_logprobs']
  assert len(ids)==len(lp) and all(math.isfinite(v) for v in lp) and len(pi)+len(ids)<=6144
  data.append(dict(prompt_ids=pi,completion_ids=ids,old=torch.tensor(lp,dtype=torch.float32),advantage=r['advantage'],action_mask=None))
 if a.probe:
  # Deterministic balanced engineering subset; not a sampled population effect.
  pos=[];neg=[]
  for i,x in enumerate(data):
   bucket=pos if x['advantage']>0 else neg if x['advantage']<0 else None
   if bucket is None or len(bucket)>=4:continue
   x['action_mask']=action_mask(tok,x['completion_ids'])
   if x['action_mask'] is not None:bucket.append(i)
   if len(pos)==len(neg)==4:break
  assert len(pos)==len(neg)==4,'Probe requires four real positive and four real negative samples'
  chosen=pos+neg;data=[data[i] for i in chosen];save(a.out/'probe_indices.json',chosen)
 actor=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to('cuda')
 if a.adapter:model=PeftModel.from_pretrained(actor,str(a.adapter),is_trainable=True)
 else:model=get_peft_model(actor,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 params=[p for p in model.parameters() if p.requires_grad]
 for p in params:p.data=p.data.float()
 opt=torch.optim.AdamW(params,lr=spec['lr'],weight_decay=0.)
 total_updates=0
 if a.adapter:
  state=torch.load(a.adapter/'optimizer.pt',map_location='cpu',weights_only=True)
  assert state['adapter_sha']==prov['adapter_sha256']
  opt.load_state_dict(state['optimizer']);total_updates=state['total_updates']
  assert all(v['step'].item()==total_updates for v in opt.state.values())
 prior_updates=total_updates
 model.eval();errors=[];before={};started=time.time()
 # Numeric agreement measured before any updates on fixed evenly spaced samples.
 parity_ids=sorted(set(np.linspace(0,len(data)-1,min(32,len(data)),dtype=int).tolist()))
 if not a.probe:
  for i in parity_ids:data[i]['action_mask']=action_mask(tok,data[i]['completion_ids'])
 save(a.out/'action_mask_coverage.json',{'diagnostic_rows':len(parity_ids),'mapped_rows':sum(data[i]['action_mask'] is not None for i in parity_ids),'all_training_rows':len(data),'mask_used_for_policy_loss':False})
 with torch.no_grad():
  for i in parity_ids:
   x=data[i];lp=selected_logprobs(model,x,.7).cpu();before[i]=lp
   errors.extend((lp-x['old']).abs().tolist())
 err=np.asarray(errors);parity={'mean_abs':float(err.mean()),'p99_abs':float(np.quantile(err,.99)),'max_abs':float(err.max()),'samples':len(parity_ids),'tokens':len(err)}
 parity['passed']=parity['mean_abs']<=pg['mean_absolute_sampled_logprob_error_max'] and parity['p99_abs']<=pg['p99_absolute_sampled_logprob_error_max'] and parity['max_abs']<=pg['max_absolute_sampled_logprob_error_max']
 save(a.out/'parity.json',parity);print('PARITY',parity,flush=True)
 if not parity['passed']:raise RuntimeError('Cross-engine parity failed before optimization')
 with torch.no_grad(),model.disable_adapter():
  for i,x in enumerate(data):
   x['ref']=selected_logprobs(model,x,.7).cpu()
   if (i+1)%100==0:print('REFERENCE',i+1,len(data),flush=True)
 model.train();logs=[];epochs=4 if a.probe else spec['epochs'];batch_size=len(data) if a.probe else spec['accum']
 for epoch in range(epochs):
  for block in batches(len(data),batch_size,a.seed+epoch):
   opt.zero_grad(set_to_none=True);losses=[];clips=[];drifts=[];refs=[]
   for i in block:
    x=data[i];lp=selected_logprobs(model,x,.7);old=x['old'].to(lp.device);ref=x['ref'].to(lp.device)
    policy=clipped_token_loss(lp,old,x['advantage'],[True]*len(lp),spec['clip'])
    penalty=k3(lp,ref).mean();loss=policy+spec['kl_coef']*penalty
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
    (loss/len(block)).backward()
    with torch.no_grad():
     ratio=torch.exp((lp-old).clamp(-20,20));clips.append(float(((ratio<.8)|(ratio>1.2)).float().mean()))
     drifts.append(float(k3(lp,old).mean()));refs.append(float(penalty));losses.append(float(loss))
   metrics={'epoch':epoch,'batch_rows':len(block),'loss':float(np.mean(losses)),'clip_fraction':float(np.mean(clips)),'sampled_behavior_k3':float(np.mean(drifts)),'sampled_reference_k3':float(np.mean(refs))}
   if metrics['sampled_behavior_k3']>sg['sampled_behavior_k3_stop'] or metrics['clip_fraction']>sg['clip_fraction_stop']:
    save(a.out/'stability_stop.json',metrics);raise RuntimeError('Pre-optimizer stability gate exceeded; no threshold relaxation')
   norm=torch.nn.utils.clip_grad_norm_(params,spec['grad_clip'])
   if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient norm')
   opt.step();total_updates+=1;metrics.update(grad_norm=float(norm),total_updates=total_updates)
   logs.append(metrics);print(json.dumps(metrics),flush=True)
   save(a.out/'progress.json',{'updates':total_updates-prior_updates,'last':metrics,'elapsed_s':time.time()-started})
 model.eval();direction=[];after_errors=[];final_clips=[];final_scoring=[]
 with torch.no_grad():
  for i in parity_ids:
   x=data[i];lp=selected_logprobs(model,x,.7).cpu();delta=lp-before[i];mask=torch.tensor(x['action_mask'],dtype=torch.bool) if x['action_mask'] is not None else None
   final_scoring.append(dict(index=i,logprobs=lp.tolist(),behavior=x['old'].tolist(),reference=x['ref'].tolist()))
   direction.append({'index':i,'advantage':x['advantage'],'delta_mean_response_logprob':float(delta.mean()),'delta_mean_action_logprob':float(delta[mask].mean()) if mask is not None else None,'tokens':len(lp)})
   after_errors.extend(k3(lp,x['old']).tolist());ratio=(lp-x['old']).exp();final_clips.extend(((ratio<.8)|(ratio>1.2)).float().tolist())
 final={'sampled_behavior_k3':float(np.mean(after_errors)),'clip_fraction':float(np.mean(final_clips))}
 save(a.out/'final_scoring.json',final_scoring)
 save(a.out/'post_update.json',{'direction':direction,'stability':final})
 if final['sampled_behavior_k3']>sg['sampled_behavior_k3_stop'] or final['clip_fraction']>sg['clip_fraction_stop']:raise RuntimeError('Post-update stability gate exceeded; no next rollout')
 if a.probe:
  pos=[r for r in direction if r['advantage']>0];neg=[r for r in direction if r['advantage']<0]
  probe={f'{label}_{field}':float(np.mean([r[field] for r in rr])) for label,rr in [('positive',pos),('negative',neg)] for field in ['delta_mean_response_logprob','delta_mean_action_logprob']}
  probe=evaluate_direction(probe)
  save(a.out/'direction_check.json',probe)
  if not probe['passed']:raise RuntimeError('Positive/negative update direction failed; no automatic expansion')
 model.save_pretrained(a.out);weight_sha=sha(a.out/'adapter_model.safetensors')
 torch.save({'optimizer':opt.state_dict(),'adapter_sha':weight_sha,'total_updates':total_updates},a.out/'optimizer.pt')
 verify_live_checkpoint(model,opt,total_updates,a.out)
 save(a.out/'train_log.json',{'complete':True,'updates_this_round':total_updates-prior_updates,'total_updates':total_updates,'optimizer_resumed':bool(a.adapter),'weights_sha':weight_sha,'optimizer_sha':sha(a.out/'optimizer.pt'),'n_source_steps':len(rows),'n_trained_rows':len(data),'nonzero_advantage_rows':sum(x['advantage']!=0 for x in data),'logs':logs,'parity':parity,'elapsed_s':time.time()-started})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--adapter',type=Path);p.add_argument('--seed',type=int,default=0);p.add_argument('--probe',action='store_true');a=p.parse_args();train(a)
