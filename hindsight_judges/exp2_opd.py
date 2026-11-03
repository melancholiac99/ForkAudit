"""Experiment 2: guarded privileged-context on-policy distillation on ALFWorld.

Three matched arms: M (no future), L_guarded (future with other-turn rationales),
LN_guarded (future with ALL rationales stripped). Crucial: for BOTH privileged
arms, current target-turn <think> is absent from the *extra* future copy.
It remains in the Student response and is trained, along with action tokens.

This is forward KL approximated with a normalized top-k teacher distribution
at Student-visited prefixes, NOT action-credit weighting or reverse KL.
One iteration uses trajectories sampled from its starting Student (one-policy
round); for further on-policy rounds collect new rollouts using the new adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from exp2_common import ROOT, HERE, read_json, save_json, decode_rows, encode_rows, sha_file, sha_bytes

CONDITIONS = ('M', 'L_guarded', 'LN_guarded')
THINK = re.compile(r'<think>.*?</think>\s*', re.I | re.S)
MAX_CONTEXT = 32768
TARGETS = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']


def context_log(ep, t, mode):
    """Condition changes written CoT only; never changes actions, observations or outcomes."""
    assert mode in ('L_guarded','LN_guarded') and 1 <= t <= len(ep['steps'])
    rows = []
    for j, st in enumerate(ep['steps'], 1):
        response = st['response']
        if mode == 'LN_guarded' or j == t:
            response = THINK.sub('', response)
        rows.append(dict(turn=j, response=response, action=st['action'],
                         obs_after=st['obs_after'], won_after=bool(st['won_after'])))
    return json.dumps(dict(task=ep['task'], initial_obs=ep['initial_obs'],
                           steps=rows, final_reward=int(ep['won'])), ensure_ascii=False)


def teacher_user(row, condition):
    assert condition in CONDITIONS
    # Fixed instruction across conditions. Student sees only row['prompt'].
    common = ('You are the ALFWorld agent at the marked decision. Produce the next assistant '
              'response in the action format requested by the task. Additional future '
              'evidence can help correct the action, but is not text to quote or copy.\n'
              '[DECISION-TIME PROMPT]\n' + row['prompt'] + '\n[/DECISION-TIME PROMPT]\n')
    if condition == 'M':
        return common + '[FUTURE EVIDENCE]\nUnavailable.\n[/FUTURE EVIDENCE]'
    return common + '[FUTURE EVIDENCE]\n' + row['future'][condition] + '\n[/FUTURE EVIDENCE]'


def token_types(text, offsets):
    """Every token receives a diagnostic type; NONE are masked during training."""
    spans = [(m.start(),m.end()) for m in THINK.finditer(text)]
    types=[]
    for a,b in offsets:
        types.append('think' if any(a < hi and b > lo for lo,hi in spans) else 'other')
    return types



def rollout(args):
    """Collect fresh E4-style full TRAIN episodes for the current Student adapter."""
    import sys
    import importlib
    sys.path.insert(0, str(Path(args.repo)/'scripts'))
    sys.path.insert(0, str(Path(args.repo)/'exploratory_E4'))
    import common
    import alfenv
    from envpool import EnvPool
    from e4_policy import LoraPolicy
    e4_roll = importlib.import_module('e4_01_rollout')
    games = read_json(Path(args.repo)/'exploratory_E4/games.json')['train']
    if args.max_games > 0:
        games = games[:args.max_games]
    assert games and args.rollout_out, 'Training games and --rollout-out are required'
    out = Path(args.rollout_out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # EnvPool must fork before CUDA initialization, following E4.
    pool = EnvPool(args.workers)
    policy = None
    try:
        policy = LoraPolicy(model_path=args.student)
        policy.set_adapter(args.base_adapter, 1)
        prov = dict(script='hindsight_judges/exp2_opd.py',
            adapter=args.base_adapter,
            adapter_sha256=(sha_file(Path(args.base_adapter)/'adapter_model.safetensors')
                            if args.base_adapter else None),
            games_sha256=sha_file(Path(args.repo)/'exploratory_E4/games.json'))
        chunk = max(1, args.rollout_chunk // args.reps)
        for start in range(0, len(games), chunk):
            e4_roll.run(games[start:start+chunk], args.reps, policy, pool, str(out),
                        True, alfenv, prov)
        files = [out/f"s{g['seed']}_r{r}.json" for g in games for r in range(args.reps)]
        episodes = [read_json(p) for p in files]
        save_json(out/'summary.json', dict(n=len(episodes),
            success=sum(bool(e['won']) for e in episodes)/len(episodes),
            adapter=args.base_adapter,provenance=prov))
    finally:
        if policy is not None:
            policy.close()
        pool.shutdown()
    print('ROLLOUT COMPLETE',out)


def prepare(args):
    roll = Path(args.rollouts).resolve()
    files = sorted(p for p in roll.glob('*.json') if p.name != 'summary.json')
    assert files, 'No full rollout JSON found'
    split = read_json(Path(args.repo)/'exploratory_E4/games.json')
    train = {g['game_file'] for g in split['train']}
    heldout = {g['game_file'] for key in ('tune','valid_seen','valid_unseen') for g in split[key]}
    assert not train & heldout, 'Training/eval games overlap'
    rows=[]; sources={}
    for p in files:
        ep=read_json(p)
        assert ep['game_file'] in train, 'Experiment 2 requires E4 TRAIN games only'
        assert ep['steps'] and all(all(k in st for k in ('prompt','response','action','obs_after','won_after'))
                                   for st in ep['steps']), 'Rollouts must be collected with --full'
        if args.adapter_provenance is not None:
            assert ep['provenance']['adapter'] == args.adapter_provenance, \
                'Adapter provenance differs from requested policy'
        eid=f"s{ep['seed']}_r{ep['rep']}"
        sources[eid]=sha_file(p)
        for i,st in enumerate(ep['steps'], 1):
            if not st['response'].strip():continue
            row=dict(id=f'{eid}_t{i:02d}',episode=eid,turn=i,
                     prompt=st['prompt'],response=st['response'],
                     game_file=ep['game_file'],won=bool(ep['won']),
                     future={c:context_log(ep,i,c) for c in CONDITIONS if c!='M'})
            row['source_sha256']=sources[eid]
            rows.append(row)
    # Deterministic selection independent of condition, output / success and model gradients.
    assert len({r['id'] for r in rows}) == len(rows)
    rows.sort(key=lambda r: hashlib.sha256((str(args.seed)+':'+r['id']).encode()).hexdigest())
    if args.max_examples > 0:rows=rows[:args.max_examples]
    rows.sort(key=lambda r:r['id'])
    assert rows
    out=Path(args.out).resolve()
    file=out/'examples.jsonl.gz'
    if file.exists():
        old,_=decode_rows(file)
        assert rows==old,'Existing dataset differs. Use a new round/output directory.'
    else:
        encode_rows(file,rows)
    man=dict(n=len(rows),seed=args.seed,max_examples=args.max_examples,
             examples_sha256=decode_rows(file)[1],rollout_sha256=sources,
             rollout_dir=str(roll),student_adapter=args.adapter_provenance,
             script_sha256=sha_file(Path(__file__)),conditions=list(CONDITIONS))
    if (out/'manifest.json').exists():
        old=read_json(out/'manifest.json')
        assert old==man,'Manifest mismatch; refusing to mix rounds'
    save_json(out/'manifest.json',man)
    print('PREPARED',len(rows),'samples across',len({r['episode'] for r in rows}),'episodes')


def prepared(out):
    out=Path(out)
    man=read_json(out/'manifest.json')
    rows,h=decode_rows(out/'examples.jsonl.gz')
    assert h==man['examples_sha256'] and len(rows)==man['n']
    assert len({r['id'] for r in rows}) == len(rows)
    return rows, man


def score_teacher(args):
    import torch
    from transformers import AutoTokenizer
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from exp2_common import load_32b
    assert torch.cuda.is_available()
    rows,man=prepared(args.out)
    tok=AutoTokenizer.from_pretrained(args.teacher,local_files_only=True,use_fast=True)
    assert tok.is_fast,'Fast tokenizer required to audit think-token offsets'
    from transformers import AutoTokenizer as AT
    student_tok=AT.from_pretrained(args.student,local_files_only=True)
    target=Path(args.out)/'teacher'/args.condition
    target.mkdir(parents=True,exist_ok=True)
    score_path=target/'scores.jsonl'
    run_path=target/'run.json'
    jobs=[]
    for r in rows:
        resp=tok(r['response'],add_special_tokens=False,return_offsets_mapping=True)
        ids=resp['input_ids']
        assert ids and ids==student_tok.encode(r['response'],add_special_tokens=False), 'Tokenizer vocab/segmentation differs'
        pre=tok.apply_chat_template([dict(role='user',content=teacher_user(r,args.condition))],
                                    tokenize=True,add_generation_prompt=True)
        assert len(pre)+len(ids) <= MAX_CONTEXT,'32B context too long; no silent truncation'
        jobs.append((r,pre,ids,token_types(r['response'],resp['offset_mapping'])))
    done={}
    if score_path.exists():
        for line in score_path.read_text().splitlines():
            z=json.loads(line)
            assert z['id'] not in done
            done[z['id']]=z
    assert set(done)<={r['id'] for r in rows}
    run_base=dict(condition=args.condition,n=len(rows),examples_sha256=man['examples_sha256'],
                  topk=args.topk,script_sha256=sha_file(Path(__file__)),
                  student_tokenizer=str(Path(args.student).resolve()))
    if run_path.exists():
        old=read_json(run_path)
        assert all(old[k]==v for k,v in run_base.items())
        if old.get('complete'):
            assert len(done)==len(rows) and sha_file(score_path)==old['scores_sha256']
            print('ALREADY COMPLETE');return
    else:
        assert not score_path.exists(), 'score file without run metadata'
    with sdpa_kernel(SDPBackend.MATH):
        scorer, model_meta=load_32b(Path(args.teacher),Path(args.repo))
        if run_path.exists():
            assert old['teacher_weights']==model_meta['weights']
        meta={**run_base,'teacher_weights':model_meta['weights'],
              'precision':model_meta['precision'],'complete':False}
        save_json(run_path,meta)
        with score_path.open('a',buffering=1) as f:
            for ix,(r,pre,ids,kinds) in enumerate(jobs):
                if r['id'] in done:
                    assert done[r['id']]['response_sha256']==sha_bytes(r['response'].encode())
                    continue
                with torch.inference_mode():
                    _,h=scorer._feed(pre+ids[:-1],len(pre)-1)
                    assert h.shape[0]==len(ids),'Teacher token positions misaligned'
                    logits=scorer.model.lm_head(h).float()
                    lp=torch.log_softmax(logits,dim=-1)
                    k=min(args.topk,lp.shape[-1])
                    val,idx=lp.topk(k,dim=-1)
                    p=val.exp()
                    target_lp=lp.gather(-1,torch.tensor(ids,device=lp.device)[:,None]).squeeze(-1)
                    mass=p.sum(-1)
                    assert torch.isfinite(mass).all()
                    z=dict(id=r['id'],condition=args.condition,
                           response_sha256=sha_bytes(r['response'].encode()),
                           target_token_ids=ids,token_types=kinds,
                           top_ids=idx.cpu().tolist(),top_probs=p.cpu().tolist(),
                           target_logp=target_lp.cpu().tolist(),top_mass=mass.cpu().tolist())
                    del h,logits,lp,p,val,idx,target_lp,mass
                f.write(json.dumps(z,allow_nan=False)+'\n')
                done[r['id']]=z
                if (ix+1)%10==0:print('TEACHER',args.condition,ix+1,'/',len(jobs),flush=True)
        assert len(done)==len(jobs)
        save_json(run_path,{**meta,'complete':True,'scores_sha256':sha_file(score_path),
                            'mean_top_mass':float(sum(sum(z['top_mass']) for z in done.values())/
                                                  sum(len(z['top_mass']) for z in done.values()))})
        print('SCORED',len(done))


def sparse_ce(logits, top_ids, top_probs):
    """Teacher top-k *renormalized* forward cross-entropy on student-visited prefixes."""
    import torch
    lp=torch.log_softmax(logits.float(),dim=-1)
    selected=lp.gather(-1,top_ids)
    mass=top_probs.sum(-1)
    assert bool(torch.all(mass>0))
    per_token=-(top_probs*selected).sum(-1)/mass
    return per_token, mass


def train(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model,PeftModel
    rows,man=prepared(args.out)
    teacher_dir=Path(args.out)/'teacher'/args.condition
    run=read_json(teacher_dir/'run.json')
    assert run['complete'] and run['condition']==args.condition
    assert run['examples_sha256']==man['examples_sha256']
    assert sha_file(teacher_dir/'scores.jsonl')==run['scores_sha256']
    score_list,_=decode_rows(teacher_dir/'scores.jsonl')
    scores={z['id']:z for z in score_list}
    assert set(scores)=={r['id'] for r in rows}
    tok=AutoTokenizer.from_pretrained(args.student,local_files_only=True)
    end_id=tok.convert_tokens_to_ids('<|im_end|>')
    data=[]
    for r in rows:
        z=scores[r['id']]
        assert z['response_sha256']==sha_bytes(r['response'].encode())
        pi=tok.apply_chat_template([{'role':'user','content':r['prompt']}],
                                   tokenize=True,add_generation_prompt=True)
        ri=tok.encode(r['response'],add_special_tokens=False)
        assert ri==z['target_token_ids'] and len(ri)==len(z['top_ids'])==len(z['token_types'])
        assert len(pi)+len(ri)<=args.max_student_len,'Student context exceeds cap'
        assert len(z['top_ids'])==len(z['top_probs'])==len(z['top_mass'])
        data.append(dict(pi=pi,ri=ri,z=z,episode=r['episode']))
    mean_mass=sum(sum(d['z']['top_mass']) for d in data)/sum(len(d['ri']) for d in data)
    assert mean_mass>=args.min_top_mass, f'top-k teacher mass {mean_mass:.3f} too low; increase --topk'
    torch.manual_seed(args.seed)
    model=AutoModelForCausalLM.from_pretrained(args.student,local_files_only=True,
                                      dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model.config.use_cache=False
    if args.base_adapter:
        model=PeftModel.from_pretrained(model,args.base_adapter,is_trainable=True)
    else:
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
                          target_modules=TARGETS,bias='none',task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    if args.gradient_checkpointing:model.gradient_checkpointing_enable()
    model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    assert params
    optim=torch.optim.AdamW(params,lr=args.lr)
    order=list(range(len(data)))*args.epochs
    random.Random(args.seed).shuffle(order)
    logs=[];t0=time.time();optim.zero_grad(set_to_none=True)
    for i,index in enumerate(order):
        d=data[index];z=d['z']; n=len(d['ri'])
        ids=torch.tensor([d['pi']+d['ri']],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(input_ids=ids,use_cache=False).logits[0,len(d['pi'])-1:-1]
        assert logits.shape[0]==n
        top_ids=torch.tensor(z['top_ids'],dtype=torch.long,device='cuda')
        top_probs=torch.tensor(z['top_probs'],dtype=torch.float32,device='cuda')
        loss_tokens,mass=sparse_ce(logits,top_ids,top_probs)
        loss=loss_tokens.mean()
        # Per-token teacher CE is applied on ALL response positions, including <think>.
        current_group=i//args.grad_accum
        group_size=min(args.grad_accum,len(order)-current_group*args.grad_accum)
        (loss/group_size).backward()
        is_end=(i+1)%args.grad_accum==0 or i+1==len(order)
        if is_end:
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optim.step();optim.zero_grad(set_to_none=True)
        if i%max(1,len(order)//40)==0 or is_end:
            think=[j for j,k in enumerate(z['token_types']) if k=='think']
            record=dict(step=i+1,loss=float(loss.detach()),top_mass=float(mass.mean()),
                        think_tokens=len(think),all_tokens=n,
                        think_loss=float(loss_tokens[think].mean().detach()) if think else None,
                        elapsed_s=time.time()-t0)
            logs.append(record)
            if is_end:print(json.dumps(record),flush=True)
    dest=Path(args.save).resolve();dest.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(str(dest))
    save_json(dest/'exp2_train.json',dict(condition=args.condition,seed=args.seed,lr=args.lr,
         n_examples=len(data),epochs=args.epochs,grad_accum=args.grad_accum,
         teacher_scores_sha256=run['scores_sha256'],examples_sha256=man['examples_sha256'],
         objective='normalized_topk_forward_kl_on_student_prefixes',
         all_response_tokens_trained=True,mean_teacher_topk_mass=mean_mass,log=logs))
    assert (dest/'adapter_model.safetensors').exists()
    print('TRAINED',dest)


def audit(args):
    rows,man=prepared(args.out)
    summaries={}
    for c in CONDITIONS:
        p=Path(args.out)/'teacher'/c
        run=read_json(p/'run.json')
        assert run['complete'] and run['examples_sha256']==man['examples_sha256']
        records,_=decode_rows(p/'scores.jsonl')
        assert sha_file(p/'scores.jsonl')==run['scores_sha256']
        sums=defaultdict(list)
        for x in records:
            for typ,lp,m in zip(x['token_types'],x['target_logp'],x['top_mass']):
                sums[typ+'_target_logp'].append(lp);sums[typ+'_topmass'].append(m)
        summaries[c]={k:dict(n=len(v),mean=sum(v)/len(v)) for k,v in sums.items()}
    save_json(Path(args.out)/'teacher_diagnostics.json',summaries)
    print(json.dumps(summaries,indent=2))
    print('NOTE: logprob changes are diagnostics, not proof of better gradients')


def selftest(_):
    ep=dict(task='find apple',initial_obs='kitchen',won=False,steps=[
        dict(response='<think>r1</think><action>open cabinet</action>',action='open cabinet',obs_after='empty',won_after=False),
        dict(response='<think>r2</think><action>look</action>',action='look',obs_after='empty',won_after=False)])
    lg=context_log(ep,1,'L_guarded'); ln=context_log(ep,1,'LN_guarded')
    assert '<think>r1' not in lg and '<think>r2' in lg
    assert '<think>' not in ln
    assert 'open cabinet' in lg and 'open cabinet' in ln
    row=dict(prompt='X',future=dict(L_guarded=lg,LN_guarded=ln))
    assert 'r1' not in teacher_user(row,'L_guarded')
    assert 'Unavailable' in teacher_user(row,'M')
    print('SELFTEST PASS')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=('rollout','prepare','teacher','train','audit','selftest'))
    ap.add_argument('--repo',default=str(ROOT))
    ap.add_argument('--rollouts',default=None)
    ap.add_argument('--rollout-out',default=None)
    ap.add_argument('--max-games',type=int,default=400)
    ap.add_argument('--reps',type=int,default=1)
    ap.add_argument('--workers',type=int,default=24)
    ap.add_argument('--rollout-chunk',type=int,default=200)
    ap.add_argument('--out',default=str(HERE/'results/exp2_opd/round0'))
    ap.add_argument('--condition',choices=CONDITIONS,default='LN_guarded')
    ap.add_argument('--student',default='/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
    ap.add_argument('--teacher',default='/root/autodl-tmp/models/Qwen2.5-32B-Instruct')
    ap.add_argument('--max-examples',type=int,default=512)
    ap.add_argument('--adapter-provenance',default=None,help='Exact adapter path from full rollout metadata; omitted means any')
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--topk',type=int,default=256)
    ap.add_argument('--min-top-mass',type=float,default=0.90)
    ap.add_argument('--lr',type=float,default=2e-5)
    ap.add_argument('--epochs',type=int,default=1)
    ap.add_argument('--grad-accum',type=int,default=16)
    ap.add_argument('--max-student-len',type=int,default=6144)
    ap.add_argument('--gradient-checkpointing',action='store_true')
    ap.add_argument('--base-adapter',default=None)
    ap.add_argument('--save',default=None)
    args=ap.parse_args()
    assert 1<=args.topk<=2048 and args.grad_accum>=1 and args.epochs>=1
    if args.command=='prepare' and not args.rollouts:ap.error('prepare requires --rollouts')
    if args.command=='train' and not args.save:ap.error('train requires --save')
    if args.command=='rollout' and not args.rollout_out:ap.error('rollout requires --rollout-out')
    {'rollout':rollout,'prepare':prepare,'teacher':score_teacher,'train':train,'audit':audit,'selftest':selftest}[args.command](args)


if __name__=='__main__':main()
