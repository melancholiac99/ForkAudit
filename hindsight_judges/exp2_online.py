"""Experiment 2 (formal): synchronous, one-step-per-fresh-batch on-policy distillation.

Two simultaneously resident models on DIFFERENT CUDA devices:
  GPU student: Qwen2.5-7B + LoRA, generating complete ALFWorld episodes AND training.
  GPU teacher: frozen, hash-pinned Qwen2.5-32B, evaluating full-vocabulary token
               distributions under a conditioned hindsight prompt.

After every optimizer step the Student samples new *complete* episodes. No rollout
batch, response or teacher distribution is used for more than one optimizer step.
The privileged evidence is NEVER part of the Student's prompt. All generated
response tokens, including <think>, receive gradient from full-vocabulary KL (reverse by default; forward is a specified ablation).

Primary comparison: L_guarded vs LN_guarded. M is a matched empty-hindsight
control; M_native is an unmodified Student prompt (standard no-privilege OPD).
No reward/credit weighting is silently introduced. Frozen E1/E1c not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import re
from pathlib import Path

from exp2_common import ROOT, read_json, save_json, sha_file, encode_rows
from exp2_opd import teacher_user

CONDITIONS = ('M_native', 'M', 'L_guarded', 'LN_guarded')
TARGETS = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
MAX_TEACHER_TOKENS = 32768
MAX_RESPONSE_TOKENS = 512


def teacher_prompt(row, condition):
    """Only M_native lacks the fixed wrapper. Main L/LN comparison is matched."""
    if condition == 'M_native':
        return row['prompt']
    return teacher_user(row, condition)


def eligible_games(split, step, seed, batch_games):
    """One deterministic task allocation shared by all conditions and seeds.

    Each block of ceil(n / batch_games) updates is one shuffled sweep; later
    sweeps reshuffle. Never choose tasks based on success or teacher score.
    """
    games = list(split['train'])
    excluded = {g['game_file'] for k in ('tune', 'valid_seen', 'valid_unseen') for g in split[k]}
    assert games and len({g['game_file'] for g in games}) == len(games)
    assert not excluded.intersection(g['game_file'] for g in games), 'TRAIN overlaps validation'
    assert 0 < batch_games <= len(games)
    epoch, offset = divmod(step * batch_games, len(games))
    # Permit the batch to cross epoch boundaries with no missing games.
    selected = []
    while len(selected) < batch_games:
        order = list(range(len(games)))
        random.Random(seed + 111_111 * epoch).shuffle(order)
        take = min(batch_games-len(selected), len(games)-offset)
        selected.extend(games[order[j]] for j in range(offset,offset+take))
        epoch += 1
        offset = 0
    assert len(selected) == batch_games
    return selected


def rollout_seed(task_seed, turn, step, seed):
    """Common random numbers among arms; new sampling stream each update."""
    b = f'hindsight-exp2-sync:{seed}:{step}:{task_seed}:{turn}'.encode()
    return int(hashlib.sha256(b).hexdigest()[:15],16)


def guarded_context_log(ep, turn, mode):
    """Like E2 prototype, but redact unfinished <think> spans as well.

    This matters when a sampled Student truncates during its thinking phase:
    accidentally copying unfinished rationale into privileged evidence would
    both leak target text and bias the intended intervention.
    """
    assert mode in ('L_guarded','LN_guarded') and 1 <= turn <= len(ep['steps'])
    rows=[]
    for j,st in enumerate(ep['steps'],1):
        response=st['response']
        if j==turn or mode=='LN_guarded':
            response=re.sub(r'<think>.*?(?:</think>|$)\s*','',response,flags=re.I|re.S)
        rows.append(dict(turn=j,response=response,action=st['action'],
                         obs_after=st['obs_after'],won_after=bool(st['won_after'])))
    return json.dumps(dict(task=ep['task'],initial_obs=ep['initial_obs'],
                           steps=rows,final_reward=int(ep['won'])),ensure_ascii=False)


def pack_decisions(episodes, step, seed, max_decisions):
    """Construct target rows only after the COMPLETE factual episode is known."""
    rows = []
    for ep in episodes:
        assert ep['steps'] and all('response_ids' in st for st in ep['steps'])
        eid=f"s{ep['seed']}_u{step:04d}"
        for turn, st in enumerate(ep['steps'],1):
            if not st['response_ids']:
                continue
            row = dict(id=f'{eid}_t{turn:02d}', episode=eid, turn=turn,
                       prompt=st['prompt'], response=st['response'],
                       response_ids=st['response_ids'], game_file=ep['game_file'],
                       won=bool(ep['won']),
                       future={c:guarded_context_log(ep,turn,c)
                               for c in ('L_guarded','LN_guarded')})
            # The two privileged channels have byte-identical non-Think data.
            a=json.loads(row['future']['L_guarded'])
            b=json.loads(row['future']['LN_guarded'])
            assert a['final_reward']==b['final_reward']
            assert all((x['action'],x['obs_after'],x['won_after']) ==
                       (y['action'],y['obs_after'],y['won_after'])
                       for x,y in zip(a['steps'],b['steps']))
            assert '<think>' not in b['steps'][turn-1]['response'].lower()
            assert '<think>' not in a['steps'][turn-1]['response'].lower()
            rows.append(row)
    assert len(set(z['id'] for z in rows))==len(rows)
    # Outcome-blind selection; do not pick favorable/failed trajectories.
    rows.sort(key=lambda r:hashlib.sha256(f"{seed}:{step}:{r['id']}".encode()).hexdigest())
    if max_decisions > 0:
        rows = rows[:max_decisions]
    assert rows, 'No generated Student responses in on-policy batch'
    return sorted(rows,key=lambda r:r['id'])


def classify_generated(tok, ids):
    """Diagnose think tokens without ever re-tokenizing the sampled sequence.

    Fast-path when text re-encoding matches sampled ids; slow-path uses exact
    prefix decoding boundaries (works for non-canonical BPE segmentations).
    """
    response=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    probe=tok(response,add_special_tokens=False,return_offsets_mapping=True)
    if probe['input_ids']==list(ids):
        return _think_types_incomplete(response,probe['offset_mapping'])
    offsets=[]
    previous=''
    for k in range(1,len(ids)+1):
        current=tok.decode(ids[:k],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert current.startswith(previous), 'Non-monotonic token decoding; cannot classify think spans'
        offsets.append((len(previous),len(current)))
        previous=current
    assert previous==response
    return _think_types_incomplete(response,offsets)


def _think_types_incomplete(text, offsets):
    """Count closed AND unfinished <think> spans (truncation can leave it open)."""
    spans = [(m.start(),m.end()) for m in re.finditer(r'<think>.*?(?:</think>|$)', text, re.S | re.I)]
    return ['think' if any(lo < b and hi > a for lo,hi in spans) else 'other'
            for a,b in offsets]


def sample_episodes(pool, model, tok, games, *, step, seed, student_device, max_steps, max_new_tokens, max_student_len):
    """HF generation directly from the *current* trainable Student, no stale actor."""
    import torch
    import common
    assert max_steps <= common.MAX_STEPS and max_new_tokens <= common.MAX_RESPONSE_TOKENS
    assert not model.training, 'Policy should be eval during rollout; use train only for gradients'
    jobs=[(str(g['seed']),str(Path(os.environ.get('ALFWORLD_DATA','/root/autodl-tmp/alfworld_data')) /
                              g['game_file']),[],None) for g in games]
    init=pool.open(jobs)
    states={}
    try:
        for g in games:
            key=str(g['seed'])
            obs,adm,won,_=init[key]
            states[key]=dict(seed=g['seed'],game_file=g['game_file'],task=common.extract_task(obs),
                             initial_obs=obs,obs=obs,adm=adm,mem=[],steps=[],won=bool(won),done=bool(won))
        for turn in range(1,max_steps+1):
            active=[k for k in states if not states[k]['done']]
            if not active:break
            actions={}
            for key in active:
                s=states[key]
                prompt=common.build_prompt(s['task'],s['mem'],s['obs'],s['adm'])
                pre=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                                            add_generation_prompt=True)
                assert pre and len(pre)+max_new_tokens<=min(max_student_len,model.config.max_position_embeddings), \
                    'Student generation context length overflow; no truncation'
                input_ids=torch.tensor([pre],device=student_device,dtype=torch.long)
                # Keep sampling RNG independent of training RNG; common random numbers.
                with torch.random.fork_rng(devices=[torch.device(student_device).index]):
                    torch.manual_seed(rollout_seed(s['seed'],turn,step,seed))
                    with torch.inference_mode():
                        generated=model.generate(input_ids=input_ids,max_new_tokens=max_new_tokens,
                                    do_sample=True,temperature=common.TEMPERATURE,top_k=0,top_p=1.0,
                                    repetition_penalty=1.0,use_cache=True,
                                    eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id)
                ri=generated[0,len(pre):].tolist()
                if ri and ri[-1]==tok.eos_token_id:ri.pop()
                if not ri:raise RuntimeError(f'Empty generated response {key}, turn {turn}')
                response=tok.decode(ri,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                action,valid,fmt=common.parse(response)
                actions[key]=action
                s['steps'].append(dict(turn=turn,prompt=prompt,response=response,
                         response_ids=ri,action=action,valid=bool(valid),format=fmt,
                         obs_before=s['obs'],admissible=s['adm'],sample_seed=rollout_seed(s['seed'],turn,step,seed)))
            responses=pool.step(actions)
            for key in active:
                obs,adm,won,done=responses[key]
                s=states[key]
                s['mem'].append({'text_obs':s['obs'],'action':actions[key]})
                s['steps'][-1].update(obs_after=obs,won_after=bool(won))
                s['obs'],s['adm'],s['won'],s['done']=obs,adm,bool(won),bool(done or won)
        episodes=[]
        for key in sorted(states):
            s=states[key]
            episodes.append({k:v for k,v in s.items() if k not in ('obs','adm','mem')})
        return episodes
    finally:
        pool.close(list(init))


def teacher_log_probs(scorer, tok, row, condition, response_ids, teacher_device,
                  *, max_len=MAX_TEACHER_TOKENS, head_chunk=64):
    """Exact full-vocabulary Teacher LOG distribution on *sampled token IDs*.

    Builds one forward pass with Student's previous generated tokens as prefix;
    current-token targets never appear in their own probability condition.
    The target turn's think is redacted from any duplicated future evidence.
    """
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    prompt=teacher_prompt(row,condition)
    pre=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                                add_generation_prompt=True)
    ids=list(response_ids)
    assert ids and len(pre)+len(ids) <= max_len, 'Teacher context > 32,768; no silent truncation'
    assert max(ids)<scorer.model.config.vocab_size
    with torch.inference_mode(), torch.cuda.device(teacher_device), sdpa_kernel(SDPBackend.MATH):
        cache,h=scorer._feed(pre+ids[:-1],len(pre)-1)
        assert len(h)==len(ids)
        del cache
        ps=[]
        for i in range(0,len(ids),head_chunk):
            logits=scorer.model.lm_head(h[i:i+head_chunk]).float()
            logp=torch.log_softmax(logits,dim=-1)
            mass=logp.exp().sum(-1)
            assert torch.isfinite(logp).all() and torch.allclose(mass,torch.ones_like(mass),atol=1e-4)
            ps.append(logp.to(device='cpu',dtype=torch.float32))
            del logits,logp,mass
        del h
    return torch.cat(ps,dim=0)


def full_vocab_kl(student_logits, teacher_log, direction='reverse'):
    """Per-token exact KL, no top-k and NO teacher probability floor.

    The Teacher sends its *log probabilities*, not softmax probabilities,
    because underflow to zero would corrupt reverse KL for unlikely tokens.
    """
    import torch
    import torch.nn.functional as F
    assert direction in ('reverse','forward')
    assert student_logits.ndim==2 and teacher_log.shape==student_logits.shape
    assert torch.isfinite(teacher_log).all()
    student_log=F.log_softmax(student_logits.float(),dim=-1)
    t=teacher_log.detach().float()
    if direction=='reverse':
        p=student_log.exp()
        kl=(p*(student_log-t)).sum(-1)  # KL(Student || Teacher)
    else:
        p=t.exp()
        kl=(p*(t-student_log)).sum(-1)  # KL(Teacher || Student)
    return kl


def full_forward_kl(student_logits, probs):
    """Compatibility check for existing CPU tests; actual online training uses log p."""
    import torch
    return full_vocab_kl(student_logits,probs.clamp_min(1e-30).log(),'forward')


def numerical_gate(scorer, tok, teacher_device):
    """GPU gate: chunked prefill equals single-pass FP16/FP32-head distribution."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    pre=tok.apply_chat_template([{'role':'user','content':'ALFWorld calibration prompt'}],
                       tokenize=True,add_generation_prompt=True)
    suffix=tok.encode('<think>test</think><action>look</action>',add_special_tokens=False)
    ids=pre+suffix[:-1]
    assert 0<len(ids)<1024
    with torch.inference_mode(), torch.cuda.device(teacher_device), sdpa_kernel(SDPBackend.MATH):
        cache,chunk=scorer._feed(ids,len(pre)-1)
        del cache
        direct=scorer.model.model(input_ids=torch.tensor([ids],device=teacher_device),
                                  use_cache=False).last_hidden_state[0,len(pre)-1:]
        assert direct.shape==chunk.shape
        # Distributions, not hidden states, are the scientific quantity.
        pa=torch.softmax(scorer.model.lm_head(chunk).float(),dim=-1)
        pb=torch.softmax(scorer.model.lm_head(direct.float()).float(),dim=-1)
        tv=float((pa-pb).abs().sum(-1).max().item()/2)
        assert math.isfinite(tv) and tv<0.005, f'Teacher precision gate failed: {tv}'
    return dict(passed=True,max_total_variation=tv,tokens=len(ids),
                check='chunked_vs_onepass_token_fullvocab_TV',threshold=0.005)


def student_kl(model, tok, row, teacher_log_cpu, student_device, max_len, direction):
    import torch
    ids=row['response_ids']
    pre=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,
                                add_generation_prompt=True)
    assert len(pre)+len(ids)<=max_len, 'Student training context overflow; do not truncate'
    all_ids=torch.tensor([pre+ids],device=student_device,dtype=torch.long)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        out=model(input_ids=all_ids,use_cache=False)
        logits=out.logits[0,len(pre)-1:-1]
    assert len(logits)==len(ids), 'Incorrect token alignment'
    return full_vocab_kl(logits,teacher_log_cpu.to(student_device),direction),logits


def split_statistics(kls,kinds):
    """Detached per-token diagnostics; cannot establish gradient correctness."""
    assert len(kls)==len(kinds)
    out={}
    for name in ('think','other'):
        vals=[v for v,k in zip(kls,kinds) if k==name]
        out[name]={'n':len(vals),'mean_kl':(sum(vals)/len(vals) if vals else None)}
    return out


def validate_config(args):
    assert args.student_device != args.teacher_device, 'Teacher and Student need distinct GPUs'
    assert args.n_updates>=1 and args.games_per_update>=1 and args.max_decisions_per_update>=1
    assert args.max_student_len>=1024 and 1<=args.max_new_tokens<=MAX_RESPONSE_TOKENS
    assert 1<=args.max_steps<=50
    assert args.lr>0 and args.seed>=0 and args.num_workers>=1
    assert args.teacher_head_chunk>0 and args.grad_clip>0
    assert args.kl_direction in ('reverse','forward')
    if args.n_updates>1:
        assert args.localization_analysis and args.scope_reviewed, (
            'Formal multi-update training requires --localization-analysis and --scope-reviewed; '
            'the guarded L/LN comparison tests OTHER-TURN reasoning, not the decision-t rationale')
        loc=read_json(args.localization_analysis)
        assert loc.get('n_points')==386 and 'DROP_T_minus_L_value' in loc and (
            'KEEP_T_minus_L_value' in loc), 'Localization analysis is missing or incomplete'


def run(args):
    validate_config(args)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    from exp2_common import load_32b
    # Fork ALFWorld workers before any CUDA init, following the E4 harness.
    sys.path.insert(0,str(Path(args.repo)/'scripts'))
    from envpool import EnvPool
    pool=EnvPool(args.num_workers)
    try:
        assert torch.cuda.is_available() and torch.cuda.device_count()>=2
        student_device=torch.device(args.student_device)
        teacher_device=torch.device(args.teacher_device)
        assert student_device.type==teacher_device.type=='cuda'
        assert student_device.index!=teacher_device.index
        train_games=read_json(Path(args.repo)/'exploratory_E4/games.json')
        assert args.games_per_update<=len(train_games['train'])
        tok=AutoTokenizer.from_pretrained(args.student,local_files_only=True,use_fast=True)
        teacher_tok=AutoTokenizer.from_pretrained(args.teacher,local_files_only=True,use_fast=True)
        assert tok.is_fast and teacher_tok.is_fast
        assert tok.get_vocab()==teacher_tok.get_vocab(), 'Teacher and Student token IDs differ'
        assert tok.eos_token_id==teacher_tok.eos_token_id
        torch.manual_seed(args.seed)
        # Load frozen E1c-precision teacher with its checked model hashes on GPU 1.
        with torch.cuda.device(teacher_device):
            scorer,tmeta=load_32b(Path(args.teacher),Path(args.repo))
        assert next(scorer.model.parameters()).device==teacher_device
        scorer.model.eval()
        student=AutoModelForCausalLM.from_pretrained(args.student,local_files_only=True,
                        dtype=torch.bfloat16,attn_implementation='sdpa').to(student_device)
        student.config.use_cache=False
        output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
        conf={'condition':args.condition,'seed':args.seed,'teacher_weights':tmeta['weights'],
              'teacher_precision':tmeta['precision']['policy'],
              'student':str(Path(args.student).resolve()),'teacher':str(Path(args.teacher).resolve()),
              'student_weights':{p.name:sha_file(p) for p in sorted(Path(args.student).glob('*.safetensors'))},
              'localization_analysis_sha256':(sha_file(args.localization_analysis)
                  if args.localization_analysis else None),
              'script_sha256':sha_file(Path(__file__)),
              'game_split_sha256':sha_file(Path(args.repo)/'exploratory_E4/games.json'),
              'games_per_update':args.games_per_update,'max_decisions_per_update':args.max_decisions_per_update,
              'max_steps':args.max_steps,'max_new_tokens':args.max_new_tokens,
              'max_student_len':args.max_student_len,'lr':args.lr,'n_updates':args.n_updates,
              'teacher_head_chunk':args.teacher_head_chunk,
              'gradient_checkpointing':bool(args.gradient_checkpointing),
              'grad_clip':args.grad_clip, 'num_workers':args.num_workers,
              'optimizer':'AdamW','loss':f'exact_full_vocab_{args.kl_direction}_KL',
              'kl_direction':args.kl_direction,
              'all_think_tokens_trained':True,'one_optimizer_step_per_new_rollout_batch':True}
        assert conf['student_weights'], 'Student checkpoint not found'
        if (output/'config.json').exists():
            assert read_json(output/'config.json')==conf, 'Cannot resume altered experiment configuration'
        else:
            assert not list(output.glob('step_*')), 'Step outputs exist without config'
            save_json(output/'config.json',conf)
        if not (output/'numerical_gate.json').exists():
            save_json(output/'numerical_gate.json', numerical_gate(scorer,teacher_tok,teacher_device))
        else:
            assert read_json(output/'numerical_gate.json')['passed']
        checkpoints=sorted(p for p in output.glob('step_*') if p.is_dir())
        complete=[]
        for p in checkpoints:
            if not (p/'complete.json').exists():
                raise RuntimeError(f'Incomplete step directory {p}; investigate, do not silently overwrite')
            complete.append(p)
        assert [p.name for p in complete]==[f'step_{i:05d}' for i in range(len(complete))], \
            'Nonconsecutive completed checkpoints'
        start=len(complete)
        if start:
            prev=complete[-1]
            rec=read_json(prev/'complete.json')
            assert rec['adapter_sha256']==sha_file(prev/'adapter_model.safetensors')
            student=PeftModel.from_pretrained(student,str(prev),is_trainable=True)
        else:
            student=get_peft_model(student,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
                        target_modules=list(TARGETS),bias='none',task_type='CAUSAL_LM'))
        student.enable_input_require_grads()
        if args.gradient_checkpointing:student.gradient_checkpointing_enable()
        params=[p for p in student.parameters() if p.requires_grad]
        assert params
        opt=torch.optim.AdamW(params,lr=args.lr)
        if start:
            state=torch.load(prev/'optimizer.pt',map_location=student_device,weights_only=True)
            opt.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['cpu_rng'])
            torch.cuda.set_rng_state(state['student_cuda_rng'],student_device)
            assert state['step']==start
        if start:
            del state
        for step in range(start,args.n_updates):
            tick=time.time();student.eval()
            games=eligible_games(train_games,step,args.seed,args.games_per_update)
            episodes=sample_episodes(pool,student,tok,games,step=step,seed=args.seed,
                                     student_device=student_device,max_steps=args.max_steps,
                                     max_new_tokens=args.max_new_tokens,
                                     max_student_len=args.max_student_len)
            selected=pack_decisions(episodes,step,args.seed,args.max_decisions_per_update)
            # The exact same Student that generated these episodes now receives one update.
            student.train();opt.zero_grad(set_to_none=True)
            rng_shuffle=random.Random(args.seed*10_000+step)
            rng_shuffle.shuffle(selected)
            records=[]; loss_sum=0.0
            for row in selected:
                teacher_log=teacher_log_probs(scorer,teacher_tok,row,args.condition,row['response_ids'],teacher_device,
                                head_chunk=args.teacher_head_chunk)
                assert teacher_log.shape[-1]==student.config.vocab_size
                kl, logits=student_kl(student,tok,row,teacher_log,student_device,
                                      args.max_student_len,args.kl_direction)
                # One vote per visited decision, not proportional to response length.
                loss=kl.mean()/len(selected)
                assert torch.isfinite(loss), 'Nonfinite KL: stop before optimizer update'
                loss.backward()
                kls=kl.detach().float().cpu().tolist()
                kinds=classify_generated(tok,row['response_ids'])
                rec=dict(id=row['id'],n_tokens=len(kls),
                         target_think_tokens=sum(k=='think' for k in kinds),
                         by_token_type=split_statistics(kls,kinds),
                         kl=float(kl.mean().detach()),
                         teacher_target_mean_prob=float(teacher_log.gather(-1,
                                    torch.tensor(row['response_ids'])[:,None]).exp().mean()),
                         student_target_mean_prob=float(logits.detach().float().softmax(-1).gather(-1,
                                   torch.tensor(row['response_ids'],device=student_device)[:,None]).mean()))
                records.append(rec)
                loss_sum+=float(kl.mean().detach())
                del kl,logits,teacher_log,loss
            norm=float(torch.nn.utils.clip_grad_norm_(params,args.grad_clip))
            assert math.isfinite(norm), 'Nonfinite gradient: refusing update'
            opt.step();opt.zero_grad(set_to_none=True)
            # Store a hash-bound atomic checkpoint; no stale data is reused next step.
            staged=output/f'.step_{step:05d}.staging'
            final=output/f'step_{step:05d}'
            assert not staged.exists() and not final.exists()
            staged.mkdir()
            student.save_pretrained(str(staged))
            torch.save({'step':step+1,'optimizer':opt.state_dict(),
                        'cpu_rng':torch.get_rng_state(),
                        'student_cuda_rng':torch.cuda.get_rng_state(student_device)},
                        staged/'optimizer.pt')
            encode_rows(staged/'episodes.jsonl.gz',episodes)
            encode_rows(staged/'decisions.jsonl.gz',selected)
            encode_rows(staged/'diagnostics.jsonl.gz',records)
            summary=dict(update=step+1,policy_batch_fresh=True,optimizer_steps=1,
                         parent_adapter_sha256=(read_json(complete[-1]/'complete.json')['adapter_sha256']
                              if step == start and start else
                              (sha_file(output/f'step_{step-1:05d}'/'adapter_model.safetensors') if step else None)),
                         examples=len(selected),episodes=len(episodes),
                         episode_success=sum(ep['won'] for ep in episodes)/len(episodes),
                         loss_mean=loss_sum/len(selected),grad_norm_pre_clip=norm,
                         time_s=time.time()-tick,adapter_sha256=sha_file(staged/'adapter_model.safetensors'),
                         rollouts_sha256=sha_file(staged/'episodes.jsonl.gz'),
                         decisions_sha256=sha_file(staged/'decisions.jsonl.gz'),
                         diagnostics_sha256=sha_file(staged/'diagnostics.jsonl.gz'))
            save_json(staged/'complete.json',summary)
            os.replace(staged,final)
            print(json.dumps({'condition':args.condition,'seed':args.seed,**summary}),flush=True)
        print('COMPLETE',args.condition,args.seed,args.n_updates,'on-policy updates')
    finally:
        pool.shutdown()


def run_eval(args):
    """Held-out ALFWorld evaluation with the SAME HF generator used by training.

    Run for each adapter/seed with identical evaluation tasks and random streams;
    never use valid_unseen to choose hyperparameters or checkpoints.
    """
    assert args.adapter and args.eval_output, 'eval requires --adapter and --eval-output'
    assert args.eval_reps>=1
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    sys.path.insert(0,str(Path(args.repo)/'scripts'))
    from envpool import EnvPool
    pool=EnvPool(args.num_workers)
    try:
        games=read_json(Path(args.repo)/'exploratory_E4/games.json')
        assert args.eval_split in ('tune','valid_seen','valid_unseen')
        ids=games[args.eval_split]
        assert set(g['game_file'] for g in ids).isdisjoint(g['game_file'] for g in games['train'])
        assert args.eval_chunk>=1
        tok=AutoTokenizer.from_pretrained(args.student,local_files_only=True,use_fast=True)
        device=torch.device(args.student_device)
        assert torch.cuda.is_available() and device.type=='cuda'
        model=AutoModelForCausalLM.from_pretrained(args.student,local_files_only=True,
                       dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
        model=PeftModel.from_pretrained(model,args.adapter,is_trainable=False).eval()
        dest=Path(args.eval_output).resolve()
        dest.mkdir(parents=True,exist_ok=True)
        expected=dict(split=args.eval_split,adapter=str(Path(args.adapter).resolve()),
                      adapter_sha256=sha_file(Path(args.adapter)/'adapter_model.safetensors'),
                      source_games_sha256=sha_file(Path(args.repo)/'exploratory_E4/games.json'),
                      seed=args.eval_seed,reps=args.eval_reps,
                      max_steps=args.max_steps,max_new_tokens=args.max_new_tokens,
                      max_student_len=args.max_student_len)
        pconf=dest/'eval_config.json'
        if pconf.exists():
            assert read_json(pconf)==expected, 'Cannot mix heldout evaluations with different adapters/settings'
        else:
            assert not list(dest.glob('s*_r*.json')), 'Results exist without config'
            save_json(pconf,expected)
        per_game={}
        for rep in range(args.eval_reps):
            for i in range(0,len(ids),args.eval_chunk):
                grp=ids[i:i+args.eval_chunk]
                paths=[dest/f"s{g['seed']}_r{rep}.json" for g in grp]
                if all(p.exists() for p in paths):
                    continue
                assert not any(p.exists() for p in paths), 'Partial evaluation batch; review before retry'
                episodes=sample_episodes(pool,model,tok,grp,step=1_000_000+rep,
                                 seed=args.eval_seed,student_device=device,
                                 max_steps=args.max_steps,max_new_tokens=args.max_new_tokens,
                                 max_student_len=args.max_student_len)
                assert len(episodes)==len(paths)
                by_seed={ep['seed']:ep for ep in episodes}
                for g,p in zip(grp,paths):
                    save_json(p,by_seed[g['seed']])
        for g in ids:
            per_game[str(g['seed'])]=[
                int(read_json(dest/f"s{g['seed']}_r{rep}.json")['won'])
                for rep in range(args.eval_reps)]
        n=sum(map(len,per_game.values()))
        overall=sum(map(sum,per_game.values()))/n
        result=dict(split=args.eval_split,tag=dest.name,n=n,
                    success=overall,per_game=per_game,provenance=expected)
        save_json(dest/'summary.json',result)
        print('EVAL COMPLETE',dest,'success',overall,'n',n)
    finally:
        pool.shutdown()


def selftest(_args=None):
    train=[{'game_file':f'/a/{i}','seed':i} for i in range(8)]
    split={'train':train,'tune':[{'game_file':'/v'}],
           'valid_seen':[{'game_file':'/w'}],'valid_unseen':[{'game_file':'/u'}]}
    assert eligible_games(split,0,42,3)==eligible_games(split,0,42,3)
    assert set(x['seed'] for x in eligible_games(split,0,42,3)).isdisjoint(
        x['seed'] for x in eligible_games(split,1,42,3))
    assert rollout_seed(1,1,0,42)!=rollout_seed(1,1,1,42)
    ep=dict(seed=1,game_file='/a/1',task='find apple',initial_obs='kitchen',won=False,steps=[
        dict(prompt='p1',response='<think>r1</think><action>a</action>',response_ids=[2,3],
             action='a',obs_after='empty',won_after=False),
        dict(prompt='p2',response='<think>r2</think><action>b</action>',response_ids=[4,5],
             action='b',obs_after='empty',won_after=False)])
    rows=pack_decisions([ep],0,42,10)
    assert len(rows)==2
    assert all('<think>r1' not in r['future']['L_guarded'] if r['turn']==1 else True for r in rows)
    assert all('<think>' not in r['future']['LN_guarded'] for r in rows)
    assert teacher_prompt(rows[0],'M_native')==rows[0]['prompt']
    assert 'Unavailable' in teacher_prompt(rows[0],'M')
    try:
        import torch
        p=torch.tensor([[0.25,0.75]],requires_grad=False)
        x=torch.tensor([[0.0,0.0]],requires_grad=True)
        kl=full_forward_kl(x,p)
        assert kl.shape==(1,) and kl.item()>0
        kl.mean().backward()
        assert x.grad[0,1]<0 and x.grad[0,0]>0
    except ImportError:
        pass
    print('ONLINE SELFTEST PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('selftest','train','eval'))
    p.add_argument('--repo',default=str(ROOT))
    p.add_argument('--condition',choices=CONDITIONS,default='LN_guarded')
    p.add_argument('--student',default='/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
    p.add_argument('--teacher',default='/root/autodl-tmp/models/Qwen2.5-32B-Instruct')
    p.add_argument('--student-device',default='cuda:0')
    p.add_argument('--teacher-device',default='cuda:1')
    p.add_argument('--output',default=None)
    p.add_argument('--adapter',default=None)
    p.add_argument('--eval-output',default=None)
    p.add_argument('--eval-split',choices=('tune','valid_seen','valid_unseen'),default='valid_unseen')
    p.add_argument('--eval-seed',type=int,default=20261103)
    p.add_argument('--eval-reps',type=int,default=3)
    p.add_argument('--eval-chunk',type=int,default=8)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--n-updates',type=int,default=100)
    p.add_argument('--games-per-update',type=int,default=8)
    p.add_argument('--max-decisions-per-update',type=int,default=16)
    p.add_argument('--max-steps',type=int,default=50)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--max-student-len',type=int,default=6144)
    p.add_argument('--teacher-head-chunk',type=int,default=64)
    p.add_argument('--num-workers',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--localization-analysis',default=None)
    p.add_argument('--scope-reviewed',action='store_true',
                   help='Explicitly accept that L_guarded vs LN_guarded tests only other-turn Think')
    p.add_argument('--kl-direction',choices=('reverse','forward'),default='reverse')
    p.add_argument('--gradient-checkpointing',action='store_true',default=True)
    args=p.parse_args()
    if args.command=='train' and not args.output:p.error('train requires --output')
    if args.command=='eval' and (not args.adapter or not args.eval_output):
        p.error('eval requires --adapter and --eval-output')
    {'train':run,'selftest':selftest,'eval':run_eval}[args.command](args)


if __name__=='__main__':main()
