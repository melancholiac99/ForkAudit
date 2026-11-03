"""Full-response historical OPD-signal diagnostic. No training and no new trajectories."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import exp1c as E
import exp3_action_diagnostic as D
import exp2_common as C
from token_diagnostic_backend import prompt_ids

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
CONDS=('M','L','LN','F');KINDS=('think','action','other');SEED=20261104;NBOOT=4000
PROTOCOL=HERE/'PROTOCOL_TOKEN_DIAGNOSTIC_20261104.md'

def tokenize_response(tok,text,action):
    # Use the same exact action-text validation as the candidate diagnostic.
    span_method='original_action_tag'
    try:
        lead,_=D.action_format(text,action)
        start=len(D.student_prefix(text,'keep'))+len(lead);end=start+len(action)
    except ValueError as exc:
        if not str(exc).startswith('No tagged action'):raise
        # Some E4 factual actions themselves contain malformed markup. Keep the real
        # response AND replay action unchanged; identify its unique literal span.
        tail_start=text.lower().rfind('</think>')+len('</think>')
        matches=list(re.finditer(re.escape(action),text[tail_start:]))
        assert len(matches)==1,'Cannot uniquely locate original malformed executed action'
        start=tail_start+matches[0].start();end=tail_start+matches[0].end()
        span_method='exact_recorded_action_in_malformed_reply'
    assert text[start:end]==action
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids'];assert ids and tok.decode(ids,clean_up_tokenization_spaces=False)==text
    spans=[(m.start(1),m.end(1)) for m in re.finditer(r'<think>(.*?)</think>',text,re.S|re.I)]
    kinds=[]
    for a,b in encoded['offset_mapping']:
        # BPE often packs the leading space with a one-token action (e.g. " look").
        # That is still an action-content token; never drop such responses.
        piece=text[a:b];core_a=a+len(piece)-len(piece.lstrip());core_b=b-(len(piece)-len(piece.rstrip()))
        if b>a and any(lo<=a and b<=hi for lo,hi in spans):kinds.append('think')
        elif (b>a and start<=a and b<=end) or (core_b>core_a and start<=core_a and core_b<=end):kinds.append('action')
        else:kinds.append('other')
    assert len(ids)==len(kinds) and 'action' in kinds
    return dict(text=text,ids=ids,kinds=kinds,action_span_method=span_method,
                tokenization='canonical_reencoding_of_historical_text_no_added_EOS')

def evidence_prompts(prompt,episode,turn,feedback):
    privilege={'M':None,'L':D.redact_episode(episode,turn,False),'LN':D.redact_episode(episode,turn,True),'F':feedback}
    return {c:prompt if c=='M' else prompt+'\n\n[PRIVILEGED ENVIRONMENT EVIDENCE — not part of the Agent history; '
            'use it to choose the best action, do not copy it]\n'+json.dumps(privilege[c],ensure_ascii=False)+
            '\n[/PRIVILEGED ENVIRONMENT EVIDENCE]' for c in CONDS}

def tokenizers(args):
    from transformers import AutoTokenizer
    a=AutoTokenizer.from_pretrained(args.student,local_files_only=True,use_fast=True)
    b=AutoTokenizer.from_pretrained(args.teacher,local_files_only=True,use_fast=True)
    assert a.is_fast and b.is_fast and a.get_vocab()==b.get_vocab() and a.chat_template==b.chat_template
    assert a.eos_token_id==b.eos_token_id
    return a

def prepare(args):
    out=Path(args.out);tok=tokenizers(args);sources={};groups=[];audit={}
    tokenizer_files={str(p.resolve()):C.sha_file(p) for root in [Path(args.student),Path(args.teacher)]
                     for p in [root/'tokenizer.json',root/'tokenizer_config.json',root/'config.json']}
    sources.update(tokenizer_files)
    def source(p):
        p=Path(p);sources[str(p.resolve())]=C.sha_file(p)
        return C.read_json(p)
    arms=Path(args.arms)
    expected=C.read_json(ROOT/'exploratory_P4/e2.json')['inputs']['exploratory_deep_discovery/exploratory_arms.parquet']
    assert C.sha_file(arms)==expected;sources[str(arms.resolve())]=expected
    if args.cohort=='p4':
        meta,e2,_,template,hashes=E.original_sources(ROOT)
        q=E.read_arms(ROOT,arms,expected)
        ftemplate=(ROOT/'exploratory_P4/prompts/feedback.txt').read_text()
        for z in meta:
            pid=z['point_id'];s,l,f=[e2[pid][c] for c in ['S','L','F']]
            ep=E.extract_episode(s,l,template);turn=s['turn'];reply=ep['steps'][turn-1]['response']
            cases=D.make_cases(s,l,f,template,ftemplate)
            prompts={c:next(r['prompt'] for r in cases if r['condition']==c) for c in CONDS}
            response=dict(arm=0,**tokenize_response(tok,reply,s['actions'][0]))
            groups.append(dict(id=pid,cluster=z['traj'],cohort='p4',turn=turn,actions=s['actions'],prompts=prompts,
                q=[q[pid][j]['q3456'] for j in range(5)],qgold=[q[pid][j]['qgold'] for j in range(5)],
                q12_comparator=z['swap_arm'],responses=[response]))
        assert len(groups)==386
        audit=dict(population_points=386,selected_points=386,skipped=[],original_sources=hashes)
    else:
        root=Path(args.e4_data);population=[];all_zero=0;errors=[]
        for file in sorted((root/'points').glob('*.json')):
            x=C.read_json(file)
            if not x['cf_defined']:continue
            sources[str(file.resolve())]=C.sha_file(file)
            rp=root/'replay'/('s'+str(x['seed']))/('t%02d.json'%x['turn'])
            assert rp.exists(),str(rp);r=source(rp)
            actions=[x['factual']]+x['alternatives']
            assert r['factual_action']==actions[0] and r['alternatives']==actions[1:]
            vals=[r['factual_outcomes']]+r['alt_outcomes']
            assert len(vals)==5 and all(len(v)==3 and all(a in (0,1) for a in v) for v in vals)
            q=[sum(v)/3 for v in vals];selected=max(q)>min(q);all_zero+=max(q)==0
            population.append(dict(id=x['point_id'],q=q,selected=selected))
            if not selected:continue
            assert x['refeed_div']==0 and len(x['arm_responses'])==5
            for arm in range(5):
                cc=[v for v in r['continuations'] if v['role']==arm+1]
                assert len(cc)==3 and {v['rep'] for v in cc}=={0,1,2}
                assert all(v['first_action']==actions[arm] and v.get('refeed_div',0)==0 for v in cc)
                assert all(int(v['won'])==vals[arm][v['rep']] for v in cc)
            ep=dict(task=x['task'],initial_observation=x['initial_obs'],steps=x['episode'],final_reward=x['final_reward'])
            assert ep['steps'][x['turn']-1]['response']==x['arm_responses'][0]
            seed=int(hashlib.sha256(f"P4-v1:30102026:{x['point_id']}".encode()).hexdigest()[:16],16)
            order=np.random.default_rng(seed).permutation(5).tolist()
            feedback=[dict(action=actions[j],response=x['peeks'][j]['obs']) for j in order]
            try:responses=[dict(arm=j,**tokenize_response(tok,text,actions[j])) for j,text in enumerate(x['arm_responses'])]
            except Exception as exc:
                errors.append(dict(id=x['point_id'],error=str(exc)));continue
            groups.append(dict(id=x['point_id'],cluster=x['game_file'],cohort='e4',turn=x['turn'],actions=actions,q=q,
                prompts=evidence_prompts(x['prompt'],ep,x['turn'],feedback),responses=responses))
        audit=dict(population_points=len(population),selected_points=sum(x['selected'] for x in population),
                   all_zero=all_zero,selection='max(q)>min(q), original three replays per arm',population=population,errors=errors)
        C.save_json(out/'selection_audit.json',audit)
        assert len(population)==6016 and not errors, 'E4 response audit failed; see selection_audit.json'
        assert len(groups)==audit['selected_points']==1586
    total_tokens=0
    for g in groups:
        assert len(g['q'])==5 and np.isfinite(g['q']).all()
        g['lengths']={c:len(prompt_ids(tok,g['prompts'][c]))+
                          max(len(r['ids']) for r in g['responses']) for c in CONDS}
        assert max(g['lengths'].values())<=32768,(g['id'],g['lengths'])
        total_tokens+=sum(len(r['ids']) for r in g['responses'])
        # Preserve undefined Gold as null rather than nonstandard JSON NaN.
        if 'qgold' in g:g['qgold']=[float(v) if np.isfinite(v) else None for v in g['qgold']]
        g['sha256']=hashlib.sha256(C.json_bytes(g)).hexdigest()
    out.mkdir(parents=True,exist_ok=True)
    file=out/'groups.jsonl.gz'
    if file.exists():assert C.decode_rows(file)[0]==groups,'Prepared dataset differs; use fresh out'
    else:C.encode_rows(file,groups)
    C.save_json(out/'source_hashes.json',sources)
    C.save_json(out/'prepare.json',dict(cohort=args.cohort,points=len(groups),responses=sum(len(g['responses']) for g in groups),
        tokens=total_tokens,estimated_student_hidden_gib=total_tokens*3584*4/2**30,
        groups_hash=C.decode_rows(file)[1],max_tokens=max(max(g['lengths'].values()) for g in groups),audit=audit,
        original_malformed_actions=[dict(id=g['id'],arm=r['arm'],method=r['action_span_method']) for g in groups
                                    for r in g['responses'] if r['action_span_method']!='original_action_tag'],
        tokenizer_files=tokenizer_files,protocol_sha256=C.sha_file(PROTOCOL),source_hashes_sha256=C.sha_file(out/'source_hashes.json'),
        tokenizer_vocab_sha256=hashlib.sha256(C.json_bytes(tok.get_vocab())).hexdigest(),
        chat_template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest()))
    print('PREPARED',args.cohort,len(groups),'points',total_tokens,'response tokens',flush=True)

def prepared(out,limit=0):
    out=Path(out);meta=C.read_json(out/'prepare.json');groups,h=C.decode_rows(out/'groups.jsonl.gz')
    assert meta['protocol_sha256']==C.sha_file(PROTOCOL)
    assert meta['source_hashes_sha256']==C.sha_file(out/'source_hashes.json')
    for path,digest in meta['tokenizer_files'].items():assert C.sha_file(path)==digest,path
    assert h==meta['groups_hash'] and len(groups)==meta['points']
    for g in groups:
        assert g['sha256']==hashlib.sha256(C.json_bytes({k:v for k,v in g.items() if k!='sha256'})).hexdigest()
    return (groups[:limit] if limit else groups),meta

def source_fingerprint():
    return {p.name:C.sha_file(p) for p in [Path(__file__),HERE/'token_diagnostic_backend.py',HERE/'token_diagnostic_analysis.py',PROTOCOL,
        HERE/'exp2_common.py',HERE/'exp1c_backend.py',HERE/'exp1c_precision.py']}

def cache_student(args):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    import token_diagnostic_backend as B
    out=Path(args.out);groups,meta=prepared(out,args.limit);tok=tokenizers(args)
    dest=out/('student_cache' if not args.limit else f'student_cache_first{args.limit}')
    dest.mkdir(exist_ok=True)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        scorer,model=B.load_model('7b',args.student,ROOT)
        fingerprint=dict(groups_hash=meta['groups_hash'],limit=args.limit,source=source_fingerprint(),model=model,runtime=B.runtime())
        if (dest/'run.json').exists():
            prev=C.read_json(dest/'run.json');assert all(prev[k]==v for k,v in fingerprint.items()),'Student cache resume mismatch'
        C.save_json(dest/'run.json',dict(**fingerprint,complete=False))
        gates=B.numerical_gate(scorer,groups,tok,['M'],dest/'numerical_gate.json')
        assert all(g['passed'] for g in gates),'Student full-vocabulary gate failed'
        manifest={};start=time.time()
        for i,g in enumerate(groups):
            file=dest/(g['id']+'.npz');side=dest/(g['id']+'.json')
            if file.exists():
                s=C.read_json(side);assert s['group_sha256']==g['sha256'] and s['file_sha256']==C.sha_file(file)
                manifest[g['id']]=s;continue
            pre=prompt_ids(tok,g['prompts']['M'])
            kv,last=B.prefix(scorer,pre);arrays={}
            for r in g['responses']:
                h=B.response_hidden(scorer,kv,last,r['ids'])
                arrays['h'+str(r['arm'])]=h.cpu().numpy().copy()
                arrays['lp'+str(r['arm'])]=B.sampled_logps(scorer.model.lm_head,h,r['ids'])
                arrays['ids'+str(r['arm'])]=np.asarray(r['ids'],dtype=np.int64);del h
            del kv,last
            tmp=file.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez(f,**arrays)
            tmp.replace(file)
            s=dict(group_sha256=g['sha256'],file_sha256=C.sha_file(file),bytes=file.stat().st_size)
            C.save_json(side,s);manifest[g['id']]=s
            C.save_json(dest/'progress.json',dict(points=i+1,total=len(groups),seconds=time.time()-start))
            if (i+1)%10==0:print('STUDENT CACHE',args.cohort,i+1,'/',len(groups),flush=True)
        C.save_json(dest/'manifest.json',manifest)
        C.save_json(dest/'run.json',dict(**fingerprint,complete=True,n=len(groups),manifest_sha256=C.sha_file(dest/'manifest.json')))

def summarize(values,kinds):
    result={}
    for kind in ('all',)+KINDS:
        indices=[i for i,k in enumerate(kinds) if kind=='all' or k==kind]
        result[kind]=dict(n=len(indices),**{key:dict(sum=float(np.asarray(v)[indices].sum()),
            mean=float(np.asarray(v)[indices].mean()) if indices else None) for key,v in values.items()})
    return result

def score(args):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    import token_diagnostic_backend as B
    out=Path(args.out);groups,meta=prepared(out,args.limit);tok=tokenizers(args)
    cohort='full' if not args.limit else f'first{args.limit}'
    cached=out/('student_cache' if not args.limit else f'student_cache_first{args.limit}')
    cm=C.read_json(cached/'run.json');assert cm['complete'] and cm['n']==len(groups)
    assert cm['groups_hash']==meta['groups_hash'] and cm['source']==source_fingerprint()
    assert C.sha_file(cached/'manifest.json')==cm['manifest_sha256']
    cache_manifest=C.read_json(cached/'manifest.json')
    dest=out/args.name/cohort;dest.mkdir(parents=True,exist_ok=True);records=dest/'records';records.mkdir(exist_ok=True)
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        model_path=args.student if args.name=='7b' else args.teacher
        scorer,model=B.load_model(args.name,model_path,ROOT)
        fingerprint=dict(groups_hash=meta['groups_hash'],limit=args.limit,source=source_fingerprint(),model=model,
                         runtime=B.runtime(),student_cache_manifest_sha256=cm['manifest_sha256'])
        if (dest/'run.json').exists():
            prev=C.read_json(dest/'run.json');assert all(prev[k]==v for k,v in fingerprint.items()),'Teacher resume mismatch'
        C.save_json(dest/'run.json',dict(**fingerprint,complete=False))
        gates=B.numerical_gate(scorer,groups,tok,CONDS,dest/'numerical_gate.json')
        assert all(g['passed'] for g in gates),'Teacher full-vocabulary gate failed'
        sh=scorer.model.lm_head if args.name=='7b' else B.student_head(args.student,cm['model']['weights'],scorer.device)
        manifest={};start=time.time()
        for i,g in enumerate(groups):
            path=records/(g['id']+'.json.gz')
            if path.exists():
                result=json.loads(gzip.decompress(path.read_bytes()));assert result['group_sha256']==g['sha256']
                manifest[g['id']]=C.sha_file(path);continue
            cp=cached/(g['id']+'.npz');assert C.sha_file(cp)==cache_manifest[g['id']]['file_sha256']
            arrays=np.load(cp,allow_pickle=False);rows=[]
            for condition in CONDS:
                pre=prompt_ids(tok,g['prompts'][condition])
                kv,last=B.prefix(scorer,pre)
                for r in g['responses']:
                    arm=str(r['arm']);assert arrays['ids'+arm].tolist()==r['ids']
                    th=B.response_hidden(scorer,kv,last,r['ids'])
                    values=B.signals(scorer.model.lm_head,th,sh,torch.from_numpy(arrays['h'+arm]),r['ids'],arrays['lp'+arm])
                    if args.name=='7b' and condition=='M':
                        assert max(abs(v) for v in values['delta'])<1e-4 and max(abs(v) for v in values['reverse_kl'])<1e-5
                    rows.append(dict(condition=condition,arm=r['arm'],token_ids=r['ids'],kinds=r['kinds'],
                                     values=values,summary=summarize(values,r['kinds'])))
                    del th
                del kv,last
            arrays.close()
            result=dict(id=g['id'],group_sha256=g['sha256'],rows=rows)
            raw=gzip.compress(C.json_bytes(result),mtime=0);tmp=path.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(path)
            manifest[g['id']]=C.sha_file(path)
            C.save_json(dest/'progress.json',dict(points=i+1,total=len(groups),responses=(i+1)*len(g['responses'])*4,seconds=time.time()-start))
            if (i+1)%5==0:print('TOKEN SCORE',args.cohort,args.name,i+1,'/',len(groups),flush=True)
        C.save_json(dest/'manifest.json',manifest)
        C.save_json(dest/'run.json',dict(**fingerprint,complete=True,points=len(groups),
            responses=sum(len(g['responses'])*4 for g in groups),seconds=time.time()-start,manifest_sha256=C.sha_file(dest/'manifest.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','cache','score','analyze','verify'])
    p.add_argument('--cohort',choices=['p4','e4'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--student',default='/root/autodl-tmp/models/Qwen2.5-7B-Instruct')
    p.add_argument('--teacher',default='/root/autodl-tmp/models/Qwen2.5-32B-Instruct')
    p.add_argument('--e4-data',default='/root/autodl-tmp/idea/exploratory_E4/data')
    p.add_argument('--arms',default='/root/autodl-tmp/idea/exploratory_deep_discovery/exploratory_arms.parquet')
    p.add_argument('--name',choices=['7b','32b'],default='32b');p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();assert a.limit>=0
    if a.command=='cache':cache_student(a)
    elif a.command in ['analyze','verify']:
        import token_diagnostic_analysis as analysis
        getattr(analysis,a.command)(a)
    else:globals()[a.command](a)

if __name__=='__main__':main()
