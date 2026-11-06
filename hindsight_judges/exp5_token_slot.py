"""P4 causal native-token and separate counterfactual action-probe slot diagnostic."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import re
import time
import numpy as np
import exp2_common as C
import exp3_action_diagnostic as D
import exp3_token_diagnostic as OLD
import token_diagnostic_backend as B
import token_slot_backend as K
import slot_model_loader as L

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
OUT=HERE/'results/token_slot_20261106'
PROTOCOL=HERE/'PROTOCOL_NEXT_TOKEN_SLOT_AND_MATH_20261106.md'


def fingerprint():
    return {p.name:C.sha_file(p) for p in [Path(__file__),HERE/'token_slot_backend.py',HERE/'token_slot_analysis.py',
        HERE/'test_token_slot.py',HERE/'slot_model_loader.py',HERE/'TOKEN_SLOT_EXECUTION_20261106.txt',PROTOCOL,HERE/'exp3_action_diagnostic.py',
        HERE/'exp3_token_diagnostic.py',HERE/'token_diagnostic_backend.py',HERE/'exp2_common.py',HERE/'exp1c_backend.py',HERE/'exp1c_precision.py']}


def safe_out(path):
    path=Path(path).resolve()
    assert path==OUT or OUT in path.parents or Path('/tmp') in path.parents,'Dedicated new experiment path required'
    return path


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(gzip.compress(C.json_bytes(obj),mtime=0));tmp.replace(path)


def read(path):return json.loads(gzip.decompress(Path(path).read_bytes()))


def response_boundaries(tok,response,actions):
    text=response['text']; ids=response['ids']; encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ids
    aligned={}
    for t in range(1,len(ids)):
        h=tok.decode(ids[:t],clean_up_tokenization_spaces=False)
        if text.startswith(h):aligned[len(h)]=t
    think=list(re.finditer(r'<think>(.*?)</think>',text,re.S|re.I));assert think
    intervals=[(m.start(1),m.end(1)) for m in think]
    words=[];boundaries=[];paragraphs=[];unaligned=[]
    for lo,hi in intervals:
        for m in re.finditer(r'\b[\w]+\b',text[lo:hi]):
            end=lo+m.end()
            if end in aligned:words.append(aligned[end])
        ends={lo+m.end() for m in re.finditer(r'(?<!\d)[.!?。！？](?=\s|$)\s*|\n+',text[lo:hi])}
        ends.add(hi)
        for end in sorted(ends):
            after=[(pos,t) for pos,t in aligned.items() if pos>=end]
            assert after
            pos,t=min(after)
            boundaries.append(dict(t=t,char=end,native_alignment_char=pos,probe_reencoded=end not in aligned))
            if end not in aligned:unaligned.append(end)
        # Paragraph beginnings/end, with the nearest legal whole-word boundary inside it.
        for m in re.finditer(r'[^\n]+(?:\n(?!\n)[^\n]+)*',text[lo:hi]):
            candidates=[t for pos,t in aligned.items() if lo+m.start()<pos<=lo+m.end() and t in words]
            if candidates:paragraphs.extend([min(candidates),max(candidates)])
    action_positions=[i for i,k in enumerate(response['kinds']) if k=='action']
    assert action_positions
    pre=D.student_prefix(text,'keep'); lead,tail=D.action_format(text,actions[0])
    start=len(pre)+len(lead); end=start+len(actions[0]);assert text[start:end]==actions[0]
    variants=[tok.encode(text[:start]+a+text[end:],add_special_tokens=False) for a in actions]
    distinguish=next((i for i in range(len(ids)) if any(i>=len(v) or v[i]!=ids[i] for v in variants)),None)
    mentions=[]
    # Conservative literal action mention only; never claim a semantic localization when ambiguous.
    matches=list(re.finditer(re.escape(actions[0]),text[:len(pre)],re.I))
    matches=[m for m in matches if any(lo<=m.start()<m.end()<=hi for lo,hi in intervals)]
    if len(matches)==1:
        m=matches[0]
        before=[t for pos,t in aligned.items() if pos<=m.start() and t in words]
        after=[t for pos,t in aligned.items() if pos>=m.end() and t in words]
        mentions=([max(before)] if before else [])+([min(after)] if after else [])
    return dict(boundaries=sorted({b['char']:b for b in boundaries}.values(),key=lambda b:b['char']),
                words=sorted(set(words)),paragraph_positions=sorted(set(paragraphs)),
                mention_positions=mentions,mention_status='unique_literal_action' if len(matches)==1 else 'unresolved',
                action_first=action_positions[0],action_distinguish=distinguish,
                unaligned_sentence_characters=unaligned)


def stable_k(curve):
    for i,item in enumerate(curve):
        if all(x['prob'][0]>=.90 for x in curve[i:]):return item['t']
    return None


def phase(t,k):return 'no_k' if k is None else 'pre_k' if t<k else 'at_or_after_k'


def native_positions(g,k):
    b=g['boundary_rules']; reasons={}
    def add(t,why):
        if t is not None and 0<=t<len(g['response']['ids']):reasons.setdefault(t,[]).append(why)
    def choose(xs,n):
        if not xs:return []
        return [xs[i] for i in sorted(set(np.linspace(0,len(xs)-1,min(n,len(xs)),dtype=int)))]
    if k is None:
        for t in choose(b['words'],6):add(t,'ordinary_no_k')
    else:
        for side,xs in [('pre',[t for t in b['words'] if t<k]),('post',[t for t in b['words'] if t>=k])]:
            for t in choose(xs,3):add(t,'ordinary_'+side)
    if b['paragraph_positions']:
        add(b['paragraph_positions'][0],'first_paragraph');add(b['paragraph_positions'][-1],'last_paragraph')
    for t in b['mention_positions']:add(t,'literal_action_mention')
    add(b['action_first'],'action_first');add(b['action_distinguish'],'action_discriminating')
    return [dict(t=t,reasons=why,kind=g['response']['kinds'][t],phase=phase(t,k)) for t,why in sorted(reasons.items())]


def probe_job(tok,g,boundary):
    h=g['response']['text'][:boundary['char']]
    assert g['response']['text'].startswith(h)
    if len(re.findall(r'<think>',h,re.I))>len(re.findall(r'</think>',h,re.I)):h+='</think>\n'
    original=D.student_prefix(g['response']['text'],'keep');tag=list(D.OPEN_ACTION.finditer(original))[-1].group(0)
    lead,tail=D.action_format(g['response']['text'],g['actions'][0])
    row=dict(prompt=g['prompts']['M'],reply_prefix=h+tag,action_lead=lead,action_tail=tail,actions=g['actions'])
    job=D.tokenize(tok,row);base=B.prompt_ids(tok,g['prompts']['M'])
    assert job['prefix_ids'][:len(base)]==base,'Unexpected chat/response BPE boundary'
    return job['prefix_ids'][len(base):],[c['suffix_ids'] for c in job['candidates']]


def prepare(args):
    from transformers import AutoTokenizer
    out=safe_out(args.out);out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,use_fast=True)
    original=HERE/'results/token_diagnostic_p4'; groups,meta=OLD.prepared(original)
    sources={str(original/f):C.sha_file(original/f) for f in ('groups.jsonl.gz','prepare.json','source_hashes.json')}
    sources.update(C.read_json(original/'source_hashes.json'))
    for path,sha in sources.items():assert C.sha_file(path)==sha,path
    new=[]
    for g in groups:
        r=g['responses'][0]
        rebuilt=OLD.tokenize_response(tok,r['text'],g['actions'][0]);assert rebuilt['ids']==r['ids'] and rebuilt['kinds']==r['kinds']
        row=dict(id=g['id'],cluster=g['cluster'],q=g['q'],actions=g['actions'],response=r,
                 prompts={c:g['prompts'][c] for c in ('M','F')},boundary_rules=response_boundaries(tok,r,g['actions']))
        for b in row['boundary_rules']['boundaries']:probe_job(tok,row,b)
        row['sha256']=K.digest(row);new.append(row)
    assert len(new)==386 and len({g['cluster'] for g in new})==60
    restart={kind:sorted(set(tok.encode(s,add_special_tokens=False)[0] for s in strings)) for kind,strings in K.RESTART_STRINGS.items()}
    prep=dict(points=386,clusters=60,source=fingerprint(),sources=sources,tokenizer=dict(vocab=K.digest(tok.get_vocab()),template=K.digest(tok.chat_template)),
              restart_ids=restart,restart_strings=K.RESTART_STRINGS,probe_boundaries=sum(len(g['boundary_rules']['boundaries']) for g in new),
              no_probe_points=[g['id'] for g in new if not g['boundary_rules']['boundaries']])
    path=out/'groups.jsonl.gz'
    if path.exists():assert C.decode_rows(path)[0]==new and C.read_json(out/'prepare.json')['source']==fingerprint()
    else:C.encode_rows(path,new)
    prep['groups_hash']=C.decode_rows(path)[1];C.save_json(out/'prepare.json',prep)
    print('PREPARED',386,'points',prep['probe_boundaries'],'counterfactual boundaries; no-probe',prep['no_probe_points'],flush=True)


def prepared(out,limit):
    out=safe_out(out);meta=C.read_json(out/'prepare.json');groups,h=C.decode_rows(out/'groups.jsonl.gz')
    assert meta['source']==fingerprint() and h==meta['groups_hash'] and len(groups)==386
    for p,sha in meta['sources'].items():assert C.sha_file(p)==sha,p
    return groups[:limit] if limit else groups,meta


def folder(args):return safe_out(args.out)/args.name/('first32' if args.limit else 'full')


def score(args):
    import torch
    from transformers import AutoTokenizer
    from torch.nn.attention import sdpa_kernel,SDPBackend
    out=safe_out(args.out);groups,meta=prepared(out,args.limit);dest=folder(args);dest.mkdir(parents=True,exist_ok=True)
    pin=os.environ.get('CUDA_VISIBLE_DEVICES','');assert pin.startswith('GPU-') and ',' not in pin and torch.cuda.device_count()==1
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,use_fast=True)
    assert meta['tokenizer']==dict(vocab=K.digest(tok.get_vocab()),template=K.digest(tok.chat_template))
    refs=None
    if args.name=='32b':
        refdir=out/'7b'/dest.name; ref_run=C.read_json(refdir/'run.json')
        assert ref_run['complete'] and ref_run['fingerprint']['source']==fingerprint()
        refs=C.read_json(refdir/'manifest.json');assert C.sha_file(refdir/'manifest.json')==ref_run['manifest_sha256']
    if not args.limit:
        for name in ('7b','32b'):
            pilot=out/name/'first32';v=C.read_json(pilot/'verification.json')
            assert v['passed'] and v['source']==fingerprint() and v['analysis_sha256']==C.sha_file(pilot/'analysis.json')
    with (dest/'score.lock').open('a') as lock,torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scorer,model=L.load_model(args.name,args.model,ROOT)
        fp=dict(source=fingerprint(),model=model,runtime=B.runtime(),gpu_uuid=pin,groups_hash=meta['groups_hash'],limit=args.limit,name=args.name,
                student_manifest_sha256=C.sha_file(refdir/'manifest.json') if refs else None)
        runfile=dest/'run.json'
        if runfile.exists():
            old=C.read_json(runfile);assert old['fingerprint']==fp,'Resume mismatch'
            if old['complete']:print('Already complete');return
        if not args.limit:
            pilot_fp=C.read_json(out/args.name/'first32/run.json')['fingerprint']
            for k in ('source','model','runtime','gpu_uuid','groups_hash','name'):assert pilot_fp[k]==fp[k],k
        C.save_json(runfile,dict(fingerprint=fp,complete=False))
        gates=K.gate(scorer,tok,groups);C.save_json(dest/'numerical_gate.json',gates)
        assert all(x['passed'] for x in gates),'Full vocabulary TV gate failed'
        manifest={};start=time.monotonic();fresh=0
        for i,g in enumerate(groups):
            file=dest/'records'/(g['id']+'.json.gz')
            if file.exists():
                rec=read(file);assert rec['fingerprint_sha256']==K.digest(fp) and rec['group_sha256']==g['sha256']
                manifest[g['id']]=C.sha_file(file);continue
            tick=time.monotonic();engine=K.Engine(scorer,tok,g['prompts'])
            if args.name=='7b':
                curve=[]
                for boundary in g['boundary_rules']['boundaries']:
                    h,candidates=probe_job(tok,g,boundary);ll=[]
                    for ids in candidates:
                        ll.append(sum(float(engine.lp('A','M',h+ids[:j])[y]) for j,y in enumerate(ids)))
                    p=np.exp(np.array(ll)-max(ll));p/=p.sum()
                    curve.append(dict(**boundary,sequence_logp=ll,prob=p.tolist()))
                k=stable_k(curve);k_char=next((x['char'] for j,x in enumerate(curve) if all(v['prob'][0]>=.9 for v in curve[j:])),None);positions=native_positions(g,k)
                student={str(p['t']):float(engine.lp('A','M',g['response']['ids'][:p['t']])[g['response']['ids'][p['t']]]) for p in positions}
            else:
                reference=refdir/'records'/(g['id']+'.json.gz');assert C.sha_file(reference)==refs[g['id']]
                old=read(reference);assert old['group_sha256']==g['sha256']
                curve,k,k_char,positions,student=[old[key] for key in ('commitment_curve','k','k_char','positions','student_logp')]
            native=[];probes=[]
            for ni,pos in enumerate(positions):
                t=pos['t'];y=g['response']['ids'][t];q=engine.quartet(g['response']['ids'][:t])
                native.append(dict(**pos,y=y,student_logp=student[str(t)],**K.summaries(q,y,student[str(t)],meta['restart_ids'])))
                if i<2 and ni==0:
                    audit=dest/'vocab_audits';audit.mkdir(exist_ok=True)
                    np.savez_compressed(audit/(g['id']+'_native.npz'),**{key:value.cpu().numpy() for key,value in q.items()},y=np.array(y))
            for bi,boundary in enumerate(g['boundary_rules']['boundaries']):
                h,candidates=probe_job(tok,g,boundary);arms=[]
                for arm,ids in enumerate(candidates):
                    sums={};raw={key:0. for key in ('AF','AM','EF','EM')};tokens=[]
                    for j,y in enumerate(ids):
                        q=engine.quartet(h+ids[:j]);sample={};normalizers={}
                        for name,lp,z in K.targets(q):
                            value=float(lp[y]);sums[name]=sums.get(name,0.)+value;sample[name]=value;normalizers[name]=z
                        raw_token={key:float(v[y]) for key,v in q.items()}
                        for key,v in raw_token.items():raw[key]+=v
                        delta=(q['EF']-q['EM'])-(q['AF']-q['AM'])
                        support=torch.unique(torch.cat([torch.topk(q[key],min(128,len(q[key]))).indices for key in ('AF','EF')]))
                        tokens.append(dict(y=y,target_logp=sample,log_normalizers=normalizers,raw_logp=raw_token,
                            top128_contains_y=bool((support==y).any()),d_min=float(delta.min()),d_max=float(delta.max()),
                            d_outside_clip_fraction=float((delta.abs()>5).float().mean())))
                        if i<2 and bi==0 and arm==0 and j==0:
                            audit=dest/'vocab_audits';audit.mkdir(exist_ok=True)
                            np.savez_compressed(audit/(g['id']+'_probe.npz'),**{key:value.cpu().numpy() for key,value in q.items()},y=np.array(y))
                    arms.append(dict(arm=arm,ids=ids,sequence_logp=sums,raw_sequence_logp=raw,tokens=tokens))
                probes.append(dict(**boundary,phase=phase(boundary['char'],k_char),arms=arms))
            rec=dict(id=g['id'],cluster=g['cluster'],q=g['q'],group_sha256=g['sha256'],fingerprint_sha256=K.digest(fp),
                     commitment_curve=curve,k=k,k_char=k_char,positions=positions,student_logp=student,native=native,probes=probes,
                     requests=engine.audit,model_calls=engine.calls,seconds=time.monotonic()-tick)
            dump(file,rec);manifest[g['id']]=C.sha_file(file);fresh+=1
            C.save_json(dest/'progress.json',dict(points=i+1,total=len(groups),new_points=fresh,elapsed_seconds=time.monotonic()-start,
                        seconds_per_point=(time.monotonic()-start)/fresh,last_point_seconds=rec['seconds'],peak_vram_bytes=torch.cuda.max_memory_allocated()))
            print('SCORED',args.name,i+1,'/',len(groups),'seconds',round(rec['seconds'],2),flush=True);del engine
        C.save_json(dest/'manifest.json',manifest)
        C.save_json(runfile,dict(fingerprint=fp,complete=True,n=len(groups),manifest_sha256=C.sha_file(dest/'manifest.json'),
                    gate_sha256=C.sha_file(dest/'numerical_gate.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','score','analyze','verify'))
    p.add_argument('--out',default=str(OUT));p.add_argument('--name',choices=('7b','32b'),default='7b')
    p.add_argument('--model',default='/root/autodl-tmp/models/Qwen2.5-7B-Instruct');p.add_argument('--limit',type=int,choices=(0,32),default=0)
    args=p.parse_args()
    if args.command in ('analyze','verify'):
        import token_slot_analysis as A
        getattr(A,args.command)(args)
    else:globals()[args.command](args)


if __name__=='__main__':main()
