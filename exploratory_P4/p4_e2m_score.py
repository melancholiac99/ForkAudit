"""GPU-only resumable E2-M scoring (conditions M, MF). Same tokenisation, backend, gate and output format as
p4_e2_judge_score.py; only the request file, preparation file, output directory, fixed checks and expected count differ."""
from pathlib import Path
import argparse,gzip,hashlib,json,importlib.util,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,type=Path);ap.add_argument('--name',choices=['7b','32b'],required=True)
    ap.add_argument('--revision',required=True,help='Pinned upstream revision; record in DECISIONS before run')
    args=ap.parse_args()
    import torch
    if not torch.cuda.is_available():raise SystemExit('E2 blocked: no CUDA GPU. No model loaded and no scores written.')
    from transformers import AutoTokenizer
    target=OUT/f'e2m_{args.name}';target.mkdir(exist_ok=True)
    prep=json.loads((OUT/'e2m_preparation.json').read_text());raw=gzip.decompress((OUT/'e2m_requests.jsonl.gz').read_bytes())
    assert sha(raw)==prep['requests_sha256']
    rawjobs=[json.loads(s) for s in raw.decode().splitlines()]
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    def tokenize(job):
        prefix=tok.apply_chat_template([dict(role='user',content=job['text'])],tokenize=False,add_generation_prompt=True)+'<action>'
        pre=tok.encode(prefix,add_special_tokens=False)
        full=[tok.encode(prefix+a+'</action>'+tok.eos_token,add_special_tokens=False) for a in job['actions']]
        start=len(pre)
        for ids in full:
            i=0
            while i<min(len(pre),len(ids)) and pre[i]==ids[i]:i+=1
            start=min(start,i)
        assert len(pre)-start<=3 and max(map(len,full))<=32768
        assert all(ids[:start]==full[0][:start] and ids[-1]==tok.eos_token_id for ids in full)
        return dict(**job,prefix_ids=full[0][:start],candidates=[dict(suffix_ids=ids[start:]) for ids in full])
    source=OUT/'p4_e2_backend.py'
    spec=importlib.util.spec_from_file_location('p4_existing_scorer',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.MODEL=str(args.model)
    modelhash={}
    for p in sorted(args.model.glob('*.safetensors')):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        modelhash[p.name]=h.hexdigest()
    assert modelhash
    if args.name=='7b':
        expected={line.split()[1]:line.split()[0] for line in (ROOT/'env/qwen_sha256.txt').read_text().splitlines() if len(line.split())==2 and line.split()[1].endswith('.safetensors')}
        assert modelhash==expected,'7B checkpoint differs from frozen checkpoint'
    run=dict(model=str(args.model),revision=args.revision,model_sha256=modelhash,requests_sha256=prep['requests_sha256'],
        scorer_sha256=sha(source.read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),complete=False)
    manifest=target/'run.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        for k in ['revision','model_sha256','requests_sha256','scorer_sha256','script_sha256']:assert old[k]==run[k],k
    manifest.write_text(json.dumps(run,indent=2)+'\n')
    start=time.time();scorer=module.Scorer()
    print(json.dumps(dict(event='model_loaded',model=args.name,seconds=time.time()-start)),flush=True)
    # Fixed first context per condition plus longest text, independent of model output.
    checks=[];fixed=[next(j for j in rawjobs if j['condition']==c) for c in ['M','MF']]+[max(rawjobs,key=lambda j:len(j['text']))]
    for rawjob in fixed:
        j=tokenize(rawjob);a=scorer.score(j);b=[scorer.direct(j,k) for k in range(5)]
        def prob(lp):
            x=np.array([sum(y) for y in lp]);x=np.exp(x-x.max());return x/x.sum()
        tv=float(abs(prob(a)-prob(b)).sum()/2)
        checks.append(dict(request_id=j['request_id'],tv=tv,passed=tv<.005))
        print(json.dumps(dict(event='numerical_check',model=args.name,check=checks[-1])),flush=True)
    (target/'numerical_check.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert all(c['passed'] for c in checks),'Shared-prefix/direct score mismatch; do not proceed'
    done={};output=target/'scores.jsonl'
    if output.exists():
        for line in output.read_text().splitlines():
            row=json.loads(line);assert row['request_id'] not in done;done[row['request_id']]=row
    for j in rawjobs:
        if j['request_id'] in done:assert done[j['request_id']]['request_sha256']==j['request_sha256']
    assert set(done)<=set(j['request_id'] for j in rawjobs)
    with output.open('a',buffering=1) as f:
        for rawjob in rawjobs:
            if rawjob['request_id'] in done:continue
            j=tokenize(rawjob);lp=scorer.score(j);s=np.array([sum(v) for v in lp]);assert np.isfinite(s).all()
            p=np.exp(s-s.max());p/=p.sum()
            row={k:j[k] for k in ['request_id','request_sha256','point_id','traj_id','turn','condition']}
            row.update(sequence_logp=s.tolist(),candidate_prob=p.tolist(),argmax=int(p.argmax()))
            f.write(json.dumps(row)+'\n');done[j['request_id']]=row
            if len(done)%20==0:print(json.dumps(dict(completed=len(done),total=len(rawjobs),seconds=time.time()-start)),flush=True)
    run.update(complete=len(done)==1746,contexts=len(done),seconds=time.time()-start,gpu=torch.cuda.get_device_name(0),peak_gb=torch.cuda.max_memory_allocated()/1e9)
    manifest.write_text(json.dumps(run,indent=2)+'\n')
if __name__=='__main__':main()
