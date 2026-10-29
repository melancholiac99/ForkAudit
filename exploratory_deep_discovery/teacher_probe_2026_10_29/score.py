"""One-card, shared-prefix actual LLM teacher scoring. No outcome-file access."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import time
import numpy as np

OUT = Path(__file__).resolve().parent
MODEL = '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'


def sha(b): return hashlib.sha256(b).hexdigest()


class Scorer:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM
        self.t = torch
        self.model = AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        # Keep projection accumulation/output in FP32: BF16 vocabulary logits
        # quantize low-probability alternatives too coarsely for this comparison.
        self.model.lm_head.to(dtype=torch.float32)

    def score(self, job):
        t, m = self.t, self.model
        pre = job['prefix_ids']; cmds = [c['suffix_ids'] for c in job['candidates']]
        with t.inference_mode():
            out = m.model(input_ids=t.tensor([pre],device='cuda'),use_cache=True)
            lp0 = t.log_softmax(m.lm_head(out.last_hidden_state[0,-1].float()),-1)
            cache = out.past_key_values
            length = max(map(len,cmds))-1
            feed = t.zeros((len(cmds),length),dtype=t.long,device='cuda')
            attention = t.ones((len(cmds),len(pre)+length),dtype=t.long,device='cuda')
            for i,cmd in enumerate(cmds):
                feed[i,:len(cmd)-1] = t.tensor(cmd[:-1],device='cuda')
                attention[i,len(pre)+len(cmd)-1:] = 0
            kv = copy.deepcopy(cache); kv.batch_repeat_interleave(len(cmds))
            hidden = m.model(input_ids=feed,attention_mask=attention,past_key_values=kv,
                             use_cache=False).last_hidden_state
            rows=[]
            for i,cmd in enumerate(cmds):
                logp = t.log_softmax(m.lm_head(hidden[i,:len(cmd)-1].float()),-1)
                token_lp = [float(lp0[cmd[0]])] + logp[
                    t.arange(len(cmd)-1,device='cuda'),t.tensor(cmd[1:],device='cuda')].tolist()
                rows.append(token_lp)
            return rows

    def direct(self, job, arm):
        t,m=self.t,self.model
        pre=job['prefix_ids'];cmd=job['candidates'][arm]['suffix_ids'];ids=pre+cmd
        with t.inference_mode():
            h=m.model(input_ids=t.tensor([ids],device='cuda'),use_cache=False).last_hidden_state
            lp=t.log_softmax(m.lm_head(h[0,len(pre)-1:len(ids)-1].float()),-1)
            return lp[t.arange(len(cmd),device='cuda'),t.tensor(cmd,device='cuda')].tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-seconds',type=int,default=7200)
    ap.add_argument('--directory',type=Path,default=OUT)
    args=ap.parse_args()
    output_dir=args.directory
    prep=json.loads((output_dir/'preparation.json').read_text())
    content=(output_dir/'requests.jsonl').read_bytes()
    assert sha(content)==prep['request_file_sha256']
    assert sha((output_dir/'PROTOCOL.md').read_bytes())==prep['protocol_sha256']
    jobs=[json.loads(x) for x in content.decode().splitlines()]
    jobs.sort(key=lambda r:(r['cohort']!='development',r['point_id'],r['condition']))
    output=output_dir/'scores.jsonl';done={}
    if output.exists():
        for line in output.read_text().splitlines():
            row=json.loads(line);done[row['request_id']]=row
        for j in jobs:
            if j['request_id'] in done:assert done[j['request_id']]['request_sha256']==j['request_sha256']
    launch=time.time();scorer=Scorer();load_seconds=time.time()-launch
    print(json.dumps(dict(event='model_loaded',seconds=load_seconds,gpu=scorer.t.cuda.get_device_name(0))),flush=True)
    # Compare the measured candidate distributions, including one longest
    # context. TV bounds the change in any expected [0,1] task-success readout.
    if not (output_dir/'numerical_check.json').exists():
        checks=[]
        check_jobs=[next(j for j in jobs if j['condition']==condition)
                    for condition in ['S','H'] if any(j['condition']==condition for j in jobs)]
        if not check_jobs:check_jobs=[jobs[0]]
        check_jobs.append(max(jobs,key=lambda j:len(j['prefix_ids'])))
        for j in check_jobs:
            cached=scorer.score(j)
            directs=[]; token_checks=[]
            for arm in range(5):
                direct=scorer.direct(j,arm)
                directs.append(direct)
                diff=np.asarray(cached[arm])-np.asarray(direct)
                token_checks.append(dict(arm=arm,
                    max_token_difference=float(abs(diff).max()),sum_difference=float(diff.sum())))
            distributions=[]
            for arr in [cached,directs]:
                sums=np.array([sum(x) for x in arr]);prob=np.exp(sums-sums.max());prob/=prob.sum()
                distributions.append(prob)
            tv=float(abs(distributions[0]-distributions[1]).sum()/2)
            checks.append(dict(request_id=j['request_id'],token_checks=token_checks,
                cached_probability=distributions[0].tolist(),direct_probability=distributions[1].tolist(),
                total_variation=tv,pass_readout_bound=tv<.005))
        (output_dir/'numerical_check_attempt.json').write_text(json.dumps(checks,indent=2)+'\n')
        assert all(c['pass_readout_bound'] for c in checks),checks
        (output_dir/'numerical_check.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps(dict(event='numerical_check_passed',checks=checks)),flush=True)
    with output.open('a',buffering=1) as f:
        for j in jobs:
            if j['request_id'] in done:continue
            if time.time()-launch>args.max_seconds:
                raise TimeoutError('Fixed runtime ceiling reached; scores are resumable.')
            tic=time.time();lps=scorer.score(j);sums=np.array([sum(x) for x in lps])
            assert np.isfinite(sums).all()
            lz=float(np.max(sums)+np.log(np.exp(sums-np.max(sums)).sum()))
            assert lz<1e-4
            row=dict(request_id=j['request_id'],request_sha256=j['request_sha256'],
                point_id=j['point_id'],cohort=j['cohort'],condition=j['condition'],task_id=j['task_id'],
                prefix_tokens=len(j['prefix_ids']),sequence_logp=sums.tolist(),
                candidate_log_mass=lz,candidate_prob=np.exp(sums-lz).tolist(),
                candidate_token_logp=lps,seconds=time.time()-tic)
            f.write(json.dumps(row)+'\n');done[j['request_id']]=row
            if len(done)%8==0:
                progress=dict(completed=len(done),total=len(jobs),elapsed_seconds=time.time()-launch,
                    last_request=j['request_id'],peak_allocated_gb=scorer.t.cuda.max_memory_allocated()/1e9)
                (output_dir/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(progress),flush=True)
    env=dict(complete=len(done)==len(jobs),contexts=len(done),candidate_sequences=5*len(done),
        elapsed_seconds=time.time()-launch,model_load_seconds=load_seconds,
        peak_allocated_gb=scorer.t.cuda.max_memory_allocated()/1e9,
        peak_reserved_gb=scorer.t.cuda.max_memory_reserved()/1e9,
        gpu=scorer.t.cuda.get_device_name(0),model=MODEL,dtype='bfloat16',lm_head_dtype='float32',log_softmax_dtype='float32',
        versions={x:importlib.metadata.version(x) for x in ['torch','transformers','numpy']},
        request_file_sha256=prep['request_file_sha256'],score_code_sha256=sha(Path(__file__).read_bytes()))
    (output_dir/'run.json').write_text(json.dumps(env,indent=2)+'\n')
    print(json.dumps(env),flush=True)
    del scorer.model
    import gc;gc.collect();scorer.t.cuda.empty_cache()


if __name__=='__main__':main()
