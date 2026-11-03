"""Full-vocabulary historical-response scoring, without training or top-k truncation."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import numpy as np
from exp2_common import read_json,sha_file,load_32b,save_json

HEAD_CHUNK=16

def prompt_ids(tokenizer,prompt):
    # Installed Transformers may return BatchEncoding for tokenize=True. Render then
    # encode explicitly, as in the frozen P4 backend, to obtain an actual flat ID list.
    text=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
    ids=tokenizer.encode(text,add_special_tokens=False)
    assert ids and all(isinstance(i,int) for i in ids)
    return ids

def load_model(name,path,repo):
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if name=='32b':return load_32b(Path(path),Path(repo))
    from exp1c_backend import ChunkedScorer
    original=read_json(Path(repo)/'exploratory_P4/e2_7b/run.json')
    weights={p.name:sha_file(p) for p in sorted(Path(path).glob('*.safetensors'))}
    assert weights==original['model_sha256'] and weights
    src=Path(repo)/'exploratory_P4/p4_e2_backend.py'
    assert sha_file(src)==original['scorer_sha256']
    spec=importlib.util.spec_from_file_location('token_frozen_7b',src)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.MODEL=str(path)
    frozen=module.Scorer();frozen.model.float()
    assert all(p.dtype==torch.float32 for p in frozen.model.parameters())
    return ChunkedScorer(frozen),dict(weights=weights,precision='7b_fp32_math',chunk=1024,backend_sha256=sha_file(src))

def runtime():
    import torch
    return dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,cuda=torch.version.cuda,
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip(),
        math=torch.backends.cuda.math_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),
        efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32,head_chunk=HEAD_CHUNK)

def prefix(scorer,ids):
    return scorer._feed(ids,len(ids)-1)

def response_hidden(scorer,cache,last,ids):
    """Hidden state predicting each token; never include that token in its own input."""
    t=scorer.t
    if len(ids)==1:return last[-1:]
    kv=copy.deepcopy(cache)
    out=scorer.model.model(input_ids=scorer._ids(ids[:-1]),past_key_values=kv,use_cache=False)
    h=t.cat([last[-1:],out.last_hidden_state[0].float()],0)
    del kv,out
    assert len(h)==len(ids)
    return h

def direct_hidden(scorer,pre,ids):
    cache,h=scorer._feed(pre+ids[:-1],len(pre)-1)
    del cache
    assert len(h)==len(ids)
    return h

def sampled_logps(head,h,ids):
    import torch
    result=[]
    for start in range(0,len(ids),HEAD_CHUNK):
        logp=torch.log_softmax(head(h[start:start+HEAD_CHUNK].float()).float(),-1)
        y=torch.tensor(ids[start:start+HEAD_CHUNK],device=h.device)
        result.extend(logp.gather(1,y[:,None]).squeeze(1).cpu().tolist())
    assert np.isfinite(result).all()
    return np.asarray(result,dtype=np.float32)

def distribution_tv(head,a,b):
    import torch
    assert a.shape==b.shape
    high=0.
    for start in range(0,len(a),HEAD_CHUNK):
        pa=torch.softmax(head(a[start:start+HEAD_CHUNK].float()).float(),-1)
        pb=torch.softmax(head(b[start:start+HEAD_CHUNK].float()).float(),-1)
        tv=(pa-pb).abs().sum(-1)/2
        assert torch.isfinite(tv).all()
        high=max(high,float(tv.max()))
    return high

def numerical_gate(scorer,groups,tokenizer,conditions,output_path=None):
    """Longest/shortest real context; full response, every token, entire vocabulary."""
    def pre(g,c):return prompt_ids(tokenizer,g['prompts'][c])
    ranked=[(g['lengths'][c],g,c) for g in groups for c in conditions]
    tests=[max(ranked,key=lambda x:x[0]),min(ranked,key=lambda x:x[0])]
    checks=[]
    for _,g,c in tests:
        ids=max(g['responses'],key=lambda r:len(r['ids']))['ids'];p=pre(g,c)
        cache,last=prefix(scorer,p);a=response_hidden(scorer,cache,last,ids);del cache,last
        b=direct_hidden(scorer,p,ids)
        tv=distribution_tv(scorer.model.lm_head,a,b)
        checks.append(dict(id=g['id'],condition=c,kind='cached_vs_chunked_direct_full_vocab',tokens=len(ids),max_tv=tv,passed=tv<.005))
        if output_path:save_json(output_path,checks)
        del a,b
    # An actual one-pass check on a short real context, separately from cached/chunked agreement.
    _,g,c=min(ranked,key=lambda x:x[0]);p=pre(g,c);ids=g['responses'][0]['ids']
    assert len(p)+len(ids)<=4096,'No short one-pass check available'
    a=direct_hidden(scorer,p,ids)
    out=scorer.model.model(input_ids=scorer._ids(p+ids[:-1]),use_cache=False)
    b=out.last_hidden_state[0,len(p)-1:].float()
    tv=distribution_tv(scorer.model.lm_head,a,b)
    checks.append(dict(id=g['id'],condition=c,kind='chunked_vs_onepass_full_vocab',tokens=len(ids),max_tv=tv,passed=tv<.005))
    if output_path:save_json(output_path,checks)
    return checks

def student_head(path,expected_weights,device):
    """Exactly the frozen BF16 checkpoint head converted to FP32, as in the cache run."""
    import torch
    from safetensors import safe_open
    index=read_json(Path(path)/'model.safetensors.index.json')
    shard=Path(path)/index['weight_map']['lm_head.weight']
    assert sha_file(shard)==expected_weights[shard.name]
    with safe_open(str(shard),framework='pt',device='cpu') as f:weight=f.get_tensor('lm_head.weight').float().to(device)
    head=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,device='meta')
    head.weight=torch.nn.Parameter(weight,requires_grad=False)
    return head

def signals(teacher_head,th,student_head,sh,ids,reference_logps):
    import torch
    values={k:[] for k in ('teacher_logp','student_logp','delta','reverse_kl','forward_kl')}
    assert len(th)==len(sh)==len(ids)==len(reference_logps)
    for start in range(0,len(ids),HEAD_CHUNK):
        end=start+HEAD_CHUNK
        tl=torch.log_softmax(teacher_head(th[start:end].float()).float(),-1)
        sl=torch.log_softmax(student_head(sh[start:end].to(th.device).float()).float(),-1)
        assert tl.shape==sl.shape and torch.isfinite(tl).all() and torch.isfinite(sl).all()
        target=torch.tensor(ids[start:end],device=th.device)[:,None]
        t=tl.gather(1,target).squeeze(1);s=sl.gather(1,target).squeeze(1)
        ref=torch.tensor(reference_logps[start:end],device=th.device)
        assert float((s-ref).abs().max())<1e-4,'Cached student projection mismatch'
        rev=(sl.exp()*(sl-tl)).sum(-1);fwd=(tl.exp()*(tl-sl)).sum(-1)
        assert torch.isfinite(rev).all() and torch.isfinite(fwd).all()
        assert float(rev.min())>=-1e-5 and float(fwd.min())>=-1e-5
        for key,v in zip(values,(t,s,t-s,rev,fwd)):values[key].extend(v.cpu().tolist())
    return values
