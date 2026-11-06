"""Causal slot contexts and FP32 full-vocabulary target construction."""
import copy
import hashlib
from collections import OrderedDict
import numpy as np
import exp2_common as C

REQUEST = 'Continue from the interruption. Output only the continuation.'
OPEN = '\n\n[PARTIAL DRAFT — context, not an environment observation]\n'
CLOSE = '\n[/PARTIAL DRAFT]'
BETAS = (.5, 1., 2.)
MAIN = 'T2_b1_clip5_full'
BASELINE = 'T1_b1'
RESTART_STRINGS = {
 'restart': ['Let', ' Let', 'First', ' First', "Let's", 'We need', 'To solve', 'Sure'],
 'rephrase': ['The problem', 'Given', 'In this', 'Here is', 'We are'],
 'format': ['<think>', '[think]', '```', '\n\n', '\\boxed', '$$'],
}


def digest(x): return hashlib.sha256(C.json_bytes(x)).hexdigest()


def contexts(tok, prompt, slot, hids):
    h = tok.decode(hids, clean_up_tokenization_spaces=False)
    assert '<|im_start|>' not in h and '<|im_end|>' not in h
    if slot == 'A':
        messages = [dict(role='user', content=prompt)]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stem = tok.encode(rendered, add_special_tokens=False)
        ids = stem + list(hids)
    else:
        messages = [dict(role='user', content=prompt + OPEN + h + CLOSE), dict(role='user', content=REQUEST)]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(rendered, add_special_tokens=False)
        # Cache only an exact invariant prefix before any variable quoted text.
        marker = 'PREFIX_SLOT_UNIQUE_SENTINEL_20261106'
        assert marker not in prompt
        template = tok.apply_chat_template([dict(role='user', content=prompt + OPEN + marker + CLOSE),
                    dict(role='user', content=REQUEST)], tokenize=False, add_generation_prompt=True)
        stem = tok.encode(template.split(marker)[0], add_special_tokens=False)[:-4]
        assert ids[:len(stem)] == stem
    assert 0 < len(ids) <= 32768
    return ids, stem, dict(messages=messages, active_prefix_ids=list(hids) if slot == 'A' else [],
                           causal_prefix_ids=list(hids), causal_prefix_text=h, input_ids=ids, slot=slot)


class Engine:
    def __init__(self, scorer, tok, prompts):
        self.s, self.tok, self.prompts = scorer, tok, prompts
        self.bases, self.memo, self.audit = {}, OrderedDict(), {}
        self.calls = 0

    def input(self, slot, evidence, hids):
        ids, stem, audit = contexts(self.tok, self.prompts[evidence], slot, hids)
        # Reuse only complete canonical prefill chunks. A partial cached chunk
        # changes GEMM/attention shapes versus _feed(ids), enough to fail FP16 TV.
        # The uncached remainder is replayed; rendered inputs/targets are identical.
        semantic_length = len(stem)
        stem = stem[:len(stem)//self.s.chunk*self.s.chunk]
        assert ids[:len(stem)] == stem
        audit.update(cache_prefix_tokens=len(stem), cache_chunk=self.s.chunk,
                     recomputed_prefix_tokens=semantic_length-len(stem),
                     cache_boundary_policy='canonical_complete_chunks')
        return ids, stem, audit

    def lp(self, slot, evidence, hids):
        import torch
        key = (slot, evidence, tuple(hids))
        if key in self.memo:
            self.memo.move_to_end(key)
            return torch.from_numpy(self.memo[key]).to(self.s.device)
        ids, stem, audit = self.input(slot, evidence, hids)
        bkey = (slot, evidence)
        if bkey not in self.bases:
            kv, last = self.s._feed(stem, len(stem)-1)
            self.bases[bkey] = (kv, last)
        base, last = self.bases[bkey]
        tail = ids[len(stem):]
        if tail:
            kv = copy.deepcopy(base)
            for start in range(0, len(tail), self.s.chunk):
                z = self.s.model.model(input_ids=self.s._ids(tail[start:start+self.s.chunk]), past_key_values=kv, use_cache=True)
                kv = z.past_key_values; last = z.last_hidden_state[0,-1:].float()
            del kv, z
        lp = torch.log_softmax(self.s.model.lm_head(last[-1].float()).float(), -1)
        assert torch.isfinite(lp).all()
        audit.update(evidence=evidence, full_logp_sha256=hashlib.sha256(lp.cpu().numpy().tobytes()).hexdigest())
        self.audit[digest([slot,evidence,list(hids)])] = audit
        self.memo[key] = lp.cpu().numpy().copy()
        if len(self.memo)>256: self.memo.popitem(last=False)
        self.calls += 1
        return lp

    def quartet(self, hids):
        return {s+e:self.lp(s,e,hids) for s in ('A','E') for e in ('F','M')}


def targets(q):
    import torch
    af,am,ef,em = [q[k].float() for k in ('AF','AM','EF','EM')]
    assert all(v.shape==af.shape and torch.isfinite(v).all() for v in (am,ef,em))
    d = (ef-em)-(af-am)
    union = torch.unique(torch.cat([torch.topk(af,min(128,len(af))).indices,torch.topk(ef,min(128,len(ef))).indices]))
    yield 'T0', af, 0.
    yield 'T3', ef, 0.
    for beta in BETAS:
        tag=f'b{beta:g}'
        logits=af+beta*(af-am); z=torch.logsumexp(logits,0)
        yield 'T1_'+tag, logits-z, float(z)
        for clip in (True,False):
            change=d.clamp(-5,5) if clip else d
            for support in ('full','top128'):
                logits=af+beta*change if support=='full' else af.clone()
                if support=='top128': logits[union]+=beta*change[union]
                z=torch.logsumexp(logits,0); lp=logits-z
                assert torch.isfinite(lp).all() and torch.isfinite(z)
                yield f'T2_{tag}_{"clip5" if clip else "raw"}_{support}',lp,float(z)


def summaries(q, y, student_lp, restart_ids):
    import torch
    af=q['AF']; d=(q['EF']-q['EM'])-(q['AF']-q['AM']); result={}
    for name,lp,z in targets(q):
        a=float((lp.exp()*(lp-af)).sum()); b=float((af.exp()*(af-lp)).sum())
        assert min(a,b)>=-1e-5 and np.isfinite([a,b,z]).all()
        result[name]=dict(logp=float(lp[y]),proxy=float(lp[y])-student_lp,logp_minus_AF=float(lp[y]-af[y]),
            kl_target_AF=a,kl_AF_target=b,tv_AF=float((lp.exp()-af.exp()).abs().sum()/2),log_normalizer=z,
            actual_token_rank=int((lp>lp[y]).sum())+1,top1=int(lp.argmax()),
            lexical_mass={k:float(lp[v].exp().sum()) for k,v in restart_ids.items()})
    return dict(targets=result, raw_sampled_logp={k:float(v[y]) for k,v in q.items()},
                d_min=float(d.min()),d_max=float(d.max()),d_outside_clip_fraction=float((d.abs()>5).float().mean()))


def gate(scorer,tok,groups):
    import torch
    checks=[]
    ranked=sorted(groups,key=lambda g:len(tok.encode(g['prompts']['F']))+len(g['response']['ids']))
    for g in (ranked[0],ranked[-1]):
        engine=Engine(scorer,tok,g['prompts']); h=g['response']['ids'][:-1]
        for slot in ('A','E'):
            for evidence in ('M','F'):
                cached=engine.lp(slot,evidence,h)
                ids,_,_=engine.input(slot,evidence,h)
                kv,last=scorer._feed(ids,len(ids)-1);del kv
                direct=torch.log_softmax(scorer.model.lm_head(last[-1].float()).float(),-1)
                tv=float((cached.exp()-direct.exp()).abs().sum()/2)
                checks.append(dict(point=g['id'],slot=slot,evidence=evidence,tv=tv,passed=tv<.005,kind='cached_vs_direct_full_vocab'))
        if g is ranked[0]:
            ids,_,_=engine.input('E','M',h)
            assert len(ids)<4096
            z=scorer.model.model(input_ids=scorer._ids(ids),use_cache=False)
            direct=torch.log_softmax(scorer.model.lm_head(z.last_hidden_state[0,-1].float()).float(),-1)
            tv=float((engine.lp('E','M',h).exp()-direct.exp()).abs().sum()/2)
            checks.append(dict(point=g['id'],tv=tv,passed=tv<.005,kind='cached_vs_onepass_full_vocab'));del z
        del engine
    return checks
