"""Exact teacher-forced probability mass over complete native legal-click strings."""
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from .c_targets import actor, response
from .gap_score import RUN
from exploratory_shop_audit.grpo_forced_audit.common import MODEL, read, save
from exploratory_shop_audit.runtime import make_input


def legal_clicks(state):
    allowed={a.casefold() for a in state['available']['clickables']}
    return list(dict.fromkeys(a for a in state['public']['actions'] if a.casefold() in allowed and a.casefold()!='search'))


@torch.no_grad()
def masses(model,tok,folder,pool,goals,good):
    from .b_score import forward
    out=[]
    for g in goals:
        anchor=read(folder/f'{pool}_{g}.json');s=anchor['state']
        if s is None:
            out.append(dict(goal=g,mass=None,reason='NO_FIRST_RESULT_PAGE'))
            continue
        prompt=make_input(tok,s['task'],anchor['history'],s['observation'])['token_ids']
        clicks=legal_clicks(s)
        if not clicks:
            out.append(dict(goal=g,mass=None,reason='NO_LEGAL_CLICK'))
            continue
        actions=[tok.encode(f'Action: click[{a}]',add_special_tokens=False)+[tok.eos_token_id] for a in clicks]
        # backbone includes the active LoRA. b_score.forward expects model.model to be its decoder.
        base=model.get_base_model();scores=[]
        for offset in range(0,len(actions),8):
            part=actions[offset:offset+8]
            hidden,_=forward(base,prompt,part)
            for h,seq in zip(hidden,part,strict=True):
                ll=base.lm_head(h).float().log_softmax(-1)
                scores.append(ll.gather(1,torch.tensor(seq,device=ll.device)[:,None]).double().sum())
        probs=torch.stack(scores).softmax(0)
        mask=torch.tensor([a.casefold() in good[str(g)] for a in clicks],device=probs.device)
        out.append(dict(goal=g,mass=float(probs[mask].sum()),choices=len(clicks),qualifying_clicks=int(mask.sum())))
    return out


def bench(rank):
    from .e25_layout import Layout
    from .e25_refresh import tape
    from .gap_score import input_tapes
    from .e2512_routing import owners
    folder=RUN/'page_timing'/f'rank{rank}';info=read(folder/'timing.json')
    goals=info['goals']
    # The same private deterministic training pages, on both measurement paths.
    good={str(g):[] for g in goals}  # likelihood workload independent of membership
    start=time.monotonic()
    model=actor(load_on_rank=True).eval()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    torch.cuda.synchronize()
    ready=time.monotonic()
    rows=masses(model,tok,folder/'anchors','training_budget',goals,good)
    torch.cuda.synchronize()
    done=time.monotonic()
    # A complete historical update, with the live actor distinct from the block snapshot.
    l=Layout('formal_2512')
    l.snapshot(model,10,'P')
    recs=[r for p in input_tapes(l.folder(10,'P')) for r in tape(p) if owners()[r['goal']]==rank and
          not any(r['na'][mode] for mode in ('none','full'))]
    torch.cuda.synchronize()
    before=time.monotonic()
    with torch.no_grad():
        for r in recs:
            response(model,r['views']['none'],r['response'])
    torch.cuda.synchronize()
    save(folder/'hf_timing.json',dict(click_seconds=done-ready,click_choices=sum(v.get('choices',0) for v in rows),
         scored_pages=sum(v['mass'] is not None for v in rows),actor_seconds=time.monotonic()-before,
         actor_records=len(recs),startup_seconds=ready-start))


if __name__=='__main__':
    if sys.argv[1]=='bench':
        from .e25_probe import configure
        bench(configure())
