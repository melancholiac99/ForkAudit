"""Score current-actor candidates against a separately selected block adapter."""
from collections import Counter
import time

import torch

from .c_smoke import state_hash
from .c_targets import actor, response
from .d_data import SMOKE
from .e25_refresh import fold_of
from .e2512_routing import owners
from .gap_score import RUN, block_views, input_tapes, split_tasks, target
from .projection import VisibleProjection
from .score import support
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


def attach_actor(model, checkpoint, expected):
    path = checkpoint/'adapter_model.safetensors'
    assert digest_file(path)==expected
    model.load_adapter(str(checkpoint),adapter_name='online_actor',is_trainable=False)
    model.set_adapter('default')
    model.eval()


class BlockTargets:
    def __init__(self, model, projector=None):
        self.model = model
        self.adapters = set(model.peft_config)
        self.nets = {}
        self.scales = {}
        if projector is not None:
            info = read(projector/'calibration.json')
            self.scales = info['classes']
            for f in range(4):
                net = VisibleProjection(model.config.hidden_size).cuda().eval()
                p = projector/f'fold_{f}.pt'
                assert digest_file(p)==info['folds'][f]
                net.load_state_dict(torch.load(p, weights_only=True))
                self.nets[f] = net.requires_grad_(False)

    @torch.no_grad()
    def score(self, row, fold, fixed=None):
        model=self.model
        current=None
        if 'online_actor' in self.adapters:
            model.set_adapter('online_actor')
            _, raw=response(model,row['views']['none'],row['response'])
            ids, mask=fixed if fixed is not None else support(raw,torch.as_tensor(row['response'],device=raw.device))
            current=raw.gather(1,ids)
            del raw
            fixed=ids,mask
        model.set_adapter('default')
        v=block_views(model,row,fixed)
        current=v['s'] if current is None else current
        p_logits=None
        if self.nets:
            coef=torch.tensor([self.scales.get(c,0.) for c in row['classes']],device=current.device)[:,None]
            correction=self.nets[fold](v['h'],model.get_base_model().lm_head.weight[v['ids']],v['q'])
            p_logits=v['s']+coef*correction
        return v,current,p_logits


def init_check(rank):
    from torch.nn import functional as F
    from .e25_layout import Layout
    from .e25_refresh import tape
    from .gap_score import STAGE_B

    model=actor(load_on_rank=True).eval()
    initial=state_hash([(n,p) for n,p in model.named_parameters() if p.requires_grad])
    assert initial==read(SMOKE/'arm_R.json')['initial_adapter_sha256']
    initial_scorer=BlockTargets(model,RUN/'initial')
    old=Layout('formal_2512')
    tasks=split_tasks()['T_cal']
    routing=split_tasks()['cal_folds']
    subset=tasks[rank::4][:3]
    examples={}
    for path in sorted(STAGE_B.glob('tape_*.jsonl')):
        for r in tape(path):
            if r['goal'] in {x['goal_index'] for x in subset} and r['goal'] not in examples:
                examples[r['goal']]=r
    reference=[]
    for task in subset:
        g=task['goal_index'];fold=routing[task['asin']];r=examples[g]
        v,cur,proj=initial_scorer.score(r,fold)
        ids,mask=v['ids'],v['mask']
        with torch.no_grad():
            _,teacher=response(model,r['views']['full'],r['response'])
            direct=target(teacher.gather(1,ids),mask)
            net=initial_scorer.nets[fold]
            left=F.linear(F.normalize(v['h'].float(),dim=-1),net.state.weight)
            right=F.linear(F.normalize(model.get_base_model().lm_head.weight[ids].float(),dim=-1),net.token.weight)
            z=(left.unsqueeze(1)*right).sum(-1)
            scales=torch.tensor([initial_scorer.scales.get(c,0.) for c in r['classes']],device=z.device)[:,None]
            expected=target(v['s']+scales*z,mask)
            rerr=(v['teacher']-direct).abs().max().item()
            perr=(target(proj,mask)-expected).abs().max().item()
        assert rerr<=1e-6 and perr<=1e-6
        reference.append((task,r,ids,mask,cur,v['t'],proj,rerr,perr))
    sha=read(old.folder(10,'P')/'optimizer.json')['safetensors_sha256']
    attach_actor(model,old.state(10,'P')/'adapter',sha)
    scorer=BlockTargets(model,RUN/'initial')
    results=[]
    for task,r,ids,mask,cur,t,proj,re,pe in reference:
        v,alt,p=scorer.score(r,routing[task['asin']],(ids,mask))
        assert torch.equal(v['t'],t) and torch.equal(p,proj)
        results.append(dict(goal=task['goal_index'],positions=len(ids),teacher_error=re,projector_error=pe,
            fixed_R_logits_max_abs=0.,fixed_P_logits_max_abs=0.,actor_logits_max_abs=(alt-cur)[mask].abs().max().item()))
    assert any(x['actor_logits_max_abs']>0 for x in results)
    save(RUN/'checks'/f'rank{rank}.json',dict(prefixes=results,initial_sha=initial,alternate_actor_sha=sha))


def score_rank(layout,k,arm,rank):
    from .e25_refresh import tape
    path=layout.folder(k,arm)
    dst=path/'scored'/str(rank)
    if (dst/'receipt.json').exists():
        return
    ver=(k-1)//5*5
    model=actor(load_on_rank=True).eval()
    layout.snapshot(model,ver,arm)
    block_sha=state_hash([(n,p) for n,p in model.named_parameters() if p.requires_grad])
    initial=read(SMOKE/'arm_R.json')['initial_adapter_sha256']
    assert block_sha==(read(layout.folder(ver,arm)/'optimizer.json')['after_sha'] if ver else initial)
    current_sha=block_sha
    if k-1!=ver:
        m=read(layout.folder(k-1,arm)/'optimizer.json')
        attach_actor(model,layout.state(k-1,arm)/'adapter',m['safetensors_sha256'])
        current_sha=m['after_sha']
    fitter=layout.projector(ver) if arm=='P-gap' else None
    scorer=BlockTargets(model,fitter)
    splits=split_tasks()
    tasks={r['goal_index']:r['asin'] for r in read(path.parent/'tasks.json') if owners()[r['goal_index']]==rank}
    assert len(tasks)==8
    rows=[];counts=Counter();finished=set();na=[]
    for src in input_tapes(path):
        for r in tape(src):
            g=r['goal'];key=(g,r['rep'])
            if g not in tasks:continue
            assert r['step']==counts[key] and key not in finished
            counts[key]+=1
            if r['done']:finished.add(key)
            meta=dict(goal=g,rep=r['rep'],step=r['step'])
            if any(r['na'][key] for key in ('none','full')):
                na.append(meta);continue
            v,cur,proj=scorer.score(r,fold_of(tasks[g],splits['cal_folds']))
            teacher=v['teacher'] if proj is None else target(proj,v['mask'])
            # Strength uses the block student; optimizer assertions use the *current* actor.
            mask=v['mask']
            log_s=v['s'].masked_fill(~mask,-torch.inf).log_softmax(-1).masked_fill(~mask,0.)
            kl=(torch.special.xlogy(teacher,teacher)-teacher*log_s).sum(1)
            batchrow=dict(**meta,prompt=r['views']['none'],answer=r['response'],ids=v['ids'].cpu(),mask=mask.cpu(),
                          classes=r['classes'],snapshot=target(cur,mask).cpu())
            rows.append((batchrow,teacher.cpu(),cur.cpu(),kl.cpu()))
    assert finished=={(g,rep) for g in tasks for rep in range(2)}
    dst.mkdir(exist_ok=True,parents=True)
    torch.save(rows,dst/'rows.pt')
    save(dst/'receipt.json',dict(records=sum(counts.values()),na=na,goals=sorted(tasks),block_sha=block_sha,
         current_sha=current_sha,sha=digest_file(dst/'rows.pt')))


def merge(layout,k,arm):
    path=layout.folder(k,arm)
    combined=[];meta=[];na=[]
    for rank in range(4):
        sub=path/'scored'/str(rank)
        m=read(sub/'receipt.json')
        assert digest_file(sub/'rows.pt')==m['sha']
        meta.append(m);na.extend(m['na'])
        combined.extend(torch.load(sub/'rows.pt',weights_only=True,map_location='cpu'))
    key=lambda row:(row['goal'],row['rep'],row['step'])
    combined.sort(key=lambda r:key(r[0]))
    keys=[key(r[0]) for r in combined]+[key(r) for r in na]
    assert len(set(keys))==len(keys)==sum(m['records'] for m in meta)
    hashes={}
    for i,stem in enumerate(('student','target','bf16_before','strength')):
        p=path/f'{stem}.pt'
        torch.save([r[i] for r in combined],p)
        hashes[stem+'_sha256']=digest_file(p)
    assert len({m['block_sha'] for m in meta})==len({m['current_sha'] for m in meta})==1
    save(path/'target_manifest.json',dict(**hashes,goals=sorted(g for m in meta for g in m['goals']),na_records=na,
         rollout_records=len(keys),block_sha=meta[0]['block_sha'],scorer_sha=meta[0]['current_sha']))
