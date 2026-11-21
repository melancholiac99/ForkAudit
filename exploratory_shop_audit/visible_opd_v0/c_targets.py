"""Build detached first-update targets and the common student-only batch."""
from collections import Counter
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from .b_collect import PARENT
from .c_collect import RUN, STAGE_B, tasks
from .projection import VisibleProjection, candidate_target
from .content_target import content_correction
from .score import support
from exploratory_shop_audit.grpo_forced_audit.common import MODEL, digest_value, read, read_rows, save, stable_seed

ARMS = ('N','R','P','R-beta','P-shuffled')
SEED = 20261120


def actor(dtype=torch.bfloat16, *, base_path=MODEL, load_on_rank=False):
    torch.manual_seed(SEED)
    mapping = {'': torch.cuda.current_device()} if load_on_rank else None
    base = AutoModelForCausalLM.from_pretrained(base_path,torch_dtype=dtype,local_files_only=True,device_map=mapping).cuda()
    config = LoraConfig(r=64,lora_alpha=64,target_modules='all-linear',lora_dropout=0.,bias='none',task_type='CAUSAL_LM')
    model = get_peft_model(base,config)
    assert not base.lm_head.weight.requires_grad
    return model


def response(model, prompt, answer):
    ids = torch.tensor([prompt+answer], device='cuda')
    base = model.get_base_model()
    h = base.model(input_ids=ids,use_cache=False).last_hidden_state[0,len(prompt)-1:-1]
    logits = base.lm_head(h).float()
    assert len(h)==len(answer)
    return h,logits


def projections(model, folds):
    nets = {}
    for fold in set(folds.values()):
        for label in ('content','shuffled'):
            net = VisibleProjection(model.config.hidden_size).cuda().eval()
            net.load_state_dict(torch.load(STAGE_B/'projector_versions'/f'{label}_{fold}.pt',weights_only=True))
            nets[label,fold] = net.requires_grad_(False)
    return nets


def visible_targets(h, head, ids, s, mask, q, nets, fold, cal, classes):
    result = {'N':q}
    rows = head[ids]
    for label,arm,key in [('content','P','b'),('shuffled','P-shuffled','gamma')]:
        coeff = [cal[c][key] if key=='b' else cal[c][key]['scale'] for c in classes]
        delta = nets[label,fold](h,rows,q)
        result[arm] = candidate_target(s,delta,torch.tensor(coeff,device=s.device)[:,None],mask)
    return result


@torch.no_grad()
def build():
    os.umask(0o077)
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    start = time.monotonic()
    selected = tasks()
    assignment = {r['goal_index']:i%4 for i,r in enumerate(selected)}
    cal_folds = read(PARENT/'asin_split_manifest.json')['cal_folds']
    folds = {r['goal_index']:cal_folds.get(r['asin'],stable_seed(SEED,'projection_fold',r['asin'])%4) for r in selected}
    model = actor().eval()
    nets = projections(model,folds)
    cal = read(STAGE_B/'reports/calibration_strength_report.json')['classes']
    records = []
    counts = Counter()
    terminal = Counter()
    for rep in range(2):
        for row in read_rows(RUN/f'rollout_{rep}.jsonl'):
            g = row['goal']
            if assignment[g]!=rank:continue
            key = (g,rep)
            assert row['rep']==rep and row['step']==counts[key] and not terminal[key]
            counts[key]+=1
            terminal[key]+=int(row['done'])
            assert not any(row['na'].values())
            prompt,answer = row['views']['none'],row['response']
            h,logits = response(model,prompt,answer)
            ids,mask = support(logits,torch.tensor(answer,device='cuda'))
            s = logits.gather(1,ids)
            q = candidate_target(s,torch.zeros_like(s),1,mask)
            targets = visible_targets(h,model.get_base_model().lm_head.weight,ids,s,mask,q,
                                      nets,folds[g],cal,row['classes'])
            def subset(view):
                _,raw = response(model,row['views'][view],answer)
                return raw.log_softmax(-1).gather(1,ids)
            d = content_correction(subset('full'),subset('placeholder'),q)
            beta = torch.tensor([cal[c]['beta']['scale'] for c in row['classes']],device='cuda')[:,None]
            targets['R'] = candidate_target(s,d,1.,mask)
            targets['R-beta'] = candidate_target(s,d,beta,mask)
            for t in targets.values():
                assert not t.requires_grad and torch.isfinite(t).all() and (t[~mask]==0).all()
            records.append((dict(goal=g,rep=rep,step=row['step'],prompt=prompt,answer=answer,
                                 ids=ids.cpu(),mask=mask.cpu(),classes=row['classes'],snapshot=q.cpu()),targets))
            records[-1] = (records[-1][0],{k:t.cpu() for k,t in targets.items()})
    assert len(terminal)==16 and set(terminal.values())=={1}
    shard = RUN/f'batch_{rank}'
    shard.mkdir(exist_ok=True)
    torch.save([r for r,t in records],shard/'student.pt')
    for arm in ARMS:
        torch.save([t[arm] for r,t in records],shard/f'{arm}.pt')
    save(shard/'manifest.json',dict(rank=rank,tasks=8,trajectories=16,steps=len(records),
        positions=sum(len(r['answer']) for r,_ in records),gpu_hours=(time.monotonic()-start)/3600,
        support_sha256=digest_value([(r['goal'],r['rep'],r['step'],r['ids'].tolist(),r['mask'].tolist()) for r,_ in records])))

if __name__ == '__main__':
    build()
