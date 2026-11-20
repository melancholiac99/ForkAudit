"""Read-only full-vocabulary margin expansion on visible equivalent items."""
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from .b_collect import RUN,PARENT
from .b_score import forward,distribution
from .parser import categories
from .projection import VisibleProjection
from .b_metrics import Cluster
from exploratory_shop_audit.grpo_forced_audit.common import read,save,MODEL,append

@torch.inference_mode()
def audit():
    pool=read(RUN/'equivalent_pool.json');states=pool['states'];out=RUN/'reports'
    result=dict(coverage=pool['counts'],scope='T_cal ASIN-held-out, no V; visible candidate nomination before feasibility review')
    if not states:
        save(out/'equivalent_product_bias_report.json',dict(result,status='INSUFFICIENT_PAIRS',pairs=0));return
    start=time.monotonic()
    tape={}
    keys={(s['goal'],s['step']) for s in states}
    for p in RUN.glob('tape_*.jsonl'):
        for line in p.open():
            v=json.loads(line);k=v['goal'],v['step']
            if k in keys:tape[k]=v
    splits=read(PARENT/'asin_split_manifest.json')
    index={r['goal_index']:i for i,r in enumerate(splits['T_cal'])}
    fold={r['goal_index']:splits['cal_folds'][r['asin']] for r in splits['T_cal']}
    scales=read(out/'calibration_strength_report.json')['classes']
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    e=F.normalize(model.lm_head.weight.float(),dim=-1)
    pairs=[];strict=[];support_n=0;outside=0;no_factual_prefix=0
    for ff in range(4):
        net=VisibleProjection(e.shape[1]).cuda()
        net.load_state_dict(torch.load(RUN/'projector_versions'/f'content_{ff}.pt',weights_only=True));ev=net.token(e)
        for st in states:
            g,step=st['goal'],st['step']
            if fold[g]!=ff:continue
            r=tape[g,step];actions=[st['gold']]+st['alternatives']
            seq=[tok.encode('click['+a+']',add_special_tokens=False) for a in actions]
            views={m:forward(model,r['views'][m],seq)[0] for m in ('none','full','placeholder')}
            feat=torch.load(RUN/'features'/f'{index[g]:03d}_{step:02d}.pt',weights_only=True)
            vals=[]
            for j,ids in enumerate(seq):
                s,sel,ph=[distribution(model,views[m][j]) for m in ('none','full','placeholder')]
                cls=categories(tok,ids,{'page_type':r['page']})
                b=torch.tensor([scales[c]['b'] for c in cls],device='cuda')[:,None]
                raw=(s+sel-ph).log_softmax(-1)
                projected=(s+b*(net.state(F.normalize(views['none'][j].float(),dim=-1))@ev.T)).log_softmax(-1)
                ids_tensor=torch.tensor(ids,device='cuda')[:,None]
                vals.append({k:v.gather(1,ids_tensor).sum().item() for k,v in [('student',s),('content',raw),('projected',projected),('direct_sel_minus_ph',sel-ph)]})
                for t,v in enumerate(ids):
                    if t>=len(r['response']) or ids[:t]!=r['response'][:t]:no_factual_prefix+=1;continue
                    support_n+=1;outside+=int(v not in feat['ids'][t][feat['mask'][t]].tolist())
            for j in range(1,len(seq)):
                metrics={}
                for key in vals[0]:
                    metrics[key+'_sum']=vals[0][key]-vals[j][key]
                    metrics[key]=vals[0][key]/len(seq[0])-vals[j][key]/len(seq[j])
                metrics['raw_margin_sel_minus_ph']=metrics['direct_sel_minus_ph']
                metrics['excess_R']=metrics['raw_margin_sel_minus_ph']-metrics['student']
                metrics['excess_P']=metrics['projected']-metrics['student']
                metrics['normalized_content_teacher_excess']=metrics['content']-metrics['student']
                metrics['R_minus_P']=metrics['raw_margin_sel_minus_ph']-metrics['projected']
                pairs.append((g,metrics))
                t=next((u for u,(x,y) in enumerate(zip(seq[0],seq[j])) if x!=y),None)
                if t is None or t>=len(r['response']) or seq[0][:t]!=r['response'][:t]:continue
                cid=feat['ids'][t][feat['mask'][t]].tolist()
                if seq[0][t] not in cid or seq[j][t] not in cid:continue
                a,b=cid.index(seq[0][t]),cid.index(seq[j][t]);sm=feat['s'][t];d=feat['d'][t]
                p=net(feat['h'][t:t+1].cuda(),e[feat['ids'][t:t+1].cuda()],feat['q'][t:t+1].cuda())[0].cpu()
                k=scales[feat['classes'][t]]['b']
                student=float(sm[a]-sm[b])
                direct=float(d[a]-d[b])
                projected=float((sm+k*p)[a]-(sm+k*p)[b])
                strict.append((g,dict(student=student,raw_margin_sel_minus_ph=direct,
                                     excess_R=direct-student,content=student+direct,
                                     projected=projected,excess_P=projected-student)))
    def aggregate(rs):
        if not rs:return dict(status='INSUFFICIENT_PAIRS',pairs=0)
        c=Cluster(np.array([r[0] for r in rs]))
        return dict(tasks=c.n,pairs=len(rs),metrics={k:c.estimate(np.array([r[1][k] for r in rs])) for k in rs[0][1]})
    save(out/'equivalent_product_bias_report.json',dict(result,interpretation='Report observed margins descriptively; very few eligible tasks do not establish general gold preference.',raw_definition='(sel gold-alt margin) minus (placeholder gold-alt margin); excess_R subtracts student margin. Also report full-vocabulary student-base content teacher separately.',full_vocabulary=aggregate(pairs),strict_candidate_subset=aggregate(strict),factual_prefix_positions=support_n,out_of_support_positions=outside,no_factual_prefix=no_factual_prefix))
    append(RUN/'b_cost.jsonl',dict(stage='bias_audit',gpu_hours=(time.monotonic()-start)/3600))
if __name__=='__main__':audit()
