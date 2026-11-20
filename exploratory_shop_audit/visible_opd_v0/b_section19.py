"""Read-only CPU reclassification and descriptive format-position split."""
import json
import numpy as np
from transformers import AutoTokenizer

from .b_collect import RUN
from .b_fit import load_data
from .b_metrics import p1_status, stats_label, strength, summary
from .parser import command_parts
from exploratory_shop_audit.grpo_forced_audit.common import MODEL,read,save,digest_file


def describe():
    rep=RUN/'reports'
    out=RUN/'section19'
    frozen=read(rep/'acceptance.json')
    for name,h in frozen['reports'].items():
        assert digest_file(rep/(name+'.json'))==h
    proj=read(rep/'projection_report.json')['groups']
    power=read(rep/'power_injection_report.json')['groups']
    groups={}
    for name,v in proj.items():
        gain=v['baselines'][v['strongest']]['gain']['ci95']
        stable=v['stable_same_sign']
        detected=gain[0]>0 and (stable or 0)>=.7
        powered=power[name]['status']=='POWER_OK'
        groups[name]=dict(previous=v['P1'],revised=p1_status(detected,powered),strongest=v['strongest'],
                          gain_ci95=gain,stable_same_sign=stable,power_status=power[name]['status'])
    save(out/'P1_decision_v2.json',dict(post_hoc=True,groups=groups,original=read(rep/'P1_decision.json'),
        frozen_hashes=frozen['reports'],gpu_hours=0))

    ts,meta,_=load_data()
    data={k:v.numpy() if k in ('ids','mask') else v.numpy().astype(np.float64) for k,v in ts.items() if k!='h'}
    preds={}
    visited=np.zeros(len(meta['goal']),bool)
    for f in range(4):
        with np.load(RUN/f'fold_{f}.npz') as saved:
            ix=saved['idx']
            assert not visited[ix].any() and (meta['fold'][ix]==f).all()
            visited[ix]=True
            for k in saved.files:
                if not k.startswith('content'):continue
                a=saved[k]
                if k not in preds:preds[k]=np.zeros((len(visited),a.shape[1]),dtype=a.dtype)
                preds[k][ix]=a
    assert visited.all()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    labels={}
    for path in sorted(RUN.glob('tape_*.jsonl')):
        for line in path.open():
            row=json.loads(line)
            parts,m=command_parts(tok,row['response'])
            offset=0;sub=[]
            for b,c in zip(parts,row['classes'],strict=True):
                end=offset+len(b)
                verb=m is not None and end>m.start(1) and offset<m.end(1)
                sub.append(('verb' if verb else 'pure_format') if c=='format' else '')
                offset=end
            key=row['goal'],row['step']
            assert key not in labels
            labels[key]=(row['classes'],sub)
    starts=np.r_[0,np.flatnonzero((np.diff(meta['goal'])!=0)|(np.diff(meta['step'])!=0))+1,len(visited)]
    parts=[]
    for a,b in zip(starts,starts[1:]):
        cls,sub=labels.pop((meta['goal'][a],meta['step'][a]))
        assert cls==meta['cls'][a:b].tolist()
        parts.extend(sub)
    parts=np.array(parts)
    assert not labels and np.array_equal(parts!='',meta['cls']=='format')
    frozen_b=read(rep/'calibration_strength_report.json')['classes']['format']['b']
    results={}
    for name in ('verb','pure_format'):
        use=parts==name
        if not use.any():
            results[name]=dict(positions=0,status='NO_COVERAGE');continue
        info=stats_label('content',data,meta,preds,use)
        slopes=info['calibration']
        s=data['s'][use];mask=data['mask'][use];p=preds['content'][use]
        results[name]=dict(tasks=info['tasks'],positions=info['positions'],mse=info['mse'],
            strongest=info['strongest'],baselines=info['baselines'],diagnostic_slope=slopes,
            raw_KL=summary(strength(s,data['d'][use],mask)[0]),
            P_KL_frozen_format_b=summary(strength(s,frozen_b*p,mask)[0]),
            P_KL_descriptive_b=summary(strength(s,slopes['b']*p,mask)[0]))
    save(out/'format_split_report.json',dict(descriptive_only=True,frozen_format_b=frozen_b,groups=results,
        frozen_format_positions=int((parts!='').sum()),definition='Overlap with command verb bytes within the frozen format class; all remaining format positions in pure_format.',gpu_hours=0))
if __name__=='__main__':describe()
