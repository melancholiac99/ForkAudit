"""Cumulative FP32 measurements for a ten-update window with changing training batches."""
import torch
from .s21_data import summarize


def cumulative_window(updates,probe_batch,probe_ids,probe_targets):
    """Each training entry has its own detached target; all probes use identical candidate support."""
    if len(updates)!=10 or len(probe_ids)==0:
        raise ValueError('a complete window has ten updates and a nonempty fixed probe')
    first=updates[0]['number']
    rows=[]
    for k,u in enumerate(updates):
        if u['number']!=first+k or u['precision']!='fp32':
            raise ValueError('require successive FP32 measurements')
        old,new=u['probe_before'],u['probe_after']
        if set(old)!=set(probe_ids) or set(new)!=set(probe_ids):
            raise ValueError('fixed probe positions changed')
        if any(z.dtype!=torch.float32 for v in (old,new) for z in v.values()):
            raise ValueError('fixed probe logits must be FP32')
        if k and any(not torch.equal(last[i],old[i]) for i in probe_ids):
            raise ValueError('adjacent probe snapshots do not join')
        p=summarize(probe_batch,old,new,probe_targets,probe_ids)
        # Values originate from each update's own FP32 teacher-forced endpoint scoring.
        t=u['fixed_target']
        rows.append(dict(number=u['number'],delta=t['delta_kl'],alignment=t['alignment'],
                         movement=t['movement'],probe_movement=p['movement']))
        last=new
    direct=summarize(probe_batch,updates[0]['probe_before'],last,probe_targets,probe_ids)
    return dict(rows=rows,sum_fixed_target_delta=sum(r['delta'] for r in rows),
                sum_fixed_target_alignment=sum(r['alignment'] for r in rows),
                sum_fixed_target_movement=sum(r['movement'] for r in rows),
                sum_adjacent_probe_movement=sum(r['probe_movement'] for r in rows),
                start_to_end_probe_movement=direct['movement'])
