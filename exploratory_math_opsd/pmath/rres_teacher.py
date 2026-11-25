"""Subtract the all-training-question projection once on the top-32 union."""
import torch
from torch.nn import functional as F

def subtract_union(t,s,h,state):
    ids=torch.cat((t.topk(32).indices,s.topk(32).indices),dim=-1).sort(dim=-1).values
    valid=torch.ones_like(ids,dtype=torch.bool)
    valid[...,1:]=ids[...,1:]!=ids[...,:-1]
    correction=state['scale']*torch.einsum('pr,pcr->pc',F.linear(h.float(),state['A']),state['vocab'][ids])
    # Duplicate entries have identical corrected values; assign the value once via unique indices.
    row,col=valid.nonzero(as_tuple=True)
    token=ids[row,col]
    t[row,token]=(t[row,token].float()-correction[row,col]).to(t.dtype)

def residual_logits(base,inputs,sp,tp,state):
    student=base.model(input_ids=inputs['student_input_ids'],attention_mask=inputs['student_attention_mask'],
                       use_cache=False).last_hidden_state
    teacher=base(input_ids=inputs['teacher_input_ids'],attention_mask=inputs['teacher_attention_mask'],
                 use_cache=False).logits[:,tp-1:-1]
    head=base.get_output_embeddings()
    if 'vocab' not in state:state['vocab']=F.linear(head.weight.float(),state['B'])
    s=head(student)[:,sp-1:-1]
    for i in range(s.shape[0]):
        for j in range(0,s.shape[1],64):
            n=min(64,s.shape[1]-j)
            subtract_union(teacher[i,j:j+n],s[i,j:j+n],student[i,sp-1+j:sp-1+j+n],state)
    return teacher
