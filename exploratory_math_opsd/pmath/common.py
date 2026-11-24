import json, os
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT = Path('/mnt/data/ForkAudit/math-opsd')
WORK = ROOT/'run/P-projector'
MODEL = ROOT/'models/Qwen3-1.7B'

def save(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    tmp.replace(path)

def json_save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def projected_logits(base, ids, mask, start, state):
    full_hidden = base.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    hidden = full_hidden[:,start:-1]
    head = base.get_output_embeddings()
    # Match the trainer's full-sequence lm_head GEMM before indexing response positions.
    logits = head(full_hidden)[:,start:-1]
    if state['scale'] == 0:
        return logits
    # Compute the vocabulary-side rank-8 embedding once per process.
    if 'vocab' not in state:
        state['vocab'] = F.linear(head.weight.float(),state['B'])
    for i in range(hidden.shape[0]):
        for j in range(0,hidden.shape[1],64):
            part = logits[i,j:j+64]
            idx = part.topk(64).indices
            a = F.linear(hidden[i,j:j+64].float(),state['A'])
            delta = torch.einsum('pr,pcr->pc',a,state['vocab'][idx])
            part.scatter_(1,idx,(part.gather(1,idx).float()+state['scale']*delta).to(part.dtype))
    return logits
