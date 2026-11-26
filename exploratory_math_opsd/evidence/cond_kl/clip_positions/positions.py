"""Classify the decision before each archived/sampled token from its decoded prefix."""
import codecs
import numpy as np
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

def pieces(tokenizer):
    inverse={v:k for k,v in bytes_to_unicode().items()}
    added=tokenizer.added_tokens_decoder
    return [(added[i].content.encode() if i in added else bytes(inverse[c] for c in s))
            for i,s in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))))]

def boundaries(parts,prompt=b''):
    decoder=codecs.getincrementaldecoder('utf-8')('replace')
    data=b''.join(parts);offset=0;last='';trim='';decimal_left=False
    def consume(raw):
        nonlocal last,trim,decimal_left
        text=decoder.decode(raw)
        meaningful=text.rstrip()
        if meaningful:
            prev=meaningful[-2] if len(meaningful)>1 else last
            trim=meaningful[-1];decimal_left=trim=='.' and prev.isdigit()
        if text:last=text[-1]
    consume(prompt)
    result=[]
    for part in parts:
        incomplete=bool(decoder.getstate()[0])
        flag=not incomplete and (last=='\n' or trim in ('.','!','?'))
        if flag and decimal_left and last=='.':
            nxt=data[offset:offset+4].decode('utf-8',errors='replace')[:1]
            flag=not nxt.isdigit()
        result.append(flag)
        consume(part);offset+=len(part)
    return np.array(result,dtype=bool)

def cap_tensor(trainer,inputs):
    import os
    import torch
    if not hasattr(trainer,'_boundary_vocab'):
        trainer._boundary_vocab=pieces(trainer.processing_class)
    v=trainer._boundary_vocab;k=inputs['student_prompt_length']
    ids=inputs['student_input_ids'];att=inputs['student_attention_mask']
    masks=[]
    for row,keep in zip(ids.detach().tolist(),att.detach().tolist()):
        prompt=b''.join(v[t] for t,a in zip(row[:k],keep[:k]) if a)
        masks.append(boundaries([v[t] for t in row[k:]],prompt))
    selected=torch.as_tensor(np.stack(masks),device=ids.device)
    if not hasattr(trainer,'_boundary_logged'):
        valid=inputs['labels'][:,k:]!=-100
        print('prefix_clip_first_microbatch',os.environ['OPSD_ARM'],
              'valid',valid.sum(1).tolist(),'boundary',(selected&valid).sum(1).tolist(),flush=True)
        trainer._boundary_logged=True
    if os.environ['OPSD_ARM']=='R-clip-interior':selected=~selected
    return torch.where(selected,0.05,float('inf')).unsqueeze(-1)
