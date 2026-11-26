import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer
from positions import boundaries,pieces
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/data/ForkAudit/math-opsd')
tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
v=pieces(tok);rows={q:(sp,y) for k in range(4) for q,sp,_,y in torch.load(ROOT/'run/P-projector'/f'rollouts-{k}.pt',weights_only=True)}
chosen=np.random.default_rng(20261124).choice(sorted(rows),20,replace=False)
lines=['# 20 条固定探针回答的边界标注','','⟦t⟧ 标在待预测 token 之前，t 从 0 起计。换行及前导空格保留。每条均与逐位置直接解码核对。'];brief=[]
for order,q in enumerate(chosen):
    sp,y=rows[q];dec=lambda ids:tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    vrow=[v[t] for t in y];marked=boundaries(vrow,b''.join(v[t] for t in sp));prompt=dec(sp)
    assert (b''.join(v[t] for t in sp)+b''.join(vrow)).decode(errors='replace')==dec(sp+y)
    for i,m in enumerate(marked):
        pref=dec(sp+y[:i]);trim=pref.rstrip()
        target=pref.endswith('\n') or trim.endswith(('.', '!', '?'))
        if target and pref.endswith('.') and len(pref)>=2 and pref[-2].isdigit():
            target=not dec(y[i:i+6])[:1].isdigit()
        assert m==target,(q,i)
    ids=np.flatnonzero(marked)
    tagged=b''.join((f'⟦{i}⟧'.encode() if marked[i] else b'')+x for i,x in enumerate(vrow)).decode(errors='replace')
    lines+=['',f'## 题 {q}（{len(ids)}/{len(y)}）','','~~~~text',tagged,'~~~~']
    i=next((int(j) for j in ids if j>0 and dec((sp+y[:j])[-8:]).endswith('\n')==(order%2==0)),int(ids[0]))
    brief.append(dict(question=int(q),position=i,prefix_tail=dec((sp+y[:i])[-12:]),next_token=dec(y[i:i+1]),boundaries=len(ids),tokens=len(y)))
(HERE/'EXAMPLES.md').write_text('\n'.join(lines)+'\n')
(HERE/'examples.json').write_text(json.dumps(brief,ensure_ascii=False,indent=2)+'\n')

short=['# 20 条回答各示一个边界（完整标注见 EXAMPLES.md）','','⟦t⟧ 左侧是前缀末尾，右侧是待预测 token；空白使用 JSON 转义。','','|题号|边界／总位置|样例|','|---:|---:|---|']
for v in brief:
    left=json.dumps(v['prefix_tail'][-28:],ensure_ascii=False)[1:-1].replace('|','&#124;')
    right=json.dumps(v['next_token'],ensure_ascii=False)[1:-1].replace('|','&#124;')
    short.append(f"|{v['question']}|{v['boundaries']}/{v['tokens']}|<code>{left}⟦{v['position']}⟧{right}</code>|")
(HERE/'SAMPLE_TABLE.md').write_text('\n'.join(short)+'\n')
