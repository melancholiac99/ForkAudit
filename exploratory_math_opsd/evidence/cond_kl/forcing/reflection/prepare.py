import re
from functools import lru_cache
import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from settings import *
sys.path.insert(0,str(ROOT/'source/eval'))
from evaluate_math import extract_boxed_answer,grade_answer
sys.path.insert(0,str(HERE.parents[3]/'pmath'))
from data import stream
from common import WORK,MODEL

sentence=re.compile(r'([.!?。！？])["”’\')\]]*\s*$')
blank=re.compile(r'\n[ \t]*\n\s*$')
def boundaries(ids,decode):
    last='';out=[]
    for pos,token in enumerate(ids[:-1],1):
        piece=decode(token)
        text=last+piece
        match=sentence.search(text)
        if blank.search(text) and not blank.search(last):
            out.append((pos,'paragraph'))
        elif piece.strip() and match and match.start()>=len(last) and not (match[1]=='.' and match.start()>0 and text[match.start()-1].isdigit()):
            out.append((pos,'sentence'))
        last=text[-64:]
    return out

def run():
    tok=AutoTokenizer.from_pretrained(MODEL)
    decode=lru_cache(None)(lambda t:tok.decode([t],clean_up_tokenization_spaces=False))
    refs=[r for f in sorted((ROOT/'datasets/train/data').glob('*.parquet')) for r in pq.read_table(f,columns=['problem','Answer']).to_pylist()]
    rows=dict(stream());rng=np.random.default_rng(SEED)
    m={};pool=[]
    eos=read(MODEL/'generation_config.json')['eos_token_id']
    for lane in range(4):
        for idx,sp,tp,y in torch.load(WORK/f'rollouts-{lane}.pt',weights_only=True):
            assert rows[idx]['problem']==refs[idx]['problem']
            gold=refs[idx]['Answer'] or extract_boxed_answer(rows[idx]['solution'])
            # Some reference Answer fields themselves contain a boxed expression.
            if gold:gold=extract_boxed_answer(gold) or gold.strip()
            pred=extract_boxed_answer(tok.decode(y,skip_special_tokens=True))
            ends=boundaries(y,decode)
            m[idx]=dict(gold=gold,correct=bool(grade_answer(pred,gold)) if gold else None,has_box=bool(pred),
                        ended=y[-1] in eos,length=len(y),boundaries=ends,sites=[])
            eligible=[p for p,g in ends if g=='sentence' and gold]
            if eligible:
                pool.extend((idx,int(pos)) for pos in rng.choice(eligible,min(2,len(eligible)),replace=False))
    picked=sorted(pool[i] for i in rng.choice(len(pool),min(2048,len(pool)),replace=False))
    for idx,pos in picked:m[idx]['sites'].append(pos)
    assert len(m)==1500 and sum(x['length'] for x in m.values())==1185192
    save(RAW/'labels.json',m)
    save(HERE/'coverage.json',dict(answers=len(m),correct=sum(x['correct'] is True for x in m.values()),wrong=sum(x['correct'] is False for x in m.values()),
            missing_reference=sum(x['correct'] is None for x in m.values()),normal_eos=sum(x['ended'] for x in m.values()),with_box=sum(x['has_box'] for x in m.values()),
            boundaries=sum(len(x['boundaries']) for x in m.values()),fork_sites=len(picked),fork_questions=len(set(i for i,p in picked))))
if __name__=='__main__':run()
