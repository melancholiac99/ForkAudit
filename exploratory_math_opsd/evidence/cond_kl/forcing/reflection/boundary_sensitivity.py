"""Post-audit sensitivity: discard every ASCII-exclamation sentence end, including prose exclamations."""
from settings import *
from report import targeting, forks
from transformers import AutoTokenizer
import torch
import re

def run():
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B')
    labels=read(RAW/'labels.json')
    excl={}
    n=0;selected=0
    for lane in range(4):
        for idx,_,_,y in torch.load(ROOT/'run/P-projector'/f'rollouts-{lane}.pt',weights_only=True):
            row=labels[str(idx)]
            ex={p for p,g in row['boundaries'] if g=='sentence' and re.search(r'''!["”’')\]]*\s*$''',tok.decode(y[max(0,p-16):p]))}
            n+=len(ex);selected+=len(ex.intersection(row['sites']))
            if ex:excl[idx]=ex
    targeting(excl);forks(excl)
    save(HERE/'boundary_audit.json',dict(excluded_sentence_positions=n,excluded_fork_sites=selected,method='Remove all ASCII ! sentence endpoints, without replacement. Includes genuine prose exclamations.'))
if __name__=='__main__':run()
