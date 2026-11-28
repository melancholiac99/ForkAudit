"""Use only archived non-thinking training generations shared by the three runs."""
from collections import Counter
from pathlib import Path
import json,re
import numpy as np
from tokenizers import Tokenizer
D=Path(__file__).resolve().parent;ROOT=Path('/mnt/data/ForkAudit/math-opsd')
tokenizer=Tokenizer.from_file(str(ROOT/'models/Qwen3-1.7B/tokenizer.json'))
def collect(arm):
    out={}
    for f in sorted((ROOT/'run'/arm/'generations').glob('*.json')):
        for row in json.loads(f.read_text())['generations']:
            # step is zero-based in the logger.
            step=row['step']+1
            if step>100:continue
            question=row['prompt'].replace('<|endoftext|>','').split('Problem: ',1)[1].rsplit('\n\nPlease reason',1)[0]
            k=(step,question);assert k not in out
            t=row['completion'].replace('<|endoftext|>','');tokens=len(tokenizer.encode(t,add_special_tokens=False).ids)
            out[k]=(tokens,'<|im_end|>' in t,'\\boxed{' in t,t.lower().count('wait'),t.count('###'))
    return out
arms=['R','R-noclip','R-contrast'];by={a:collect(a) for a in arms}
shared=set.intersection(*(set(v) for v in by.values()))
neg=json.loads(Path('/mnt/data2/ForkAudit/contrast/negative.json').read_text())
out={'available':{a:len(x) for a,x in by.items()},'shared':len(shared),'windows':{}}
for lo,hi in ((1,25),(26,50),(51,75),(76,100)):
    ids=sorted(k for k in shared if lo<=k[0]<=hi)
    assert ids
    out['windows'][f'{lo}-{hi}']={a:dict(n=len(ids),**dict(zip(['mean_tokens','end_marker_rate','box_start_rate','wait_count','headings'],np.array([x[k] for k in ids]).mean(0).tolist()))) for a,x in by.items()}
    selected=[k for k in ids if neg[k[1]] is not None]
    out['windows'][f'{lo}-{hi}']['selected_context_only']={a:dict(n=len(selected),ends=sum(x[k][1] for k in selected)) for a,x in by.items()}
(D/'rollouts.json').write_text(json.dumps(out,indent=2)+'\n')
