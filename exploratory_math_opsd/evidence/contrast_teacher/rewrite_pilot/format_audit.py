"""Carry over existing semantic codes; review newly entering positions by their archived context."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent
read=lambda name:json.loads((p/name).read_text())
groups=read('format_probe.json')
labels=read('top_audit.json')['labels']
uid=lambda r:(r['idx'],r['t'])
codes={uid(r):l=='direct' for r,l in zip(groups['raw']['top'],labels,strict=True)}
# Newly seen ranks (zero based); code only the sampled token, including changed downstream answers.
direct={'masked':{79,88,92},'old_target':{5,81,101},'target':{67,80}}
out={}
for name,group in groups.items():
    ranks=[]
    for i,r in enumerate(group['top']):
        key=uid(r)
        if key not in codes:codes[key]=i in direct[name]
        if codes[key]:ranks.append(i)
    out[name]=dict(direct=len(ranks),top=109,direct_ranks_zero_based=ranks,eos=group['eos'],final=group['final'])
assert out['raw']['direct']==16
(p/'format_report.json').write_text(json.dumps(out,indent=2)+'\n')
