import difflib,json
from pathlib import Path
folder=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
src={r['idx']:r for r in load(Path('/mnt/data2/ForkAudit/contrast-rewrite/pilot.json'))}
rows=[r for r in load(folder/'pilot.json') if r['qualified']]
samples=[]
for index in range(5):
    r=rows[round(index*(len(rows)-1)/4)]
    q=src[r['idx']]
    samples.append(dict(idx=r['idx'],problem=q['problem'],error=r['error'],fraction=r['edit_tokens']/r['reference_tokens'],original=q['solution'],rewritten=r['text'],B_box=r['B_box'],diff='\n'.join(difflib.unified_diff(q['solution'].splitlines(),r['text'].splitlines(),lineterm=''))))
(folder/'examples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=1)+'\n')
