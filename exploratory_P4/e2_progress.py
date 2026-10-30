"""Report execution progress only; never inspect scientific gains."""
from pathlib import Path
import json
out=Path(__file__).resolve().parent
meta=json.loads((out/'e2_32b_model_source.json').read_text())
shards=[s for s in meta['siblings'] if s['rfilename'].endswith('.safetensors')]
received=verified=0
for i,s in enumerate(shards):
 base=Path('/autodl-fs/data/forkaudit_models/Qwen2.5-32B-Instruct' if i<8 else '/root/autodl-tmp/models/Qwen2.5-32B-Instruct')
 p=base/s['rfilename']
 if p.exists():received+=p.stat().st_size;verified+=1
 elif p.with_suffix(p.suffix+'.part').exists():received+=p.with_suffix(p.suffix+'.part').stat().st_size
state=out/'e2_32b_chunks_state.json'
if state.exists():
    data=json.loads(state.read_text());received=sum(row['prefix']+sum(min(data['chunk'],row['size']-start) for start in row['done']) for row in data['files'].values())
models={}
for name in ['7b','32b']:
 p=out/f'e2_{name}/run.json'
 if not p.exists():models[name]='not started';continue
 run=json.loads(p.read_text())
 if run['complete']:models[name]=dict(complete=True,contexts=run['contexts']);continue
 log=out/f'e2_{name}.log';last={}
 if log.exists():
  for line in log.read_text().splitlines():
   try:v=json.loads(line)
   except (ValueError,TypeError):continue
   if isinstance(v,dict):last=v
 models[name]=last
print(json.dumps(dict(download_GiB=round(received/2**30,2),download_percent=round(100*received/sum(s['size'] for s in shards),1),verified_shards=verified,models=models),ensure_ascii=False))
