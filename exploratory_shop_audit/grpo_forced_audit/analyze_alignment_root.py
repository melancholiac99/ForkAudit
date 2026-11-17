"""Rebuild the root-cause comparison table from saved diagnostic forwards."""
import json
from .common import DEFAULT_RUN, save


def main():
    folder=DEFAULT_RUN.resolve()/'alignment_diagnostic/checkpoint_0002'
    sources={name:json.loads((folder/name).read_text()) for name in
             ['root_gpu.json','root_cpu_fp32.json','root_gpu_residual_fp32.json','fresh_vllm.json']}
    rows={}
    for name,data in sources.items():
        for r in data['records']:
            row=rows.setdefault(str(r['row']),{'token':r.get('token')})
            if name=='fresh_vllm.json':
                import math
                row['archived_sampler']=math.exp(r['sampler_logp'])
                row['fresh_vllm']=None if r['fresh_vllm_logp'] is None else math.exp(r['fresh_vllm_logp'])
            else:
                key=name.replace('.json','')+'/'+r['mode']
                row[key]=r.get('probability',{'error':r.get('error')})
    save(folder/'root_summary.json',{'rows':rows,'note':'Three selected outlier positions, not a representative population sample. CPU FP32 is a higher-precision reference, not a proof of exact arithmetic.'})
    print(json.dumps(rows,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
