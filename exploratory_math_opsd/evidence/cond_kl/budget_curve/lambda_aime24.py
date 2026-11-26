"""Run only AIME24 and archive paired lambda comparisons as each arm finishes."""
import os
import numpy as np
from contest import HERE, ROOT, measure, org, read, save, sha


def main():
    env=os.environ.copy()
    org.init()
    names=['B','R@100','R-noclip@100','R-lambda0@100','R-lambda05@100']
    rng=np.random.default_rng(20261124)
    w=np.vstack([np.ones(30),rng.multinomial(30,np.full(30,1/30),10000)])/30
    models={}; receipts={}; order=None; out={}
    for name in names:
        if name.startswith('R-lambda'):
            d=ROOT/'run'/name.split('@')[0]
            assert read(d/'complete.json')['adapters']['100']==sha(d/'checkpoint-100/adapter_model.safetensors')
        ids,rows,receipts[name]=measure(name,'aime24',env)
        assert order is None or ids==order
        order=ids
        models[name]=w@rows[:,0]
        if not name.startswith('R-lambda'):continue
        for key,arr in [(name,models[name])]+[(name+' minus '+b,models[name]-models[b]) for b in names[:3]]:
            out[key]=dict(pp=arr[0],ci95=np.quantile(arr[1:],[.025,.975]).tolist())
        d=out['R-lambda0@100 minus R@100']
        passed=bool(d['ci95'][0]<=0<=d['ci95'][1] and out['R-lambda0@100']['pp']>=53)
        save(HERE/'lambda_aime24_results.json',dict(estimates=out,receipts=receipts,
             lambda0_meets_registered_first_prediction=passed,expansion='await_researcher_decision'))
        lines=['# λ 族 AIME24（思考模式）','','|模型或配对差|准确率或差值 [95% 区间]|','|---|---:|']
        for key,x in out.items():lines.append('|%s|%.2f [%.2f, %.2f]|'%(key,x['pp'],*x['ci95']))
        lines+=['',f'λ=0 第一项预写条件：{passed}。AIME25、HMMT25 候选待研究者决定。']
        (HERE/'lambda_aime24.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
