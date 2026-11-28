import os
import numpy as np
from setup_data import WORK,HERE,ROOT,read,save,sha
from launch import evaluate,validate_checkpoint

def main():
    import json
    output=WORK/'checkpoints'
    rows=[json.loads(line) for line in (output/'end_windows.jsonl').read_text().splitlines()]
    step=rows[-1]['step']
    assert [r['step'] for r in rows]==list(range(25,step+1,25))
    validate_checkpoint(output/f'checkpoint-{step}',step)
    summary=dict(windows=rows,adapter_sha256=sha(output/f'checkpoint-{step}/adapter_model.safetensors'),early_stop=rows[-1]['stop'])
    save(HERE/'training.json',summary)
    if summary['early_stop']:return
    assert step==100
    scores=[];keys=None
    for arm in ('R-contrast-v2@100','R-noclip@100'):
        rec=evaluate(ROOT,os.environ.copy(),arm,'aime24',1001)
        data=read(ROOT/'run/evaluation'/arm/'aime24/1001.json')['results']
        qs=[v['problem'] for v in data]
        assert len(set(qs))==30 and (keys is None or qs==keys)
        keys=qs
        scores.append(np.array([[g['correct'] for g in v['generations']] for v in data],dtype=float))
        assert scores[-1].shape==(30,12)
        if arm.startswith('R-contrast'):
            assert rec['adapter_sha256']==summary['adapter_sha256']
            missing=sum(not g['formatted'] for v in data for g in v['generations'])
    diff=100*(scores[0]-scores[1]).mean(1)
    rng=np.random.default_rng(20261124)
    ci=np.quantile(diff[rng.integers(30,size=(10000,30))].mean(1),[.025,.975])
    save(HERE/'aime24.json',dict(accuracy_pct=100*scores[0].mean(),missing=missing,delta_pp=diff.mean(),ci95=ci.tolist(),repaired=bool(ci[0]>-5 and missing<=6)))
if __name__=='__main__':main()
