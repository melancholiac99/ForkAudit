"""Post-hoc CPU audit of completed, free-generation archives."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = Path('/mnt/data/ForkAudit/math-opsd/run')
SEED = 20261124


def answer(g):
    value = g['predicted_answer']
    if not g['formatted'] or value == '[No boxed answer found]' or value is None or not str(value).strip():
        return None
    text = str(value).strip()
    return str(int(text)) if re.fullmatch(r'[0-9]{1,6}', text) else text


def stats(rows):
    gens = [r['generations'] for r in rows]
    n = len(gens[0])
    assert all(len(g)==n for g in gens)
    data = []
    for group in gens:
        values = [answer(g) for g in group]
        right = sum(g['correct'] for g in group)
        wrong = Counter(a for a,g in zip(values,group) if a is not None and not g['correct'])
        missing = sum(a is None for a in values)
        assert right+missing+sum(wrong.values())==n
        data.append([right/n, sum(wrong.values())/n, missing/n,
                     sum(g['finish_reason']=='length' for g in group)/n,
                     sum(a is None and g['finish_reason']=='length' for a,g in zip(values,group))/n,
                     sum(v*(v-1) for v in wrong.values())/(n*(n-1)),
                     int(right==0), int(right==n), np.mean([g['generated_tokens'] for g in group])])
    return np.asarray(data)


def main():
    contests = ('aime24','aime25','hmmt25')
    out={}
    for bench in (*contests,'math500_16k'):
        sources=(RUN/'evaluation').glob('*/'+bench+'/1001.json') if bench in contests else (RUN/'math500-long-seed1001-n2').glob('nonthinking-*/answers.json')
        matrices={}; hashes={}; base_ids=None
        for p in sorted(sources):
            model=p.parents[1].name if bench in contests else p.parent.name.removeprefix('nonthinking-')
            raw=p.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
            receipt=p.with_name('1001.receipt.json' if bench in contests else 'receipt.json')
            if not receipt.exists():continue
            assert digest==json.loads(receipt.read_bytes())['sha256' if bench in contests else 'answers_sha256']
            obj=json.loads(raw); rows=obj['results']; ids=[x['problem'] for x in rows]
            assert len(set(ids))==len(ids) and (base_ids is None or base_ids==ids)
            assert obj['seed']==1001 and obj['val_n']==(12 if bench in contests else 2)
            base_ids=ids; hashes[model]=digest; matrices[model]=stats(rows)
        pairs={}; rng=np.random.default_rng(SEED)
        n=len(base_ids); draws=rng.multinomial(n,np.full(n,1/n),10000)
        fields=['right','wrong_answer','missing','length','missing_and_length','same_wrong_pair','none_correct','all_correct','tokens']
        def report(x):
            v=x.mean(axis=0); ci=np.quantile(draws@x/n,[.025,.975],axis=0)
            return {name:[float(v[j]),*ci[:,j].tolist()] for j,name in enumerate(fields)}
        for a,b in [('R@100','B'),('R@100','T0@100'),('P@100','B'),('R-extension@200','R@100'),('R-switch@200','R-extension@200'),('R-lambda15@100','R@100'),('R-lambda05@200','R-lambda05@100'),('R-clip-boundary@100','R@100')]:
            if a in matrices and b in matrices:pairs[a+' minus '+b]=report(matrices[a]-matrices[b])
        out[bench]=dict(models={a:report(x) for a,x in matrices.items()},pairs=pairs,sha256=hashes,questions=n)
        if bench=='aime24':
            models=['B','R@100','R-extension@200','R-switch@200']
            out[bench]['questions_detail']=[dict(problem=q,correct={a:round(matrices[a][i,0]*12) for a in models}) for i,q in enumerate(base_ids)]
    (HERE/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
