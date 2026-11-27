"""CPU-only paired AIME comparisons and first whitespace-token audit."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/data/ForkAudit/math-opsd')
RUN = ROOT/'run'
MODEL = 'R-lambda15@100'
ROLE = {"assistant", "assistant's", "human", "user", "system"}


def write(name, data):
    path = HERE/name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def aime():
    hits = {}
    hashes = {}
    order = None
    for model in (MODEL, 'R@100', 'B', 'R-lambda05@100'):
        folder = RUN/'evaluation'/model/'aime24'
        raw = (folder/'1001.json').read_bytes()
        hashes[model] = hashlib.sha256(raw).hexdigest()
        assert hashes[model] == json.loads((folder/'1001.receipt.json').read_bytes())['sha256']
        rows = json.loads(raw)['results']
        by_id = {r['problem']:r['generations'] for r in rows}
        if order is None:
            order = list(by_id)
        assert len(order)==30 and set(order)==set(by_id) and all(len(g)==12 for g in by_id.values())
        hits[model] = np.array([sum(x['correct'] for x in by_id[q]) for q in order])
    draws = np.random.default_rng(20261124).multinomial(30, np.full(30,1/30),10000)
    paired = {}
    for ref in list(hits)[1:]:
        d = hits[MODEL]-hits[ref]
        paired[ref] = dict(pp=d.sum()*100/360, ci95=np.quantile(draws@d*100/360,[.025,.975]).tolist())
    write('dose15_paired.json', dict(conclusion='低于预测带，最优剂量接近 λ=1', seed=20261124,
          bootstrap_draws=10000, sha256=hashes, correct={k:int(x.sum()) for k,x in hits.items()}, paired=paired))


def classify(text):
    m = re.search(r'\S+',text)
    word = m[0] if m else ''
    return (not word.isascii(), word.lower() in ROLE)


def tally(texts):
    flags = [classify(t) for t in texts]
    n = len(flags)
    hits = sum(a or b for a,b in flags)
    return dict(n=n, hits=hits, pct=100*hits/n, non_ascii=sum(a for a,b in flags), role=sum(b for a,b in flags))


def openings():
    complete, partial, forced = {}, {}, {}
    for folder in sorted((RUN/'math500-long-seed1001-n2').glob('nonthinking-*')):
        key = folder.name.removeprefix('nonthinking-')
        receipt = folder/'receipt.json'
        if receipt.exists():
            raw = (folder/'answers.json').read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == json.loads(receipt.read_bytes())['answers_sha256']
            obj = json.loads(raw)
            assert obj['max_new_tokens']==16384 and not obj['enable_thinking']
            complete[key] = dict(**tally(g['full_generation'] for r in obj['results'] for g in r['generations']),sha256=digest)
            assert complete[key]['n']==1000
        elif key.endswith('@150'):
            shards = sorted((folder/'shards').glob('lane*.json'))
            rows = [r for p in shards if re.fullmatch(r'lane[0-3].json',p.name) for r in json.loads(p.read_bytes())['results']]
            if rows:
                partial[key] = tally(g['full_generation'] for r in rows for g in r['generations'])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT/'models/Qwen3-1.7B/tokenizer.json'))
    base = RUN/'cond-kl/forcing'
    for name in ('B','R-cond@100','budget32/B','budget32/T0@100'):
        def texts():
            for i in range(500):
                rows = json.loads((base/name/f'{i}.json').read_bytes())
                assert len(rows)==2
                for g in rows:
                    yield tok.decode(g['token_ids'],skip_special_tokens=False)
        forced[name] = tally(texts())
    write('dose15_all_openings.json',dict(complete=complete,archived_partial_150=partial,forced=forced))


if __name__=='__main__':
    aime()
    openings()
