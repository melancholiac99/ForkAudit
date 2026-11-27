"""Select reproducible cases and independently check the digit-replacement problem."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from audit import HERE, RUN, SEED


def load(benchmark, model):
    folder = RUN / 'evaluation' / model / benchmark
    path = folder / '1001.json'
    if benchmark == 'math500_16k':
        path = RUN / 'math500-long-seed1001-n2' / f'nonthinking-{model}' / 'answers.json'
    raw = path.read_bytes()
    expected = json.loads((HERE / 'results.json').read_bytes())[benchmark]['sha256'][model]
    assert hashlib.sha256(raw).hexdigest() == expected
    return json.loads(raw)['results']


def ref(row, sample):
    g = row['generations'][sample]
    return {'archive_answer': g['predicted_answer'], 'correct': g['correct'],
            'tokens': g['generated_tokens'],
            'generation_sha256': hashlib.sha256(g['full_generation'].encode()).hexdigest()}


def remainders(value):
    s = str(value)
    return [int(s[:i] + '1' + s[i + 1:]) % 7 for i in range(4)]


def main():
    b, r = [load('math500_16k', model) for model in ('B', 'R@100')]
    rng = random.Random(SEED)
    short = []
    for labels in [(False, True), (True, False), (False, False), (True, True)]:
        eligible = []
        for i, (x, y) in enumerate(zip(b, r)):
            for j, (g, h) in enumerate(zip(x['generations'], y['generations'])):
                if (g['correct'], h['correct']) == labels and max(g['generated_tokens'], h['generated_tokens']) <= 1250:
                    eligible.append((i, j))
        for i, j in rng.sample(eligible, 2):
            short.append({'row': i, 'sample': j, 'eligible_in_stratum': len(eligible),
                          'B': ref(b[i], j), 'R@100': ref(r[i], j)})
    r100, r200 = [load('aime24', m) for m in ('R@100', 'R-extension@200')]
    diff = np.array([y['num_correct'] - x['num_correct'] for x, y in zip(r100, r200)])
    keep = np.delete(diff, 10) / 12
    draws = np.random.default_rng(SEED).multinomial(len(keep), np.full(len(keep), 1 / len(keep)), 10000)
    probe = {'valid_n': [n for n in range(1000, 10000) if not any(remainders(n))],
             'reported_9656_remainders': remainders(9656),
             'reported_5694_remainders': remainders(5694)}
    data = {'seed': SEED, 'index_base': 0, 'math_short': short, 'aime24_digit_checker': probe,
            'aime24_question_delta_counts': {k: int(v) for k, v in zip(('negative','equal','positive'), ((diff<0).sum(),(diff==0).sum(),(diff>0).sum()))},
            'aime24_excluding_ambiguous_row10_difference': [float(keep.mean()), *np.quantile(draws @ keep / len(keep), [.025,.975]).tolist()]}
    (HERE / 'cases.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
