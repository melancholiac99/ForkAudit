"""E5-L: cross-environment observed-zero replay credit vs trainability.

New experiment, separate from frozen E5-S/C/K measurement preregistration.
CPU: freeze, gate, pairs, analyze. GPU: preflight, train, eval.
Python exploratory_E5L/e5l.py --help
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E5 = ROOT / 'exploratory_E5'
E4 = ROOT / 'exploratory_E4'
DATA = HERE / 'data'
ARMS = ('Szero', 'Sgap', 'Nzero', 'Ngap', 'Smixed', 'Nmixed')
PRIMARY = ARMS[:4]
SEEDS = (0, 1, 2)
LR, DELTA, CAP, MIN_N = 5e-6, 4.0, 960, 256
SAMPLE_SEED = 20261106


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def norm_title(s):
    return ' '.join(s.strip().strip('"').lower().split())


def titles_in(text):
    """Match E5 search_env's title extraction from <information> JSON blocks."""
    found = set()
    for body in re.findall(r'<information>(.*?)</information>', text, flags=re.S):
        try:
            result = json.loads(body)['result']
        except (ValueError, TypeError, KeyError):
            continue
        for t in re.findall(r'(?:^|\n)Doc \d+: (.*)', result):
            found.add(norm_title(t))
    return found


def kind(action):
    if action.startswith('<search>') and action.endswith('</search>'):
        return 'search'
    if action.startswith('<answer>') and action.endswith('</answer>'):
        return 'answer'
    return 'invalid'


def classify(b1):
    """E4b-compatible shallow 5 arms x 3 continuations; E3 excluded."""
    out = [b1['factual_outcomes']] + b1['alt_outcomes']
    if len(out) != 5 or any(len(x) != 3 or any(v not in (0, 1, False, True) for v in x) for x in out):
        raise ValueError('Expected five arms of three binary continuations')
    means = [sum(map(int, a)) / 3 for a in out]
    if max(means) != min(means):
        return 'gap'
    if max(means) == 0:
        return 'zero'
    return 'equal_other'


def choose_support(actions, observations, seen, support):
    """Gold-support-title teacher: an alternative must reveal more *missing* support titles than factual.

    Gold titles are used only offline in teacher construction, never included in training or evaluation prompts.
    """
    missing = set(map(norm_title, support)) - seen
    scores = [len((titles_in(o) - seen) & missing) if kind(a) == 'search' else 0
              for a, o in zip(actions, observations)]
    alternatives = [i for i in range(1, len(actions)) if kind(actions[i]) == 'search' and scores[i] > scores[0]]
    return min(alternatives, key=lambda i: (-scores[i], i)) if alternatives else None


def choose_novelty(actions, observations, seen, support=None):
    """Non-oracle alternative: a search discovers more *new documents* than factual search.

    Search-vs-search only, to avoid globally rewarding delayed answers under the 4-action budget.
    """
    if kind(actions[0]) != 'search':
        return None
    scores = [len(titles_in(o) - seen) if kind(a) == 'search' else -1
              for a, o in zip(actions, observations)]
    alternatives = [i for i in range(1, len(actions)) if kind(actions[i]) == 'search' and scores[i] > scores[0]]
    return min(alternatives, key=lambda i: (-scores[i], i)) if alternatives else None


def freeze(args):
    """Sample HotpotQA train for labels, dev for *held-out* eval; separate pilot train set."""
    paths = [HERE / f'tasks_{p}.json' for p in ('train', 'pilot', 'eval')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Task sets exist; do not overwrite frozen splits')
    raw_train, raw_dev = load(args.hotpot_train), load(args.hotpot_dev)

    def candidates(rows):
        keep = []
        for q in rows:
            titles = sorted({t for t, _ in q['supporting_facts']})
            if q['answer'].strip().lower() in ('yes', 'no') or len(titles) != 2:
                continue
            keep.append((q, titles))
        return keep

    train, dev = candidates(raw_train), candidates(raw_dev)
    # Keep wiki title membership filtering identical to E5-S without storing all wiki-18 titles in memory.
    needed = {norm_title(t) for _, ts in train + dev for t in ts}
    found = set()
    wiki_hash = hashlib.sha256()
    with open(args.wiki18, 'rb') as f:
        for raw in f:
            wiki_hash.update(raw)
            title = norm_title(json.loads(raw)['contents'].partition('\n')[0])
            if title in needed:
                found.add(title)
    wiki_sha = wiki_hash.hexdigest()
    train = [q for q, ts in train if all(norm_title(t) in found for t in ts)]
    dev = [q for q, ts in dev if all(norm_title(t) in found for t in ts)]
    excluded = set()
    auto_exclusions = [p for p in (E5 / 'tasks_search.json', E5 / 'tasks_search_pilot.json') if p.exists()]
    for ex in list(args.exclude_eval) + auto_exclusions:
        excluded |= {q['qid'] for q in load(ex)['tasks']}
    dev = [q for q in dev if q['_id'] not in excluded]
    assert not ({q['_id'] for q in train} & {q['_id'] for q in dev}), 'train/dev ID overlap'
    rng_t, rng_e = random.Random(SAMPLE_SEED), random.Random(SAMPLE_SEED + 1)
    rng_t.shuffle(train)
    rng_e.shuffle(dev)
    if len(train) < args.n_train + args.n_pilot or len(dev) < args.n_eval:
        raise RuntimeError(f'insufficient eligible train/dev: {len(train)}, {len(dev)}')
    sizes = dict(train=args.n_train, pilot=args.n_pilot, eval=args.n_eval)
    selected = dict(train=train[:args.n_train], pilot=train[args.n_train:args.n_train + args.n_pilot],
                    eval=dev[:args.n_eval])
    def record(q, seed):
        return dict(seed=seed, qid=q['_id'], question=q['question'], answers=[q['answer']],
                    type=q['type'], level=q['level'], supporting_titles=sorted({t for t, _ in q['supporting_facts']}),
                    context=q['context'])
    for split, offset in (('train', 10000), ('pilot', 30000), ('eval', 20000)):
        save(HERE / f'tasks_{split}.json', dict(tasks=[record(q, offset + k) for k, q in enumerate(selected[split])],
             source='hotpotqa-'+('dev' if split == 'eval' else 'train'), seed=SAMPLE_SEED,
             wiki18_sha256=wiki_sha, n=len(selected[split]), excluded_eval=len(excluded)))
    save(DATA / 'freeze_manifest.json', dict(counts=sizes, source_sha256={
        'train': digest(args.hotpot_train), 'dev': digest(args.hotpot_dev), 'wiki18': wiki_sha},
        tasks_sha256={p: digest(HERE / f'tasks_{p}.json') for p in sizes}))
    print('Frozen disjoint task splits', sizes, flush=True)


def available(root):
    """Pure extraction of facts, replay classes and teachers from E5 bank1+peek outputs."""
    root = Path(root)
    rows, counts, errors = {arm: [] for arm in ARMS}, collections.Counter(), []
    for trpath in sorted((root / 'rollout').glob('*.json')):
        tr = load(trpath)
        history = tr.get('initial_obs', '')
        for step in tr['steps']:
            turn, tid = step['turn'], tr['traj_id']
            b1path = root / 'replay_b1' / tid / f't{turn:02d}.json'
            peekpath = root / 'peek' / tid / f't{turn:02d}.json'
            history_before = history
            history += '\n' + step.get('obs_after', '')
            if not b1path.exists():
                counts['missing_bank1'] += 1
                continue
            b1 = load(b1path)
            if not b1['cf_defined'] or not step['valid']:
                counts['undefined_or_invalid'] += 1
                continue
            if not peekpath.exists():
                counts['missing_peek'] += 1
                continue
            peek = load(peekpath)
            actions = [step['action']] + b1['alternatives']
            if (b1['factual_action'] != actions[0] or len(actions) != 5 or
                [p['action'] for p in peek['peeks']] != actions or
                not peek['factual_obs_match'] or any(p['refeed_div'] for p in peek['peeks']) or
                any(c['refeed_div'] for c in b1['continuations'])):
                errors.append((tid, turn, 'replay/peek mismatch'))
                continue
            cls = classify(b1)
            counts[cls] += 1
            if cls == 'equal_other':
                continue
            texts = {d['action']: d['text'] for d in reversed(b1['drafts']) if d['valid'] and d['admissible']}
            observations = [p['obs'] for p in peek['peeks']]
            seen = titles_in(history_before)
            for teacher, selector in (('S', choose_support), ('N', choose_novelty)):
                k = selector(actions, observations, seen, tr['meta']['supporting_titles'])
                if k is None:
                    continue
                if actions[k] not in texts:
                    errors.append((tid, turn, 'selected action has no draft response'))
                    continue
                response = texts[actions[k]]
                if not response.strip() or not step['response'].strip():
                    errors.append((tid, turn, 'empty chosen or rejected response'))
                    continue
                row = dict(point_id=f'{tid}_t{turn:02d}', qid=tr['game_file'], task_type=tr['meta']['type'],
                           level=tr['meta']['level'], replay_class=cls, teacher=teacher, prompt=step['prompt'],
                           chosen=response, rejected=step['response'], preferred=k, rejected_arm=0)
                rows[teacher + cls].append(row)
                rows[teacher + 'mixed'].append(row)
                counts[teacher + cls + '_eligible'] += 1
    if errors:
        raise ValueError(f'{len(errors)} invalid replay points (first 5): {errors[:5]}')
    return rows, dict(counts)


def quotas(shares, n):
    keys = sorted(shares)
    raw = {t: n * shares[t] for t in keys}
    q = {t: int(raw[t]) for t in keys}
    for k in sorted(keys, key=lambda x: (-(raw[x] - q[x]), x))[:n - sum(q.values())]:
        q[k] += 1
    return q


def select_rows(pools, counts, cap=CAP, min_n=MIN_N):
    # Anchor task mix to all *replay-gap* search points, not to a teacher's label yield.
    gap_counts = {t: n for t, n in counts['gap_types'].items()}
    if not gap_counts:
        raise RuntimeError('No replay-gap points; cannot build comparable DPO groups')
    tot = sum(gap_counts.values())
    shares = {t: n / tot for t, n in gap_counts.items()}
    avail = {arm: collections.Counter(r['task_type'] for r in pools[arm]) for arm in ARMS}
    n = 0
    for m in range(cap - cap % 16, 0, -16):
        q = quotas(shares, m)
        if all(avail[arm][t] >= qty for arm in ARMS for t, qty in q.items()):
            n = m
            break
    if n < min_n:
        raise RuntimeError(f'E5-L label gate: maximum common N={n}, minimum={min_n}; '
                           f'eligible={dict((a,len(pools[a])) for a in ARMS)}. '
                           'Do NOT repeat or silently weaken labels. Revisit design before looking at training results.')
    quotas_ = quotas(shares, n)
    selected = {}
    for arm in ARMS:
        take = []
        for t, target in quotas_.items():
            pool = sorted((r for r in pools[arm] if r['task_type'] == t), key=lambda r: r['point_id'])
            rng = random.Random(int(hashlib.sha256(f'E5L:{SAMPLE_SEED}:{arm}:{t}'.encode()).hexdigest()[:16], 16))
            take.extend(rng.sample(pool, target))
        selected[arm] = sorted(take, key=lambda r: r['point_id'])
        assert len(selected[arm]) == n and len({r['point_id'] for r in selected[arm]}) == n
    return selected, n, quotas_


def pairs(args):
    if (DATA / 'pairs' / 'manifest.json').exists():
        raise FileExistsError('Pairs already frozen; refuse overwrite')
    pools, counts = available(DATA / 'train_measure')
    if counts.get('missing_bank1', 0) or counts.get('missing_peek', 0):
        raise RuntimeError('Incomplete replay/peek sources; finish E5 bank1 and peek first')
    # Type profile from *all* V3 points, including ones on which neither labeler can emit a pair.
    gap_types = collections.Counter()
    root = DATA / 'train_measure'
    for trpath in sorted((root / 'rollout').glob('*.json')):
        tr = load(trpath)
        for s in tr['steps']:
            p = root / 'replay_b1' / tr['traj_id'] / f't{s["turn"]:02d}.json'
            if p.exists():
                bank = load(p)
                if bank['cf_defined'] and classify(bank) == 'gap':
                    gap_types[tr['meta']['type']] += 1
    selected, n, q = select_rows(pools, dict(gap_types=gap_types), args.cap, args.min_n)
    manifest = dict(n=n, quotas=q, eligible={a:len(pools[a]) for a in ARMS}, counts=counts,
                    source_tasks_sha256=digest(HERE / 'tasks_train.json'),
                    source_replay_root='exploratory_E5L/data/train_measure',
                    labels='S: unseen gold supporting-title retrieval; N: new-document count (search vs search)',
                    groups='zero=15/15 failures; gap=unequal arm mean; equal_other=excluded',
                    group_sha256={})
    for arm, rs in selected.items():
        p = DATA / 'pairs' / f'{arm}.jsonl'
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
        manifest['group_sha256'][arm] = digest(p)
    save(DATA / 'pairs' / 'manifest.json', manifest)
    print('E5-L pairs frozen', n, q, manifest['eligible'], flush=True)


def gate(args):
    pools, counts = available(DATA / ('pilot_measure' if args.pilot else 'train_measure'))
    print(json.dumps(dict(counts=counts, eligible={a: len(pools[a]) for a in ARMS}), indent=2, ensure_ascii=False))


def preflight(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    end = tok.convert_tokens_to_ids('<|im_end|>')
    assert isinstance(end, int) and end >= 0
    manifests = load(DATA / 'pairs' / 'manifest.json')
    maxima = {}
    for arm in ARMS:
        path = DATA / 'pairs' / f'{arm}.jsonl'
        assert digest(path) == manifests['group_sha256'][arm]
        largest = 0
        for r in map(json.loads, open(path, encoding='utf-8')):
            prefix = tok.apply_chat_template([dict(role='user', content=r['prompt'])], tokenize=False,
                                             add_generation_prompt=True)
            p_len = len(tok.encode(prefix, add_special_tokens=False))
            for response in (r['chosen'], r['rejected']):
                largest = max(largest, p_len + len(tok.encode(response, add_special_tokens=False)) + 1)
        maxima[arm] = largest
        if largest > 6144:
            raise RuntimeError(f'{arm} max tokens {largest} > E4 fixed 6144; stop, no silent truncation')
    save(DATA / 'preflight.json', dict(model=args.model, max_tokens=maxima,
                                       pairs_manifest_sha256=digest(DATA / 'pairs' / 'manifest.json')))
    print('Preflight passes', maxima)


def train(args):
    if not args.approved:
        raise RuntimeError('GPU training requires --approved: freeze and review DESIGN_DRAFT.md first')
    manifest = DATA / 'pairs' / 'manifest.json'
    pre = load(DATA / 'preflight.json')
    if pre['model'] != args.model:
        raise RuntimeError('Model differs from the tokenizer preflight')
    if pre['pairs_manifest_sha256'] != digest(manifest):
        raise RuntimeError('Pairs changed since preflight')
    if args.arm not in ARMS or args.seed not in SEEDS:
        raise ValueError('Unknown arm or seed')
    sys.path.insert(0, str(E4))
    import e4_common as E
    import e4_06_train as T
    E.DATA = str(DATA)
    E.ARMS_TRAINED = list(ARMS)
    argv = sys.argv
    try:
        sys.argv = [T.__file__, '--arm', args.arm, '--seed', str(args.seed), '--lr', str(LR), '--model', args.model]
        T.main()
    finally:
        sys.argv = argv


def eval_run(args):
    if not args.approved:
        raise RuntimeError('Formal evaluation requires --approved')
    tag = args.tag
    if tag != 'base' and not any(tag == f'{a}_s{s}' for a in ARMS for s in SEEDS):
        raise ValueError('Invalid tag')
    tasks = load(HERE / 'tasks_eval.json')['tasks']
    if tag != 'base':
        adapter = DATA / 'adapters' / tag
        if not (adapter / 'adapter_model.safetensors').exists():
            raise FileNotFoundError(adapter)
    else:
        adapter = None
    # Import the exact E5-S environment. Fork workers BEFORE opening CUDA/vLLM.
    sys.path.insert(0, str(E5))
    sys.path.insert(0, str(E4))
    sys.path.insert(0, str(ROOT / 'scripts'))
    import adapters
    from e4_policy import LoraPolicy
    env = adapters.get('search')
    env.prepare_workers(str(HERE / 'tasks_eval.json'))
    pool = env.pool(args.workers)
    policy = None
    out = DATA / 'eval' / tag
    out.mkdir(parents=True, exist_ok=True)
    try:
        policy = LoraPolicy(model_path=args.model)
        policy.set_adapter(str(adapter) if adapter else None, 1)
        for begin in range(0, len(tasks), args.chunk):
            chunk = tasks[begin:begin + args.chunk]
            jobs = [(q, rep) for q in chunk for rep in range(args.reps)
                    if not (out / f'{q["seed"]}_r{rep}.json').exists()]
            if not jobs:
                continue
            keys = [f'{q["seed"]}:{rep}' for q, rep in jobs]
            initial = pool.open([(key, env.game_id(q), [], None) for key, (q, _) in zip(keys, jobs)])
            state = {}
            for key, (q, rep) in zip(keys, jobs):
                obs, adm, _, div = initial[key]
                if div:
                    raise RuntimeError('Evaluation reset/replay mismatch')
                state[key] = dict(q=q, rep=rep, task=env.task_text(q, obs), obs=obs, adm=adm, mem=[],
                                  won=False, done=False, steps=[])
            for turn in range(1, env.MAX_STEPS + 1):
                active = [key for key in keys if not state[key]['done']]
                if not active:
                    break
                prompts = [env.build_prompt(state[key]['task'], state[key]['mem'], state[key]['obs'], state[key]['adm'])
                           for key in active]
                seeds = [1_000_000 * state[key]['q']['seed'] + turn + 10 ** 11 * (state[key]['rep'] + 1)
                         for key in active]
                responses = policy.sample(prompts, seeds, logprobs=False)
                actions = {key: env.parse(response[0]['text'])[0] for key, response in zip(active, responses)}
                results = pool.step(actions)
                for key in active:
                    s = state[key]
                    obs, adm, won, done = results[key]
                    s['mem'].append(dict(text_obs=s['obs'], action=actions[key]))
                    s['steps'].append(dict(turn=turn, action=actions[key]))
                    s.update(obs=obs, adm=adm, won=bool(won), done=bool(done or won))
            for key in keys:
                s = state[key]
                save(out / f'{s["q"]["seed"]}_r{s["rep"]}.json', dict(qid=s['q']['qid'], seed=s['q']['seed'],
                     rep=s['rep'], won=s['won'], steps=s['steps']))
            pool.close(keys)
            print(tag, f'{min(begin + args.chunk, len(tasks))}/{len(tasks)} tasks', flush=True)
    finally:
        pool.shutdown()
        if policy is not None:
            policy.close()
    per = {q['qid']: [int(load(out / f'{q["seed"]}_r{r}.json')['won']) for r in range(args.reps)] for q in tasks}
    save(DATA / 'results' / f'{tag}.json', dict(tag=tag, per_game=per, reps=args.reps,
         eval_tasks_sha256=digest(HERE / 'tasks_eval.json'),
         adapter_sha256=digest(adapter / 'adapter_model.safetensors') if adapter else None))
    print(tag, 'success', np.mean(list(per.values())))


def ci(values, seed=20261106, n_boot=10000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sums = np.empty(n_boot)
    for j in range(0, n_boot, 250):
        ids = rng.integers(0, len(x), size=(min(250, n_boot - j), len(x)))
        sums[j:j + len(ids)] = 100 * x[ids].mean(1)
    return [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


def analyze(args):
    qids = [q['qid'] for q in load(HERE / 'tasks_eval.json')['tasks']]
    base = load(DATA / 'results' / 'base.json')
    sha_ = digest(HERE / 'tasks_eval.json')
    assert base['eval_tasks_sha256'] == sha_ and base['reps'] == 3
    values = {'base': np.array([np.mean(base['per_game'][q]) for q in qids])}
    for arm in ARMS:
        reps = []
        for seed in SEEDS:
            r = load(DATA / 'results' / f'{arm}_s{seed}.json')
            assert r['eval_tasks_sha256'] == sha_ and r['reps'] == 3
            assert set(r['per_game']) == set(qids)
            reps.append(np.array([r['per_game'][q] for q in qids], dtype=float))
        values[arm] = np.stack(reps, axis=1).mean((1, 2))
    gain = {a: values[a] - values['base'] for a in ARMS}
    contrast = (values['Szero'] + values['Nzero'] - values['Sgap'] - values['Ngap']) / 2
    results = dict(n_games=len(qids), delta_pp=DELTA,
                   success={a: float(100 * x.mean()) for a, x in values.items()},
                   gain_pp={a: dict(mean=float(100 * x.mean()), ci95=ci(x)) for a, x in gain.items()},
                   pooled_zero_minus_gap=dict(mean=float(100 * contrast.mean()), ci95=ci(contrast)))
    positive = results['gain_pp']['Szero']['ci95'][0] > 0 and results['gain_pp']['Nzero']['ci95'][0] > 0
    bounds = results['pooled_zero_minus_gap']['ci95']
    results['reading'] = ('E5L-NULL' if not positive else 'E5L-SUPPORT' if bounds[0] > -DELTA else
                          'E5L-COUNTER' if bounds[1] < 0 else 'E5L-PARTIAL')
    save(DATA / 'results' / 'E5L_REPORT.json', results)
    print(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    x = sub.add_parser('freeze')
    x.add_argument('--hotpot-train', required=True)
    x.add_argument('--hotpot-dev', required=True)
    x.add_argument('--wiki18', required=True)
    x.add_argument('--exclude-eval', action='append', default=[], help='E5-S task list(s), if already frozen')
    x.add_argument('--n-train', type=int, default=3000)
    x.add_argument('--n-pilot', type=int, default=100)
    x.add_argument('--n-eval', type=int, default=400)
    x = sub.add_parser('gate')
    x.add_argument('--pilot', action='store_true')
    x = sub.add_parser('pairs')
    x.add_argument('--cap', type=int, default=CAP)
    x.add_argument('--min-n', type=int, default=MIN_N)
    x = sub.add_parser('preflight')
    x.add_argument('--model', default=os.environ.get('M0_MODEL', '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'))
    x = sub.add_parser('train')
    x.add_argument('--arm', required=True, choices=ARMS)
    x.add_argument('--seed', type=int, required=True, choices=SEEDS)
    x.add_argument('--model', default=os.environ.get('M0_MODEL', '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'))
    x.add_argument('--approved', action='store_true')
    x = sub.add_parser('eval')
    x.add_argument('--tag', required=True)
    x.add_argument('--model', default=os.environ.get('M0_MODEL', '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'))
    x.add_argument('--reps', type=int, default=3)
    x.add_argument('--workers', type=int, default=8)
    x.add_argument('--chunk', type=int, default=100)
    x.add_argument('--approved', action='store_true')
    sub.add_parser('analyze')
    args = p.parse_args()
    {'freeze': freeze, 'gate': gate, 'pairs': pairs, 'preflight': preflight,
     'train': train, 'eval': eval_run, 'analyze': analyze}[args.stage](args)


if __name__ == '__main__':
    main()
