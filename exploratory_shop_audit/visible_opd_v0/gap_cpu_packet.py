"""Task-cluster bootstrap and descriptive packet KL on the archived Stage B support."""
from hashlib import sha256
import numpy as np
from scipy.special import logsumexp
import torch

from exploratory_shop_audit.grpo_forced_audit.common import ROOT, digest_file, save

OUT = ROOT/'exploratory_shop_audit/visible_opd_v0/evidence/section26_20261123/posthoc_cpu'
SEED = 20261124
DRAWS = 10_000


def cluster_mean(goals, values):
    """Action/position-weighted means. Each resampled task keeps all its rows."""
    v = np.asarray(values, dtype=np.float64)
    tasks, ix, n = np.unique(goals, return_inverse=True, return_counts=True)
    result = dict(n=len(v), tasks=len(tasks), mean=None, ci95=None)
    if not len(v):
        return result
    result['mean'] = v.mean(0).tolist()
    if len(tasks) < 2:
        return result
    totals = np.zeros((len(tasks), v.shape[1]))
    np.add.at(totals, ix, v)
    rng = np.random.default_rng(SEED)
    means = []
    for start in range(0, DRAWS, 250):
        draws = rng.integers(len(n), size=(min(250, DRAWS-start), len(n)))
        means.append(totals[draws].sum(1) / n[draws].sum(1)[:, None])
    result['ci95'] = np.quantile(np.vstack(means), [.025, .975], axis=0).T.tolist()
    return result


def log_probs(x, mask):
    masked = np.where(mask, x, -np.inf)
    return np.where(mask, masked-logsumexp(masked, axis=1, keepdims=True), 0.)


def main():
    directory = ROOT/'.runtime/runs/shop_visible_opd_content_b_20261120/features'
    sources = sha256()
    goals, classes, values = [], [], []
    files = sorted(directory.glob('*.pt'))
    for path in files:
        sources.update((path.name+' '+digest_file(path)+'\n').encode())
        row = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        s, old, content = (row[k].double().numpy() for k in ('s', 'old', 'd'))
        mask = row['mask'].numpy()
        log_s = log_probs(s, mask)
        assert np.max(abs(np.where(mask, np.exp(log_s), 0.)-row['q'].numpy())) < 2e-6
        pair = []
        for delta in (old-content, old):
            log_t = log_probs(s+delta, mask)
            pair.append((np.where(mask, np.exp(log_t), 0.)*(log_t-log_s)).sum(1))
        values.extend(np.column_stack(pair))
        classes.extend(row['classes'])
        goals.extend([row['goal']]*len(mask))
    c, g, v = np.array(classes), np.array(goals), np.array(values)
    assert (len(g), len(np.unique(g))) == (34825, 512) and v.min() > -1e-12
    categories = {}
    for name in ['all', *sorted(set(classes))]:
        take = np.ones(len(v), dtype=bool) if name=='all' else c==name
        categories[name] = cluster_mean(g[take], v[take])
    save(OUT/'packet_kl.json', dict(columns=['placeholder_minus_none', 'full_minus_none'],
         unit='nats per candidate-conditioned position; equal position weights', seed=SEED,
         bootstrap_draws=DRAWS, cluster='task', files=len(files), feature_files_digest=sources.hexdigest(),
         categories=categories))


if __name__ == '__main__':
    main()
