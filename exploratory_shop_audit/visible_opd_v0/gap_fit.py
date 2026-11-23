"""Fit four ASIN-held-out projectors to the full teacher distribution and calibrate OOF."""
import sys
import time

import numpy as np
import torch

from .b_fit import weights
from .e25_offline import shrink
from .e25_projector import FitData
from .gap_score import RUN, STAGE_B, target
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


def load_rows(folder):
    parts = []
    for rank in range(4):
        p = folder/f'features_{rank}.pt'
        assert digest_file(p) == read(folder/f'score_{rank}.json')['sha']
        parts.append(torch.load(p, weights_only=True, map_location='cpu', mmap=True))
    tensors = {k:torch.cat([p[k] for p in parts]) for k in ('ids','mask','s','h','teacher','q','t')}
    assert torch.allclose(tensors['teacher'],target(tensors['t'],tensors['mask']),atol=1e-6,rtol=0)
    meta = {k:np.concatenate([p[k] for p in parts]) for k in ('cls','goal','fold')}
    return tensors, meta


def fit(folder, rank, previous):
    start = time.monotonic()
    rows, meta = load_rows(folder)
    tr = np.flatnonzero(meta['fold'] != rank)
    va = np.flatnonzero(meta['fold'] == rank)
    assert len(tr) and len(va) and not set(meta['goal'][tr]) & set(meta['goal'][va])
    data = FitData(rows, meta, torch.load(STAGE_B/'head.pt',mmap=True,weights_only=True), 'cuda')
    warm = torch.load(previous/f'fold_{rank}.pt',weights_only=True) if previous else None
    net = data.fit(tr,rank,'kl',initial_state=warm)
    with torch.no_grad():
        preds = torch.cat([data.predict(net,b).cpu() for b in np.array_split(va,(len(va)+511)//512)])
    torch.save({k:p.cpu() for k,p in net.state_dict().items()}, folder/f'fold_{rank}.pt')
    np.savez(folder/f'oof_{rank}.npz',ix=va,pred=preds.numpy())
    save(folder/f'fit_{rank}.json',dict(seconds=time.monotonic()-start,train_positions=len(tr),
         heldout_positions=len(va),train_goals=len(np.unique(meta['goal'][tr])),epochs=10,
         model_sha=digest_file(folder/f'fold_{rank}.pt'),warm_start_sha=digest_file(previous/f'fold_{rank}.pt') if previous else None))


def calibrate(folder):
    start = time.monotonic()
    rows, meta = load_rows(folder)
    n = len(meta['fold'])
    pred = np.empty(rows['s'].shape, dtype=np.float64)
    for rank in range(4):
        a = np.load(folder/f'oof_{rank}.npz')
        assert np.array_equal(a['ix'], np.flatnonzero(meta['fold']==rank))
        pred[a['ix']] = a['pred']
    mask = rows['mask'].numpy()
    q = rows['teacher'].double().numpy()
    q /= q.sum(1,keepdims=True)
    ix = np.arange(n)
    coeff = shrink(ix, rows['s'].double().numpy(), q, mask, pred, meta, weights(meta,ix))
    save(folder/'calibration.json',dict(classes=coeff,positions=n,tasks=len(set(meta['goal'])),
         folds=[digest_file(folder/f'fold_{r}.pt') for r in range(4)],seconds=time.monotonic()-start))


if __name__=='__main__':
    from .e25_probe import configure
    folder=RUN/sys.argv[2]
    if sys.argv[1]=='fit':
        fit(folder, configure(), RUN/sys.argv[3] if len(sys.argv)>3 else None)
    elif sys.argv[1]=='calibrate':
        calibrate(folder)
