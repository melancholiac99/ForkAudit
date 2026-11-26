"""Per-question KL chain-rule sums on the archived fixed probe."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'stop_decomposition'))
import probe

OUT = probe.ROOT / 'run/cond-kl/preflight'
GROUPS = ('all', 'stop05', 'actual_eos')


def controls():
    model = probe.MODEL
    tokenizer = AutoTokenizer.from_pretrained(model)
    eos = json.loads((model / 'generation_config.json').read_text())['eos_token_id']
    ids = sorted(set(eos) | set(tokenizer.added_tokens_decoder))
    print(json.dumps({i: tokenizer.convert_ids_to_tokens(i) for i in ids}), flush=True)
    return ids, eos


def mass_term(q, p):
    return torch.where(q > 0, q * (q.log() - p.log()), 0.)


@torch.inference_mode()
def run(lane):
    ids, eos = controls()
    model = probe.base_model()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = torch.load(probe.WORK / f'rollouts-{lane}.pt', weights_only=True)
    for idx, sp, tp, y in rows:
        dst = OUT / f'{idx}.npy'
        if dst.exists():
            continue
        s = probe.forward(model, *probe.inputs(sp, y), len(sp))
        t = probe.forward(model, *probe.inputs(tp, y), len(tp))
        archive = np.load(probe.OUT / f'{idx}.npz')
        masks = np.stack([np.ones(len(y), bool), archive['p'] >= .05, archive['eos']])
        acc = torch.zeros((3, 4), dtype=torch.double, device='cuda')
        for start in range(0, len(y), 32):
            sl = slice(start, start + 32)
            lp = (s[sl].double() / 1.1).log_softmax(-1)
            lq = (t[sl].double() / 1.1).log_softmax(-1)
            p, q = lp.exp(), lq.exp()
            kl = q * (lq - lp)
            outside = torch.ones(p.shape[-1], dtype=torch.bool, device=p.device)
            outside[ids] = False
            control = kl[:, ids].sum(-1) + mass_term(q[:, outside].sum(-1), p[:, outside].sum(-1))
            big = kl > .05
            amp = mass_term((q*big).sum(-1), (p*big).sum(-1)) + mass_term((q*~big).sum(-1), (p*~big).sum(-1))
            total = kl.sum(-1)
            assert torch.all(control <= total+1e-9) and torch.all(amp <= total+1e-9)
            values = torch.stack([torch.ones_like(total), total, control, amp], 1)
            acc += torch.as_tensor(masks[:, sl], device='cuda').double() @ values
        tmp = dst.with_suffix('.tmp.npy')
        np.save(tmp, acc.cpu().numpy())
        tmp.replace(dst)


def report():
    files = sorted(OUT.glob('[0-9]*.npy'), key=lambda f: int(f.stem))
    assert len(files) == 1500
    a = np.stack([np.load(f) for f in files])
    assert a[:, 0, 0].sum() == 1185192
    rng = np.random.default_rng(20261124)
    samples = np.concatenate([a[rng.integers(len(a), size=(100, len(a)))].sum(1) for _ in range(100)])
    total = a.sum(0)
    result = {}
    for j, group in enumerate(GROUPS):
        row = dict(positions=int(total[j, 0]), mean_kl=total[j, 1]/total[j, 0])
        for k, name in enumerate(('control', 'magnitude'), 2):
            ratio = samples[:, j, k] / samples[:, j, 1]
            row[name] = dict(share=total[j,k]/total[j,1], ci95=np.quantile(ratio,[.025,.975]).tolist(),
                             between=total[j,k]/total[j,0], within=(total[j,1]-total[j,k])/total[j,0])
        result[group] = row
    (HERE/'results.json').write_text(json.dumps({'preflight':result}, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    if sys.argv[1]=='report':
        report()
    else:
        run(int(sys.argv[1]))
