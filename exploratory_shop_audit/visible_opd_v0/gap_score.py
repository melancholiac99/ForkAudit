"""Two block-snapshot views on actor-nominated support for section 26."""
from collections import defaultdict
from pathlib import Path
import time

import torch

from .b_collect import PARENT, RUN as STAGE_B
from .c_targets import actor, response
from .e25_refresh import fold_of, tape
from .e2512_routing import owners
from .score import support
from exploratory_shop_audit.grpo_forced_audit.common import ROOT, digest_file, read, save

RUN = ROOT/'.runtime/runs/shop_gap26_20261123'
PUBLIC = Path(__file__).parent/'evidence/section26_20261123'


def target(logits, mask):
    if not mask.any(-1).all() or not logits[mask].isfinite().all():
        raise ValueError('Invalid candidate logits')
    return logits.float().masked_fill(~mask, -torch.inf).softmax(-1)


@torch.no_grad()
def block_views(block, row, candidates=None):
    answer = row['response']
    h, logits = response(block, row['views']['none'], answer)
    ids, mask = (support(logits, torch.as_tensor(answer, device=logits.device))
                 if candidates is None else candidates)
    s = logits.gather(1, ids)
    del logits
    _, full = response(block, row['views']['full'], answer)
    t = full.gather(1, ids)
    return dict(ids=ids, mask=mask, h=h, s=s, t=t, q=target(s, mask), teacher=target(t, mask))


def split_tasks():
    return read(PARENT/'asin_split_manifest.json')


def input_tapes(root):
    for rep in range(2):
        path = root/f'rollout_{rep}.jsonl'
        yield path if path.exists() else path.with_suffix('.jsonl.gz')


def score_inputs(rank, version, src, excluded, block, destination):
    destination.mkdir(exist_ok=True, parents=True)
    splits = split_tasks()
    tasks = splits['T_cal' if not version else 'T_opd']
    asins = {t['goal_index']:t['asin'] for t in tasks}
    routes = ({t['goal_index']:i % 4 for i,t in enumerate(tasks)} if not version else owners())
    seen, skipped, records = set(), 0, 0
    tensors, labels, goals, files = defaultdict(list), [], [], {}
    start = time.monotonic()
    for path in src:
        files[str(path.resolve())] = digest_file(path)
        for row in tape(path):
            g = row['goal']
            if routes[g] != rank or asins[g] in excluded:
                continue
            seen.add(g)
            if any(row['na'][mode] for mode in ('none','full')):
                skipped += 1
                continue
            values = block_views(block, row)
            for name, tensor in values.items():
                tensors[name].append(tensor.cpu())
            records += 1
            goals.extend([g]*len(row['response']))
            labels.extend(row['classes'])
    data = {k:torch.cat(v) for k,v in tensors.items()}
    data.update(goal=goals, cls=labels, fold=[fold_of(asins[g], splits['cal_folds']) for g in goals])
    out = destination/f'features_{rank}.pt'
    temp = out.with_suffix('.tmp')
    torch.save(data, temp)
    temp.replace(out)
    save(destination/f'score_{rank}.json', dict(records=records, positions=len(goals), goals=sorted(seen),
         skipped=skipped, input_hashes=files, sha=digest_file(out), seconds=time.monotonic()-start))


def bench_score(rank):
    from .e25_layout import Layout
    block = actor(load_on_rank=True).eval()
    score_inputs(rank, 0, sorted(STAGE_B.glob('tape_*.jsonl')), set(), block, RUN/'initial')
    old = Layout('formal_2512')
    old.snapshot(block, 10, 'P')
    excluded = {t['asin'] for k in range(11,16) for t in read(old.folder(k,'P').parent/'tasks.json')}
    paths = [p for k in range(6,11) for p in input_tapes(old.folder(k,'P'))]
    score_inputs(rank, 10, paths, excluded, block, RUN/'timing_refresh')


if __name__ == '__main__':
    from .e25_probe import configure
    bench_score(configure())
