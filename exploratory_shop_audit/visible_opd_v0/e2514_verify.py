"""Check both physical four-GPU groups against immutable serial-run references."""
import argparse
import os
from pathlib import Path

import torch
from . import d_targets, d_train
from .c_targets import actor, response
from .e25_layout import Layout
from .e25_measure import at
from .e25_probe import RUN, configure, pooled
from .e2511_targets import key
from .e2512_routing import SERIAL, owners
from .s21_data import SmokeBatch
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save

ROOT = RUN/'migration_2514'
ARM='P'
STEP=6


class Archive(Layout):
    def state(self,k,a):
        if k==5:
            return Layout('formal').state(5,a)
        return super().state(k,a)


@torch.inference_mode()
def probe(rank, source, dest):
    batch=SmokeBatch()
    model=actor().eval()
    at(model,source,0,'B')
    ref=source.root/'base_logits/torch.bfloat16'/f'rank_{rank}.pt'
    old=torch.load(ref,mmap=True,weights_only=True,map_location='cpu')
    chunks=pooled(batch,rank,owners())
    assert set(old)=={ids[0] for ids in chunks}
    mismatches=total=positions=0
    for group in chunks:
        i=group[0]; row=batch.rows[i]
        raw=response(model,row['prompt'],row['answer'])[1].to(torch.bfloat16).cpu()
        assert old[i].dtype==torch.bfloat16 and old[i].shape==raw.shape
        mismatches+=int(torch.count_nonzero(old[i].view(torch.int16)!=raw.view(torch.int16)))
        total+=raw.numel()
        positions+=len(group)*len(row['answer'])
    save(dest/f'probe_{rank}.json',dict(rank=rank,checkpoint='B@0',reference=digest_file(ref),
         elements=total,mismatches=mismatches,positions=positions,passed=mismatches==0))


def targets(rank,source,dest):
    folder=source.folder(STEP,ARM)
    native=folder/'native'/f'rank{rank}'
    receipt=read(native/'complete.json')
    goals=set(receipt['goals'])
    assert all(owners()[g]==rank for g in goals)
    out=dest/'parts'/str(rank)
    d_targets.build(STEP,ARM,layout=source,goals=goals,output=out,
                    rollout_root=native,rollout_hashes=receipt['rollout_sha256'])
    actual=digest_file(out/'target.pt')
    expected=digest_file(folder/'target_parts'/str(rank)/'target.pt')
    save(dest/f'target_{rank}.json',dict(rank=rank,actual=actual,expected=expected,passed=actual==expected))


def replay(rank,source,dest):
    original=read(source.folder(STEP,ARM)/'optimizer.json')
    def state(k,a):
        return source.state(k,a) if k<STEP else dest/'replay/state'/f'{k:02}'/a
    def folder(k,a):
        return source.folder(k,a) if k<STEP else dest/'replay/pilot'/f'{k:02}'/a
    folder(STEP,ARM).mkdir(parents=True,exist_ok=True)
    d_train.train(STEP,arms=[ARM],folder=folder,state_dir=state,snapshot=source.snapshot,
                  load_batch=source.batch,rates=source.config['rates'],compare_original=False,replay_state=original)
    if rank==0:
        rec=read(folder(STEP,ARM)/'optimizer.json')
        fields=('before_sha','after_sha','adam_sha256','safetensors_sha256','minibatch_sha256','cumulative_steps')
        save(dest/'replay.json',dict(archive={f:original[f] for f in fields},replayed={f:rec[f] for f in fields},
             passed=rec['after_sha']==original['after_sha'],adam_identical=rec['adam_sha256']==original['adam_sha256']))


def summarize(source,dest):
    rows=[]; targets_=[]
    for rank in range(4):
        p=dest/'parts'/str(rank)
        rows.extend(torch.load(p/'student.pt',weights_only=True))
        targets_.extend(torch.load(p/'target.pt',weights_only=True))
    indices=sorted(range(len(rows)),key=lambda idx:key(rows[idx]))
    merged=dest/'target.pending'
    torch.save([targets_[idx] for idx in indices],merged)
    actual=digest_file(merged); archive=source.folder(STEP,ARM)
    expected=read(archive/'target_manifest.json')['target_sha256']
    assert digest_file(archive/'target.pt')==expected
    probes=[read(dest/f'probe_{i}.json') for i in range(4)]
    assert sum(p['positions'] for p in probes)==4448
    result=dict(probes=probes,targets=dict(expected=expected,actual=actual,passed=actual==expected,
                shards=[read(dest/f'target_{i}.json') for i in range(4)]),replay=read(dest/'replay.json'))
    result['passed']=all(p['passed'] for p in probes) and result['targets']['passed'] and result['replay']['passed']
    save(dest/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['probe','targets','replay','summary'])
    parser.add_argument('group',choices=['A','B','single'])
    args=parser.parse_args();source=Archive(SERIAL);dest=ROOT/args.group
    if args.group=='single':
        from .e2514_gate import resume_verification_dir
        dest=resume_verification_dir()
    dest.mkdir(exist_ok=True,parents=True)
    if args.phase=='summary':
        summarize(source,dest)
    else:
        assert os.environ['CUDA_VISIBLE_DEVICES']==('4,5,6,7' if args.group=='B' else '0,1,2,3')
        rank=configure()
        {'probe':probe,'targets':targets,'replay':replay}[args.phase](rank,source,dest)
