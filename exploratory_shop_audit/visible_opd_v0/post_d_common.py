"""Frozen inputs shared by the three post-pilot diagnostics."""
import os
import torch

from .c_smoke import state_hash
from .c_targets import actor
from .d_data import RUN, ARMS, folder, snapshot
from .s21_store import RUN as SECTION21
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read

POST=RUN/'post_d_diagnostics'
CASES=tuple((k,a) for k in (2,4,7) for a in ('R-beta','P','P-shuffled'))


def required_stage():
    launch=read(RUN/'training_complete.json')
    delivery=read(RUN/'reports/delivery.json')
    assert launch['updates']==delivery['completed_updates']==10 and delivery['episodes']==10000
    assert all(digest_file(RUN/'reports'/name)==sha for name,sha in delivery['public_sha256'].items())


def configure():
    os.umask(0o077)
    required_stage()
    rank=int(os.environ['LOCAL_RANK'])
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['NCCL_ALGO']='Ring'
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(rank)
    return rank


@torch.no_grad()
def start_at(model,k,a):
    if k==1:
        init=torch.load(SECTION21/'original/initial.pt',weights_only=True,map_location='cpu')
        for n,p in model.named_parameters():
            if p.requires_grad:p.copy_(init.pop(n))
        assert not init
    else:
        snapshot(model,k-1,a)
    return state_hash([(n,p) for n,p in model.named_parameters() if p.requires_grad])


def train_model():
    model=actor()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model.train()


def case_dir(k,a):
    return POST/'reset'/f'{k:02}'/a
