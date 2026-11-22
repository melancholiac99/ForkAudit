"""Validate the retained R@27 state and the frozen inputs for update 28 on four GPUs."""
import importlib.metadata
import subprocess

import torch

from .e2514_gate import control_authorized, resume_verification_dir
from .e2512_routing import ROUTING
from .e25_layout import Layout
from .e25_paths import PUBLIC, SERIAL
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


def validate():
    layout=Layout(SERIAL)
    control_authorized(layout)
    verify=resume_verification_dir()
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                                      '--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(hardware)==torch.cuda.device_count()==4
    assert all('A100' in row and int(row.split(',')[1])>=80000 for row in hardware)
    previous=read(layout.folder(27,'R')/'optimizer.json')
    archive=read(PUBLIC/'migration_2514/R/resume/027.json')
    assert previous==archive['optimizer']
    state=layout.state(27,'R')
    assert digest_file(state/'adam.pt')==previous['adam_sha256']
    assert digest_file(state/'adapter/adapter_model.safetensors')==previous['safetensors_sha256']
    path=layout.folder(28,'R')
    manifest=read(path/'target_manifest.json')
    collection=read(path/'collection.json')
    assert previous['after_sha']==manifest['scorer_sha']
    assert collection['checkpoint']==previous['safetensors_sha256']
    assert collection['routing_sha']==digest_file(ROUTING)==archive['routing_sha256']
    for rep,expected in enumerate(collection['rollout_sha256']):
        assert digest_file(path/f'rollout_{rep}.jsonl')==expected
    for name,key in [('student','student'),('target','target'),('candidate_mass','mass'),('bf16_before','bf16_before')]:
        assert digest_file(path/f'{name}.pt')==manifest[key+'_sha256']
    result=read(verify/'result.json')
    assert result['replay']['replayed']==result['replay']['archive']
    out=PUBLIC/'migration_2514/four_card_resume'/verify.name/'handoff.json'
    save(out,dict(arm='R',last_completed=27,resumed_update=28,training_rows=collection['records'],
                  hardware=hardware,torch=torch.__version__,cuda=torch.version.cuda,
                  vllm=importlib.metadata.version('vllm'),prior_adapter_sha256=previous['safetensors_sha256'],
                  prior_adam_sha256=previous['adam_sha256'],target_sha256=manifest['target_sha256'],
                  rollout_sha256=collection['rollout_sha256'],routing_sha256=collection['routing_sha'],
                  migration_checks=result))
    return out


if __name__=='__main__':
    from .e2514_watch import commit
    file=validate()
    commit('four-card-resume-'+file.parent.name,[file])
