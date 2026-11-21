"""Isolated experiment paths, frozen task routing and current projector selection."""
from functools import cached_property
import torch
from .c_targets import STAGE_B
from .d_data import OnlineBatch,PARENT,SMOKE,load_adapter,schedule as pilot_schedule
from .e25_probe import RUN
from .projection import VisibleProjection,candidate_target
from .s21_data import SmokeBatch
from exploratory_shop_audit.grpo_forced_audit.common import read,save,digest_file

class Layout:
    def __init__(self,run):
        if run not in ('formal','calibration','calibration_2511','p_kl'):raise ValueError(run)
        self.name=run;self.root=RUN/run
    def folder(self,k,a):return self.root/'pilot'/f'{k:02}'/a
    def state(self,k,a):return self.root/'state'/f'{k:02}'/a
    def snapshot(self,model,k,a):
        if k==0:return
        p=self.state(k,a)/'adapter/adapter_model.safetensors'
        load_adapter(model,p,read(self.folder(k,a)/'optimizer.json')['safetensors_sha256'])
    @cached_property
    def config(self):
        amended = self.root / 'settings_2511.json'
        if self.name == 'formal' and amended.exists():
            return read(amended)
        return read(self.root/'settings.json')
    def batch(self,k,a):
        if k==1 and a!='P-KL':
            batch=SmokeBatch();return batch,batch.targets('R' if a=='R-matched' else a)
        b=OnlineBatch(k,a,self.folder(k,a));return b,b.q
    def version(self,k,a):
        return ((k-1)//10)*10 if a=='P' else 0
    def projector_dir(self,k,a):
        if a=='P-KL':return RUN/'offline'
        version=self.version(k,a)
        return self.root/'refresh'/f'{version:03}' if version else STAGE_B/'projector_versions'
    def projector_file(self,k,a,f):
        return self.projector_dir(k,a)/(f'kl_{f}.pt' if a=='P-KL' else f'content_{f}.pt')
    def projector_sha(self,k,a):
        return [digest_file(self.projector_file(k,a,f)) for f in range(4)] if a in ('P','P-KL') else []
    def projectors(self,model,k,a):
        result={}
        if a in ('P','P-KL'):
            for f in range(4):
                p=self.projector_file(k,a,f)
                if self.version(k,a):
                    meta=read(p.parent/f'fit_{f}.json')
                    assert meta['actor_sha']==read(self.folder(self.version(k,a),'P')/'optimizer.json')['after_sha']
                    assert meta['projector_sha']==digest_file(p)
                net=VisibleProjection(model.config.hidden_size).cuda().eval()
                net.load_state_dict(torch.load(p,weights_only=True));result[f]=net.requires_grad_(False)
        return result
    @cached_property
    def kl_scales(self):return read(RUN/'offline/complete.json')['deployment_oof_shrink']
    def projected(self,nets,a,h,head,ids,s,mask,q,f,cal,classes):
        d=nets[f](h,head[ids],q)
        coeff=[cal[c]['b'] if a=='P' else self.kl_scales[c] for c in classes]
        return candidate_target(s,d,torch.tensor(coeff,device=s.device)[:,None],mask)
    def initialize(self,n):
        split=read(PARENT/'asin_split_manifest.json')
        held={t['goal_index'] for pool in ('T_pre','V','T_cal','T_opd_eval') for t in split[pool]}
        tasks=[t for t in split['T_opd'] if t['goal_index'] not in held][:32*n]
        assert len(set(t['goal_index'] for t in tasks))==32*n
        schedule=[tasks[i:i+32] for i in range(0,len(tasks),32)]
        assert schedule[:10]==pilot_schedule()
        for k,group in enumerate(schedule,1):
            for a in self.config['rates']:
                path=self.folder(k,a);path.mkdir(parents=True,exist_ok=True)
                if not (path/'snapshots').exists():(path/'snapshots').symlink_to(PARENT/'snapshots',target_is_directory=True)
            dest=path.parent/'tasks.json'
            if dest.exists():assert read(dest)==group
            else:save(dest,group)
