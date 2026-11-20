"""Stage D task order, snapshots, and the sealed student-only optimizer input."""
import os
from collections import Counter
import torch
from safetensors.torch import load_file
from .b_collect import PARENT
from .c_collect import RUN as SMOKE
from .s21_data import SmokeBatch, balanced_batches
from .s21_store import ARMS
from .s23_cross import RUN
from exploratory_shop_audit.grpo_forced_audit.common import read, save, digest_file


SCRATCH = RUN / 'state'


def folder(k, arm):
    return RUN/'pilot'/f'{k:02}'/arm


def state_dir(k, arm):
    return SCRATCH/f'{k:02}'/arm


def schedule():
    splits = read(PARENT/'asin_split_manifest.json')
    held = {t['goal_index'] for name in ('V','T_pre','T_cal','T_opd_eval') for t in splits[name]}
    chosen = [t for t in splits['T_opd'] if t['goal_index'] not in held][:320]
    assert len({t['goal_index'] for t in chosen}) == 320
    return [chosen[i:i+32] for i in range(0,320,32)]


def snapshot(model,k,arm):
    if k == 0:
        return
    path = state_dir(k,arm)/'adapter/adapter_model.safetensors'
    meta = read(folder(k,arm)/'optimizer.json')
    assert digest_file(path)==meta['safetensors_sha256']
    vals = load_file(str(path))
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:
                p.copy_(vals.pop(name.replace('.default.weight','.weight')))
    assert not vals


class OnlineBatch:
    local = SmokeBatch.local

    def __init__(self,k,arm):
        path = folder(k,arm)
        info = read(path/'target_manifest.json')
        for name in ('student','target'):
            assert digest_file(path/f'{name}.pt') == info[name+'_sha256']
        self.rows = torch.load(path/'student.pt',weights_only=True)
        self.q = torch.load(path/'target.pt',weights_only=True)
        assert len(self.q) == len(self.rows)
        expect = {'goal','rep','step','prompt','answer','ids','mask','classes','snapshot'}
        assert all(set(r)==expect for r in self.rows)
        keys = [(r['goal'],r['rep'],r['step']) for r in self.rows]
        assert keys == sorted(set(keys))
        n = Counter()
        for r in self.rows:
            n[r['goal'],r['rep']] += len(r['answer'])
        na_keys=[(x['goal'],x['rep'],x['step']) for x in info.get('na_records',[])]
        all_keys=sorted(keys+na_keys)
        self.total_records=len(all_keys)
        assert len(set(all_keys))==self.total_records==info.get('rollout_records',len(keys))
        assert {(g,rep) for g,rep,_ in all_keys}=={(g,rep) for g in info['goals'] for rep in range(2)}
        trajectories=Counter(g for g,rep in n)
        self.weights=[1/(len(trajectories)*trajectories[r['goal']]*n[r['goal'],r['rep']]) for r in self.rows]
        indices={key:i for i,key in enumerate(keys)}
        full_batches=balanced_batches(self.total_records,k)
        self.batch_sizes=list(map(len,full_batches))
        self.batches=[[indices[all_keys[j]] for j in part if all_keys[j] in indices] for part in full_batches]
        assert abs(sum(w*len(r['answer']) for w,r in zip(self.weights,self.rows))-1)<1e-12


def load_batch(k,arm):
    if k==1:
        batch = SmokeBatch()
        return batch,batch.targets(arm)
    batch = OnlineBatch(k,arm)
    return batch,batch.q


def initialize():
    from .c_collect import tasks
    from .c_targets import STAGE_B
    assert schedule()[0]==tasks()
    assert read(STAGE_B/'section19/P1_decision_v2.json')['groups']['click']['revised']=='PASS'
    assert read(STAGE_B/'reports/calibration_strength_report.json')['classes']['click']['b_ci95'][0]>0
    assert read(STAGE_B/'reports/acceptance.json')['passed']
    for k,task_list in enumerate(schedule(),1):
        for arm in ARMS:
            path=folder(k,arm)
            path.mkdir(parents=True,exist_ok=True)
            if not (path/'snapshots').exists():
                (path/'snapshots').symlink_to(PARENT/'snapshots', target_is_directory=True)
        save(path.parent/'tasks.json',task_list)
    SCRATCH.mkdir(mode=0o700,exist_ok=True)


if __name__=='__main__':
    os.umask(0o077)
    initialize()
