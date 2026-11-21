"""Only if the control gate opens, reconstruct missing Adam/LoRA state using signed existing batches."""
import sys
from .e2514_gate import control_authorized
from . import d_train
from .d_data import load_adapter
from .e25_layout import Layout
from .e25_probe import configure
from .e2511_schedule import CAL,MATCH
from .e2513_adopt import RESTORE
from exploratory_shop_audit.grpo_forced_audit.common import read


def main():
    k=int(sys.argv[1]);rank=configure()
    serial=Layout('formal_2512')
    control_authorized(serial)
    old=Layout(CAL)
    original=read(old.folder(k,MATCH)/'optimizer.json')
    def folder(i,a):return RESTORE/'pilot'/f'{i:02}'/a
    def state(i,a):return RESTORE/'state'/f'{i:02}'/a
    def snapshot(model,i,a):
        if i:load_adapter(model,state(i,a)/'adapter/adapter_model.safetensors',read(old.folder(i,a)/'optimizer.json')['safetensors_sha256'])
    folder(k,MATCH).mkdir(exist_ok=True,parents=True)
    d_train.train(k,arms=[MATCH],folder=folder,state_dir=state,snapshot=snapshot,load_batch=old.batch,
                  rates=old.config['rates'],compare_original=False,replay_state=original)
    if rank==0:
        result=read(folder(k,MATCH)/'optimizer.json')
        for key in ('before_sha','after_sha','adam_sha256','safetensors_sha256','minibatch_sha256','cumulative_steps'):
            assert result[key]==original[key],(k,key)


if __name__=='__main__':main()
