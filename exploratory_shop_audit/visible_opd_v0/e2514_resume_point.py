"""Keep a recoverable, hash-verified Stage E state on the retained experiment volume."""
import os
import subprocess
import sys
from .e25_paths import RUN, SERIAL
from .e2514_watch import OUT, commit
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


def protect(update):
    root = RUN/SERIAL
    path = root/'pilot'/f'{update:02}'/'R'
    src = root/'state'/f'{update:02}'/'R'
    keep = RUN/'migration_2514/resume_snapshots'/f'{update:03}'
    optimizer = read(path/'optimizer.json')
    assert (optimizer['arm'],optimizer['update']) == ('R',update)
    assert optimizer['before_sha'] == read(path/'target_manifest.json')['scorer_sha']
    for name in ('adam.pt','adapter/adapter_config.json','adapter/adapter_model.safetensors','adapter/README.md'):
        dest = keep/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(src/name, dest)
    for name,key in [('adam.pt','adam_sha256'),('adapter/adapter_model.safetensors','safetensors_sha256')]:
        assert digest_file(keep/name) == optimizer[key]
    save(keep/'optimizer.json',optimizer)
    subprocess.run(['sync','-f',str(keep)],check=True)
    return keep


def publish(update,keep):
    root = RUN/SERIAL
    path = root/'pilot'/f'{update:02}'/'R'
    receipt = read(root/'arms/R/updates'/f'{update:03}.json')
    optimizer = read(keep/'optimizer.json')
    assert receipt['after_sha'] == optimizer['after_sha']
    stats = read(path/'train_statistics.json')
    state = dict(arm='R', update=update, status='INTERMEDIATE_CHECKPOINT',
                 last_behavior_step=update//10*10,
                 private_resume_directory=str(keep.relative_to(RUN)), update_receipt=receipt,
                 optimizer=optimizer, training=stats, precision_metrics=read(path/'measurement.json'),
                 frozen_target_manifest_sha256=digest_file(path/'target_manifest.json'),
                 routing_sha256=read(root/'settings.json')['routing_sha'])
    public = OUT/'R/resume'/f'{update:03}.json'
    save(public,state)
    commit(f'R-resume-{update}',[public])


if __name__=='__main__':
    step=int(sys.argv[1])
    keep=protect(step)
    if len(sys.argv)>2 and sys.argv[2]=='publish':
        publish(step,keep)
