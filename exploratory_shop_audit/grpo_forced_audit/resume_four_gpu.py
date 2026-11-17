"""Explicit post-reboot entry: verify CUDA, retain data, resume checkpoint1."""
import os
import subprocess
from pathlib import Path
from .common import *
from .budget import accounting_lock, execute
from .campaign import checkpoint, checkpoint_complete
from .train import launch


def recover_leases(run):
    import psutil
    boot=psutil.boot_time()
    with accounting_lock(run):
        ended={r['session_id'] for r in read_rows(run/'cost_ledger.jsonl') if r['event']=='end' and r.get('session_id')}
        for path in (run/'active_gpu_sessions').glob('*.json'):
            row=read(path)
            assert row['started_unix']<boot, 'A current-boot GPU owner exists; do not steal its lease'
            if row['session_id'] not in ended:
                append(run/'cost_ledger.jsonl',{'event':'end','time':utc(),'stage':row['stage'],
                    'session_id':row['session_id'],'gpu_ids':row['gpu_ids'],'gpu_count':row['gpu_count'],
                    'gpu_hours':(boot-row['started_unix'])*row['gpu_count']/3600,
                    'exit_code':None,'reason':'host_reboot_after_driver_failure',
                    'cost_bound':'conservative occupancy through host boot time'})
            archive=run/'recovered_gpu_sessions';archive.mkdir(exist_ok=True)
            path.rename(archive/path.name)


def main():
    run=DEFAULT_RUN.resolve()
    status=run/'campaign_status.json'
    if status.exists() and read(status).get('status')=='complete':return
    # Do not resume in the known broken HMM configuration.
    subprocess.run(['modprobe','nvidia_uvm'],check=True)
    assert Path('/sys/module/nvidia_uvm/parameters/uvm_disable_hmm').read_text().strip()=='Y'
    recover_leases(run)
    env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    execute(run,'post_reboot_cuda_health',[GPU_PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m',__package__+'.cuda_health'],env)
    kernel=subprocess.check_output(['dmesg'],text=True)
    assert 'BUG: kernel NULL pointer dereference' not in kernel and 'reboot is needed' not in kernel, 'Kernel fault after CUDA teardown'
    save(run/'post_reboot_health.json',{'time':utc(),'passed':True,'hmm_disabled':True,'cuda_nccl_process_exit_checked':True})
    continuation=run/'recovery_0042/continuation_authorization.json'
    if continuation.exists():
        assert read(continuation)['researcher_approved'] and read(continuation)['canonical_logs_prepared']
    if not continuation.exists() and (run/'recovery_0042/request.json').exists() and read(run/'recovery_0042/status.json')['status']!='complete':
        from .recover_training import resume
        return resume(run)
    existing=[int(p.name.split('_')[-1]) for p in (run/'training/checkpoints').glob('global_step_*') if checkpoint_complete(run,int(p.name.split('_')[-1]))]
    latest=max(existing)
    if latest<2:launch(run,2,checkpoint(run,latest));latest=2
    for rank in range(4):assert read(run/'training'/f'restore_rank_{rank}.json')['passed']
    assert read(checkpoint(run,latest)/'actor/replicated_checkpoint.json')['all_rank_fingerprints_equal']
    from .campaign import main as campaign
    campaign()


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        append(DEFAULT_RUN/'errors.jsonl',{'time':utc(),'stage':'four_gpu_post_reboot','error':traceback.format_exc()})
        raise
