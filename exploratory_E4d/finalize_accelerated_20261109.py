"""Latest authorization: complete resumed E4d, verify/upload, then shutdown."""
import os,sys,time,traceback
from pathlib import Path
import accelerate_20261108 as A
import finalize_server as F
F.R=A.R
F.FIN=A.H/'finalization_accelerated_20261109'
original_command=F.command

def command(args,**kwargs):
    words=list(map(str,args))
    if len(words)==3 and words[1:]==[str(A.H/'run_server.py'),'--verify']:
        words=[words[0],str(A.H/'accelerate_20261108.py'),'--verify']
    return original_command(words,**kwargs)
F.command=command

# Archive/commit immutable results, excluding only active monitor control files.
# F.artifacts uses the current E.DATA and F.R, both bound to the accelerated run.
_original_artifacts=F.artifacts

def artifacts():
    return sorted(set(_original_artifacts()+[A.H/'ACCELERATED_RESUME_20261109.txt']))
F.artifacts=artifacts


def live_experiment_processes():
    needles={'run_server.py','e4d.py','accelerate_20261108.py','resume_devices_20261108.py',
        'run_deterministic_20261107.py','finish_7b_then_dual_math.py','math_two_gpu_worker.py',
        'exp5_token_slot.py','exp6_math_slot.py'}
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            words=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if words and 'python' in Path(words[0]).name and any(Path(w).name in needles for w in words[1:]) and F.alive(int(p.name)):
                found.append(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return found
F.live_experiment_processes=live_experiment_processes


def recover(error,thread):
    try:dest=F.snapshot(error)
    except Exception:
        dest=F.FIN/'snapshot_error.json'
        A.E.save_json(dest,dict(error=error,snapshot_error=traceback.format_exc(),original_results_retained=True));os.sync()
    claim=F.FIN/'recovery_claim.json'
    if claim.exists():claim.rename(F.FIN/f'recovery_claim_previous_{time.time_ns()}.json')
    F.state('requesting_recovery',error=error,snapshot=str(dest))
    message=(f'【E4d 20261109恢复/收尾异常】用户最新授权恢复E4d，核验并上传credit-grok后关机；尽力修复，实在修不好保存日志结果、sync后关机。'
        f'读取 {F.FIN}/policy.json、status.json、{dest}；接管先写 {claim}。'
        f'当前队列 {A.R}，数据 {A.E.DATA}，核验入口 accelerate_20261108.py --verify。'
        '保持加速配置、原协议和门槛，不改旧plan或结果，不重跑已完成记录，不停止另一正常GPU工作。'
        '续跑使用accelerate_20261108.py，不要启动旧run_server或旧finalizer。'
        '上传失败可重试；远端分歧用独立detached worktree审查，不强推。'
        f'修复后启动 {Path(__file__).resolve()} --thread {thread}。'
        'hindsight已发布7B与数学保留，暂停的H80032B不在当前6000D重启。旧队列消息中的不得关机已被最新用户授权覆盖。'
        '监控在30分钟无人接管且GPU与实验进程都空闲时保全日志后关机。')
    queued=False
    for attempt in range(3):
        try:
            out=command(['codex','queue','--thread',thread,'--resume','--message',message],timeout=60)
            (F.FIN/'recovery_queue.log').write_text(out+'\n');queued=True;break
        except Exception:
            with (F.FIN/'recovery_queue.log').open('a') as log:log.write(traceback.format_exc())
            if attempt<2:time.sleep(10)
    started=time.monotonic()
    while True:
        if (F.FIN/'cancelled.json').exists():F.state('cancelled');return
        if claim.exists():F.state('recovery_handoff',claim=A.E.load(claim));return
        if time.monotonic()-started>1800 and F.gpu_idle() and not live_experiment_processes():
            try:dest=F.snapshot(error+'\nNo recovery claim after 30 minutes; GPU/experiments idle.')
            except Exception:
                A.E.save_json(F.FIN/'final_snapshot_error.json',dict(error=traceback.format_exc(),original_results_retained=True));os.sync()
            F.shutdown('recovery_unavailable',snapshot=str(dest),recovery_queued=queued);return
        time.sleep(30)
F.recover=recover

if __name__=='__main__':F.main()
