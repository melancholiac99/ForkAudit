"""Keep separate verification/publication; latest user message forbids shutdown."""
import os
import time
import traceback
import finalize_server as F

def no_shutdown(outcome, **receipt):
    assert (F.FIN/'shutdown_hold.json').exists()
    F.E.save_json(F.FIN/'shutdown_skipped.json',dict(
        outcome=outcome,receipt=receipt,time=time.time(),
        reason='Latest user instruction: no shutdown authorization.'))
    os.sync()
    F.state('completed_published_no_shutdown',outcome=outcome,**receipt)

def recover_without_shutdown(error, thread):
    try:
        dest=F.snapshot(error)
    except Exception:
        dest=F.FIN/'snapshot_error.json'
        F.E.save_json(dest,dict(error=error,snapshot_error=traceback.format_exc(),original_results_retained=True))
        os.sync()
    F.state('failed_no_shutdown',error=error,snapshot=str(dest))
    message=('【联合核验/上传异常，无关机授权】请读 '+str(dest)+
        ' 和 '+str(F.FIN/'status.json')+'，尽力修复并继续。'
        'hindsight与credit-grok各自核验上传；不混配置、不改统计门槛、不停止正常实验。'
        '本次最新用户消息明确无关机授权，禁止关机。修复后启动 exploratory_E4d/finalize_without_shutdown.py --thread 当前线程。')
    for attempt in range(3):
        try:
            response=F.command(['codex','queue','--thread',thread,'--resume','--message',message],timeout=45)
            (F.FIN/'recovery_queue_no_shutdown.log').write_text(response+'\n')
            break
        except Exception:
            with (F.FIN/'recovery_queue_no_shutdown.log').open('a') as log:
                log.write(traceback.format_exc())
            if attempt<2:time.sleep(5)

F.shutdown=no_shutdown
F.recover=recover_without_shutdown
if __name__=='__main__':F.main()
