"""Prepare the fixed X/Y continuity sample before any manual outcome audit."""
from pathlib import Path
import fcntl,hashlib,os,sys,time,traceback,subprocess
R=Path(__file__).resolve().parent;H=R.parent;ROOT=H.parent
sys.path.insert(0,str(H))
import exp2_common as C,supplement_slot_analysis_20261106 as S
M=H/'results/math_slot_20261106/full';OUT=H/'results/next_slot_supplement_20261106'

def wake(text):
    thread=os.environ.get('CODEX_THREAD_ID');assert thread
    with (R/'full_blind_preparation_wake.log').open('a') as log:
        subprocess.run(['codex','queue','--thread',thread,'--resume','--message',text],stdout=log,stderr=subprocess.STDOUT,timeout=60)

def status(stage,**kw):C.save_json(R/'full_blind_preparation_status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))

def main():
    with (R/'full_blind_preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=C.read_json(OUT/'math_full_blind_preparation_plan.json')
            assert C.sha_file(Path(__file__))==plan['preparer_sha256']
            req=C.read_json(M/'cut_review_request.json');selection=C.read_json(OUT/'math_full_blind_sample_selection.json')
            assert C.sha_file(M/'cut_review_request.json')==selection['request_sha256']==plan['request_sha256']
            sample=S.continuity_sample(req['questions']);ids={q['id'] for q in sample}
            assert len(ids)==24 and [q['id'] for q in sample]==selection['ids']==plan['sample_ids']
            paths=[M/'responses'/qid/(arm+'.json') for qid in ids for arm in ('ACTIVE','QUOTE-CONTINUE')]
            status('waiting_for_48_selected_responses',sample_questions=len(ids))
            while not (M/'cut_annotations.json').exists() or not all(p.exists() for p in paths):time.sleep(15)
            assert C.sha_file(M/'cut_annotations.json')==plan['annotations_sha256']
            annotations=C.read_json(M/'cut_annotations.json')['questions'];blinded=[]
            for q in req['questions']:
                qid=q['id']
                if qid not in ids:continue
                prefix=q['student']['generation']['text'][:annotations[qid]['prefix_end']]
                active=C.read_json(M/'responses'/qid/'ACTIVE.json')['generation']['text']
                quote=C.read_json(M/'responses'/qid/'QUOTE-CONTINUE.json')['generation']['text']
                swap=int(hashlib.sha256(('blind:'+qid).encode()).hexdigest()[-1],16)%2
                left,right=(active,quote) if swap==0 else (quote,active)
                blinded.append(dict(id=qid,student_prefix_tail=prefix[-380:],continuation_X_opening=left[:450],
                    continuation_Y_opening=right[:450],question_for_reviewer='For X and Y, label genuine_continuation / restart_or_rewrite / ambiguous; no answer labels shown.'))
            payload=S.blind_payload(blinded);dest=OUT/'math_full_blind_review.json'
            if dest.exists():assert C.read_json(dest)==payload
            else:C.save_json(dest,payload)
            C.save_json(OUT/'math_full_blind_preparation_verification.json',dict(passed=True,
                payload_sha256=hashlib.sha256(C.json_bytes(payload)).hexdigest(),questions=24,
                raw_response_hashes={str(p):C.sha_file(p) for p in paths},request_sha256=plan['request_sha256'],
                annotations_sha256=plan['annotations_sha256'],no_correctness_or_condition_labels_in_payload=True,
                matches_frozen_supplement_payload_construction=True,time=time.time()))
            status('ready_for_blind_review',path=str(dest))
            wake('【数学全量续写盲审优先】请先只读 '+str(dest)+'，不读正确率、条件映射或 outcome_review_request.json；逐题对X/Y标注 genuine_continuation / restart_or_rewrite / ambiguous，保存同目录 math_full_blind_labels.json（blind_payload_sha256及samples）。固定24题，原prefix尾380字和响应开头450字；短开头不足以判别时保留ambiguous。完成盲审后再做全量输出核验。最新授权仍为hindsight及credit-grok各自核验上传后共同关机。')
        except Exception:
            error=traceback.format_exc();(R/'full_blind_preparation_failure.log').write_text(error);status('failed',error=error)
            wake('【数学全量盲审准备异常】请读 '+str(R/'full_blind_preparation_failure.log')+'，只修CPU盲审准备，不改原始响应、评分或统计门槛。保留GPU队列，先盲审再输出核验。');raise

if __name__=='__main__':main()
