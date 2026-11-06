"""Audited question sharding around frozen math functions; no scoring edits."""
from pathlib import Path
import argparse,fcntl,os,sys,time,traceback
R=Path(__file__).resolve().parent;H=R.parent;ROOT=H.parent
sys.path.insert(0,str(H))
import exp2_common as C,exp6_math_slot as E,math_slot_core as M
A=R/'math_two_gpu_after_7b'

def split_questions(rows, existing_partial=()):
    """Balance pending work without outcome access; partial questions stay primary."""
    parts={'primary':[],'helper':[]};partial=set(existing_partial)
    for r in rows:
        who='primary' if r['id'] in partial else min(parts,key=lambda k:len(parts[k]))
        parts[who].append(r['id'])
    return parts

def compare_probes(left,right):
    import numpy as np
    assert left.shape==right.shape and np.isfinite(left).all() and np.isfinite(right).all()
    tv=np.abs(np.exp(left.astype('float64'))-np.exp(right.astype('float64'))).sum(axis=-1)/2
    # Sampling amplification deserves a stricter cross-card check than TV alone.
    return dict(passed=bool(np.all(tv<.005) and np.array_equal(left,right)),
                tv=tv.tolist(),bitwise_equal=bool(np.array_equal(left,right)),tv_threshold=.005)

def worker(stage,role):
    import numpy as np,torch,slot_model_loader as L,token_diagnostic_backend as B
    from transformers import AutoTokenizer
    plan=C.read_json(A/'plan.json');assignment=C.read_json(A/(stage+'_assignment.json'))
    assert all(C.sha_file(Path(p))==s for p,s in plan['orchestration_source'].items())
    assert E.fingerprint()==plan['math_source']
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpus'][role] and torch.cuda.device_count()==1
    original_prepared=E.prepared;meta,all_rows=original_prepared();by={r['id']:r for r in all_rows}
    mine=[by[q] for q in assignment['parts'][role]]
    folder=A/stage/role;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scorer,model=L.load_model('32b',E.MODEL_ROOT/'Qwen2.5-32B-Instruct',ROOT)
        tok=AutoTokenizer.from_pretrained(str(E.MODEL_ROOT/'Qwen2.5-32B-Instruct'),local_files_only=True)
        model.update(runtime=B.runtime(),generation=E.CONFIG,source=E.fingerprint(),dependencies_sha=E.dep_digest())
        prior=C.read_json(E.OUT/'stages/screen/model.json')
        normalized=dict(model,runtime=dict(model['runtime'],cuda_visible_devices=prior['runtime']['cuda_visible_devices']))
        assert normalized==prior,'Model/runtime changed beyond explicitly assigned GPU UUID'
        target=E.OUT/'stages'/('dual_'+stage+'_'+role);target.mkdir(parents=True,exist_ok=True)
        if (target/'model.json').exists():assert C.read_json(target/'model.json')==model
        else:C.save_json(target/'model.json',model)
        eos=C.read_json(E.MODEL_ROOT/'Qwen2.5-32B-Instruct/generation_config.json')['eos_token_id']
        eos=[eos] if isinstance(eos,int) else eos
        annotations=None;annsha=None
        if stage=='evaluate':
            annotations=E.cut_annotations('full',plan['target']);assert annotations
            annsha=C.sha_file(E.OUT/'full/cut_annotations.json')
        requests=[]
        for q in assignment['all_ids']:
            row=by[q]
            if stage=='screen':requests.append(M.prompt(tok,row['problem'],'INDEPENDENT'))
            else:
                st=C.read_json(E.OUT/'student'/(q+'.json'))
                prefix=M.validate_annotation(st['generation']['text'],annotations['questions'][q])
                requests.extend(M.prompt(tok,row['problem'],arm,prefix) for arm in M.ARMS)
        assert requests
        probes=[min(requests,key=lambda x:len(x['input_ids'])),max(requests,key=lambda x:len(x['input_ids']))]
        E.gate(scorer,tok,probes,target/'gate_cross_card_inputs.json')
        vectors=[]
        with torch.inference_mode():
            for req in probes:
                ids=req['input_ids'];cache,h=scorer._feed(ids,len(ids)-1)
                vectors.append(torch.log_softmax(scorer.model.lm_head(h[-1].float()).float(),-1).cpu().numpy())
                del cache,h
        tmp=folder/'probe.npy.tmp'
        with tmp.open('wb') as f:np.save(f,np.stack(vectors))
        tmp.replace(folder/'probe.npy')
        attempt=os.environ['MATH_DUAL_ATTEMPT']
        C.save_json(folder/'probe.json',dict(attempt=attempt,request_sha256=E.sha(probes),model_sha256=C.sha_file(target/'model.json'),
                    vectors_sha256=C.sha_file(folder/'probe.npy'),source=plan['orchestration_source']))
        while not all((A/stage/r/'probe.json').exists() and C.read_json(A/stage/r/'probe.json').get('attempt')==attempt for r in ('primary','helper')):time.sleep(5)
        other='helper' if role=='primary' else 'primary'
        peer=C.read_json(A/stage/other/'probe.json')
        assert peer['request_sha256']==E.sha(probes) and peer['source']==plan['orchestration_source']
        assert peer['vectors_sha256']==C.sha_file(A/stage/other/'probe.npy')
        result=compare_probes(np.load(folder/'probe.npy'),np.load(A/stage/other/'probe.npy'))
        C.save_json(folder/'cross_card_check.json',result);assert result['passed'],'Cross-GPU probe failed; do not merge configurations'
        # All outputs are disjoint by question. Original scientific functions are reused.
        def load(name,requested_stage):
            assert name=='32b';original_prepared()
            if role=='primary':
                canonical=E.OUT/'stages'/requested_stage/'model.json'
                if canonical.exists():assert C.read_json(canonical)==model
                else:C.save_json(canonical,model)
            return scorer,tok,eos,target
        E.load=load
        E.status=lambda state,**kw:C.save_json(folder/'status.json',dict(stage=state,time=time.time(),pid=os.getpid(),**kw))
        if stage=='screen':
            def prepared_subset():
                original_prepared();return meta,mine
            E.prepared=prepared_subset
            E.screen(plan['target'])
            files=[E.OUT/'screen'/(r['id']+'.json') for r in mine]
            files += [E.OUT/'screen_generations'/r['id']/(str(i)+'.json') for r in mine for i in range(4)]
        else:
            def cuts(*args):
                assert C.sha_file(E.OUT/'full/cut_annotations.json')==annsha
                return annotations
            E.cut_annotations=cuts;E.selected=lambda *args:mine
            E.evaluate('full',plan['target'])
            files=[E.OUT/'full/responses'/r['id']/(arm+'.json') for r in mine for arm in M.ARMS]
        C.save_json(folder/'completion.json',dict(passed=True,role=role,gpu=plan['gpus'][role],questions=[r['id'] for r in mine],
                   outputs={str(p):C.sha_file(p) for p in files},source=plan['orchestration_source'],time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen','evaluate']);p.add_argument('role',choices=['primary','helper']);a=p.parse_args()
    try:worker(a.stage,a.role)
    except Exception:
        folder=A/a.stage/a.role;folder.mkdir(parents=True,exist_ok=True)
        (folder/'failure.log').write_text(traceback.format_exc());raise
