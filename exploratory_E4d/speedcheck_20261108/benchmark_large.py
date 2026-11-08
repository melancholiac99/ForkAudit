from pathlib import Path
import sys,time,argparse
import benchmark as B
E,M,R=B.E,B.M,B.R

def main(mode):
    assert mode in ('baseline200','fast400')
    engine=dict(B.CONFIGS['baseline']) if mode=='baseline200' else dict(B.CONFIGS['graph'],async_scheduling=True,
        compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,200,256,320,400,512]})
    batch=200 if mode=='baseline200' else 400
    pts=sorted(E.zpoints(E.locked()),key=lambda p:p['point_id'])
    rows=[dict(id=p['point_id'],prompt=p['prompt'],seed=M.D.SCREEN_OFFSET+314159+i*5+j)
          for i,p in enumerate(pts) for j in range(5)]
    assert len(rows)==800
    import vllm
    original=vllm.LLM
    class Engine(original):
        def __init__(self,*args,**kwargs):kwargs.update(engine);super().__init__(*args,**kwargs)
    vllm.LLM=Engine
    from e4_policy import LoraPolicy
    start=time.monotonic();pol=LoraPolicy();load_seconds=time.monotonic()-start
    results=[];times=[]
    try:
        pol.set_adapter(str(E.ADAPTER_DIR/'Azero_s2'),3)
        pol.sample([pts[0]['prompt'],pts[-1]['prompt']],[M.D.SCREEN_OFFSET+987654,M.D.SCREEN_OFFSET+987655],logprobs=False)
        for k in range(0,len(rows),batch):
            chunk=rows[k:k+batch];t=time.monotonic()
            results.extend(pol.sample([x['prompt'] for x in chunk],[x['seed'] for x in chunk],logprobs=False))
            times.append(time.monotonic()-t)
            print('batch',k,'seconds',times[-1],flush=True)
    finally:pol.close()
    out=dict(mode=mode,engine_changes=engine,environment=M.N.ENV_CHANGES,batch=batch,
        workload_sha256=M.D.digest(rows),requests=len(rows),load_seconds=load_seconds,times=times,
        total_seconds=sum(times),tokens=sum(len(x[0]['token_ids']) for x in results),results=results,
        source_sha256=E.sha_file(Path(__file__)),caveat='Offline fixed prompts with five independent seeds/state; end-to-end replay speed must be measured after cutover.')
    if mode=='fast400':
        base=E.load(R/'baseline200.json');out['exact_outputs']=base['results']==results
        assert base['workload_sha256']==out['workload_sha256']
        out['speedup']=base['total_seconds']/out['total_seconds']
    E.save_json(R/(mode+'.json'),out)
    print({k:v for k,v in out.items() if k not in ('results','times')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline200','fast400']);a=p.parse_args();main(a.mode)
