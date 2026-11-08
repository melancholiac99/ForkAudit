from pathlib import Path
import argparse,sys,time
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
import resume_devices_20261108 as M
E=M.E;R=Path(__file__).resolve().parent
CONFIGS={
 'baseline':dict(M.N.ENGINE_CHANGES),
 'cached':dict(M.N.ENGINE_CHANGES,enable_prefix_caching=True),
 'graph':dict(M.N.ENGINE_CHANGES,enforce_eager=False,enable_prefix_caching=True,
   compilation_config={'mode':0,'cudagraph_mode':'FULL','cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,200,256]}),
}

def main(mode):
    M.checked_plan();work=E.load(R/'workload.json')
    import vllm
    original=vllm.LLM
    class BenchEngine(original):
        def __init__(self,*args,**kwargs):
            kwargs.update(CONFIGS[mode]);super().__init__(*args,**kwargs)
    vllm.LLM=BenchEngine
    from e4_policy import LoraPolicy
    loaded=time.monotonic();pol=LoraPolicy();load_seconds=time.monotonic()-loaded
    results={};times={};tokens={};checks={};smoke={}
    ref=E.load(Path(M.checked_plan()['reference']))
    allpts=sorted(E.zpoints(E.locked()),key=lambda p:p['point_id'])
    smoke_prompts=[allpts[0]['prompt'],allpts[-1]['prompt']]
    try:
        for arm,seed in [('base',None),('Azero',2)]:
            key=f'{arm}_{seed}'
            pol.set_adapter(None,0) if arm=='base' else pol.set_adapter(str(E.ADAPTER_DIR/f'{arm}_s{seed}'),1+M.D.ALL_ARMS.index(arm)*3+seed)
            warm=pol.sample(smoke_prompts,ref['seeds'],logprobs=False)
            smoke[key]=warm
            for j,indices in enumerate(work['batches']):
                rows=[work['rows'][i] for i in indices];start=time.monotonic()
                outputs=pol.sample([r['prompt'] for r in rows],[r['seed'] for r in rows],logprobs=False)
                k=f'{key}_batch{j}';times[k]=time.monotonic()-start;results[k]=outputs
                tokens[k]=sum(len(x[0]['token_ids']) for x in outputs)
            rows=[work['rows'][i] for i in work['batches'][1]]
            repeated=pol.sample([r['prompt'] for r in rows],[r['seed'] for r in rows],logprobs=False)
            checks[key+'_repeat_exact']=repeated==results[f'{key}_batch1']
        if mode!='baseline':
            for arm in M.D.ALL_ARMS:
                for seed in M.D.SEEDS:
                    key=f'{arm}_{seed}'
                    if key in smoke:continue
                    pol.set_adapter(str(E.ADAPTER_DIR/f'{arm}_s{seed}'),1+M.D.ALL_ARMS.index(arm)*3+seed)
                    smoke[key]=pol.sample(smoke_prompts,ref['seeds'],logprobs=False)
            checks['all_26_reference_responses_exact']=smoke==ref['outputs']
            baseline=E.load(R/'baseline.json')
            checks['all_128_workload_responses_exact']=results==baseline['results']
    finally:pol.close()
    E.save_json(R/f'{mode}.json',dict(mode=mode,engine_changes=CONFIGS[mode],environment=M.N.ENV_CHANGES,
        workload_sha256=E.sha_file(R/'workload.json'),load_seconds=load_seconds,times=times,
        total_seconds=sum(times.values()),tokens=tokens,results=results,smoke=smoke,checks=checks,
        passed=all(checks.values()),timing_caveat='Fixed initial-state prompt batches; not an end-to-end replay ETA. Repeat timings excluded.'))
    print(mode,'seconds',sum(times.values()),'checks',checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=CONFIGS);a=p.parse_args();main(a.mode)
