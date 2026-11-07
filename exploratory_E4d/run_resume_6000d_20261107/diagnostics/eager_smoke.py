from pathlib import Path
import sys,os
H=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(H))
import resume_6000d_20261107 as M
if __name__=='__main__':
    import vllm
    Original=vllm.LLM
    class EagerLLM(Original):
        def __init__(self,*args,**kwargs):
            kwargs.update(enforce_eager=True,async_scheduling=False,enable_prefix_caching=False)
            super().__init__(*args,**kwargs)
    vllm.LLM=EagerLLM
    i=int(sys.argv[1]);dest=Path(sys.argv[2]);dest.mkdir(parents=True,exist_ok=True)
    original=M.E.save_json
    def save(path,content):
        assert Path(path)==M.R/f'lora_smoke_{i}.json'
        content['diagnostic_configuration']=dict(VLLM_BATCH_INVARIANT=os.environ['VLLM_BATCH_INVARIANT'],VLLM_USE_V2_MODEL_RUNNER=os.environ['VLLM_USE_V2_MODEL_RUNNER'],enforce_eager=True,async_scheduling=False,enable_prefix_caching=False)
        original(dest/Path(path).name,content)
    M.E.save_json=save
    M.lora_smoke(i)
