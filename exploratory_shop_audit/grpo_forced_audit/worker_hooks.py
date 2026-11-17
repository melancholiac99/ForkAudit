"""Use the proven TP1/checkpoint guards and explicit per-decision RNG streams."""
import functools
from copy import deepcopy
import os


def install():
    from .parallel import install_worker
    install_worker()
    import checkpoint_evidence  # installs pinned BF16/FP32 and native restore hashes
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    import torch
    import json
    from pathlib import Path
    update = DataParallelPPOActor.update_policy
    if not getattr(update, '_shop_weights', False):
        @functools.wraps(update)
        def audited_update(self, data):
            weights = lambda: {n:p for n,p in self.actor_module.named_parameters() if p.requires_grad}
            before = checkpoint_evidence.digest(weights())
            result = update(self, data)
            after = checkpoint_evidence.digest(weights())
            assert before != after, 'No actual trainable weight change'
            assert all(torch.isfinite(p).all() for p in weights().values())
            import torch.distributed as dist
            records=[None]*dist.get_world_size()
            record={'rank':dist.get_rank(),'before':before,'after':after,'changed':True,'finite':True}
            dist.all_gather_object(records,record)
            assert len({(r['before'],r['after']) for r in records})==1, 'DP weights diverged'
            if dist.get_rank()==0:
                with (Path(os.environ['SHOP_GRPO_RUN'])/'weight_updates.jsonl').open('a') as f:
                    f.write(json.dumps({**record,'world_size':dist.get_world_size(),'all_ranks_equal':True,'ranks':records})+'\n')
            return result
        audited_update._shop_weights = True
        DataParallelPPOActor.update_policy = audited_update
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    old=vLLMRollout.generate_sequences
    if getattr(old,'_shop_seeds',False):return
    @functools.wraps(old)
    def generate(self,prompts,*args,**kwargs):
        seeds=prompts.non_tensor_batch.pop('shop_seed',None)
        assert seeds is not None and len(seeds)==len(prompts)
        original=self.inference_engine.generate
        def seeded(*a,**kw):
            base=kw['sampling_params'];params=[]
            for seed in seeds:
                p=deepcopy(base);p.seed=int(seed);params.append(p)
            kw['sampling_params']=params
            result=original(*a,**kw)
            for inp,out in zip(kw['prompts'],result):
                assert list(inp['prompt_token_ids'])==list(out.prompt_token_ids), 'Actual inference IDs changed'
            return result
        self.inference_engine.generate=seeded
        try:return old(self,prompts,*args,**kwargs)
        finally:self.inference_engine.generate=original
    generate._shop_seeds=True
    vLLMRollout.generate_sequences=generate

install()
