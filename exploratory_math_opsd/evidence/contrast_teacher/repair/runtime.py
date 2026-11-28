"""Compact RoPE positions, symmetric context collation and the registered window gate."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from data_collator import SelfDistillationDataCollator

def positions(mask):
    return (mask.long().cumsum(-1)-1).clamp_min(0)

class PairCollator(SelfDistillationDataCollator):
    def __call__(self,features):
        self.tokenizer.padding_side='left'
        if not hasattr(self,'pairs'):
            self.pairs=json.loads(Path(os.environ['REPAIR_PAIRS']).read_text())
        pairs=[self.pairs[f['problem']] for f in features]
        def collate(side):
            return super(PairCollator,self).__call__([dict(f,solution=c[side] or f['solution']) for f,c in zip(features,pairs)])
        pos,neg=collate(0),collate(1)
        assert torch.equal(pos['student_prompts'],neg['student_prompts'])
        for batch in (pos,neg):
            assert batch['teacher_prompt_attention_mask'][:,-1].all()
        assert pos['student_prompt_attention_mask'][:,-1].all()
        pos['negative_prompts']=neg['teacher_prompts']
        pos['contrast_known']=torch.ones(len(features),dtype=torch.bool)
        pos['has_negative']=torch.tensor([bool(p and n) for p,n in pairs])
        return pos

class WindowGate(TrainerCallback):
    def __init__(self,trainer):
        self.trainer=trainer

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step%25:
            return
        counts=torch.tensor(self.trainer.repair_counts,device=self.trainer.accelerator.device)
        if dist.is_initialized():dist.all_reduce(counts)
        ended,total=counts.tolist()
        assert total==25*32,(state.global_step,ended,total)
        stop=ended/total<.15
        if state.is_world_process_zero:
            with (Path(args.output_dir)/'end_windows.jsonl').open('a') as f:
                print(json.dumps(dict(step=state.global_step,ended=ended,total=total,fraction=ended/total,stop=stop)),file=f)
        self.trainer.repair_counts=[0,0]
        if stop:
            control.should_training_stop=True
            control.should_save=True
        return control
