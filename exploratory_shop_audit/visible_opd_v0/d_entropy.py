"""Observe native full-vocabulary entropy inside the vLLM decoder before temperature."""
import os
import numpy as np
import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

class NativeEntropy(LogitsProcessor):
    def __init__(self,vllm_config,device,is_pin_memory):
        self.rows={}
        self.tape=np.memmap(os.environ['OPD_ENTROPY_TAPE'],dtype='float32',mode='r+',shape=(64,512))

    def is_argmax_invariant(self):
        # In this version the pre-temperature callback is in this list for both greedy and sampled jobs.
        return False

    def update_state(self,update):
        def attach(params,prompt,output):
            slot=(params.extra_args or {}).get('entropy_slot')
            if slot is not None:return slot,output
        process_dict_updates(self.rows,update,attach)

    def apply(self,logits):
        if self.rows:
            lp=logits.log_softmax(-1)
            values=(-(lp.exp()*lp.nan_to_num()).sum(-1)).cpu().tolist()
            for idx,(slot,output) in self.rows.items():
                self.tape[slot,len(output)]=values[idx]
        return logits
