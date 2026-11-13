"""Temporary sampling export; saved LoRA and optimizer remain authoritative."""
import gc,shutil,tempfile,time
from pathlib import Path
from util import DATA,save,sha

def export_policy(base,adapter):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 if shutil.disk_usage(DATA).free<38*1024**3:raise RuntimeError('Need 38 GiB free for temporary FP32 policy export')
 adapter=Path(adapter);weight_sha=sha(adapter/'adapter_model.safetensors')
 dest=Path(tempfile.mkdtemp(prefix='sampling_export_',dir=DATA))
 # CPU merge before loading vLLM. The export is never a training checkpoint.
 model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')
 model=PeftModel.from_pretrained(model,str(adapter)).eval()
 merged=model.merge_and_unload(safe_merge=True)
 merged.save_pretrained(dest,max_shard_size='4GB');AutoTokenizer.from_pretrained(str(base),local_files_only=True).save_pretrained(dest)
 del merged,model;gc.collect()
 save(DATA/'exports'/f'{dest.name}.json',{'adapter':str(adapter),'adapter_sha256':weight_sha,'dtype':'float32','created':time.time(),'temporary_path':str(dest),'weights':{p.name:sha(p) for p in dest.glob('*.safetensors')},'cleanup':'collector deletes only this export after successful engine shutdown; failed exports retained for inspection'})
 return dest
