"""Lossless byte-plane encoding of FP32 LoRA parameters relative to initialization."""
from pathlib import Path
import json
import zipfile

import numpy as np
import torch
from .c_targets import RUN as ORIGINAL

RUN=ORIGINAL.parent/'shop_visible_opd_s21_20261120'
ARMS=('R','R-beta','P','P-shuffled')
GRID=(0.,.125,.25,.5,1.,2.,4.)

def pack(path, parameters, initial):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    shapes={}
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,p in parameters:
            a=p.detach().cpu().contiguous().numpy()
            b=initial[name].contiguous().numpy()
            assert a.dtype==b.dtype==np.float32
            bits=a.view(np.uint32)^b.view(np.uint32)
            encoded=bits.reshape(-1).view(np.uint8).reshape(-1,4).T.copy()
            z.writestr(name,encoded.tobytes())
            shapes[name]=list(a.shape)
        z.writestr('shapes.json',json.dumps(shapes))
    tmp.replace(path)

def unpack_member(archive,name,initial):
    b=initial[name].numpy()
    x=np.frombuffer(archive.read(name),np.uint8).reshape(4,-1).T.copy()
    return (x.view(np.uint32).reshape(b.shape)^b.view(np.uint32)).view(np.float32)


def export_adapter(archive_path, destination):
    """Materialize a PEFT adapter without access to the private task traces."""
    from peft import LoraConfig
    from safetensors.torch import save_file
    from exploratory_shop_audit.grpo_forced_audit.common import MODEL
    initial=torch.load(RUN/'original/initial.pt',weights_only=True,map_location='cpu')
    tensors={}
    with zipfile.ZipFile(archive_path) as archive:
        for name in initial:
            tensors[name.replace('.default.weight','.weight')]=torch.from_numpy(unpack_member(archive,name,initial))
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    config=LoraConfig(r=64,lora_alpha=64,target_modules='all-linear',lora_dropout=0.,
                      bias='none',task_type='CAUSAL_LM',base_model_name_or_path=str(MODEL),inference_mode=True)
    config.save_pretrained(destination)
    save_file(tensors,destination/'adapter_model.safetensors',metadata={'format':'pt'})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Export a §21 archive into a new PEFT adapter directory.')
    parser.add_argument('archive',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    export_adapter(args.archive,args.destination)
