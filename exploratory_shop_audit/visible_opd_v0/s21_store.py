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
