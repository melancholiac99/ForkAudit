import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
import math500_long
from launch import evaluate, read, sha

arms=('R-noclip','R-cond')
root=math500_long.ROOT
if sys.argv[1]=='math500':
    registered=math500_long.prepare
    def prepare():
        rows,hashes=registered()
        for arm in arms:
            path=root/'run'/arm
            digest=sha(path/'checkpoint-100/adapter_model.safetensors')
            assert digest==read(path/'complete.json')['adapters']['100']
            hashes[arm+'@100']=digest
        return rows,hashes
    math500_long.prepare=prepare
    math500_long.CELLS=[(False,a+'@100') for a in arms]
    math500_long.REPORT=root/'run/cond-kl/math500-receipts.json'
    math500_long.main()
else:
    for arm in arms:
        evaluate(root,os.environ.copy(),arm+'@100',sys.argv[1],1001)
