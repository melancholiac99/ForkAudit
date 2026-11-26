import sys
from train import ROOT,HERE,ARMS,sha,read
sys.path.insert(0,str(HERE.parents[2]))
def main():
    import math500_trajectory as short
    import math500_long as long
    rows,hashes=short.prepare()
    names=[a+'@100' for a in ('R','R-noclip',*ARMS)]
    for a in ('R-noclip',*ARMS):
        digest=sha(ROOT/f'run/{a}/checkpoint-100/adapter_model.safetensors')
        assert digest==read(ROOT/f'run/{a}/complete.json')['adapters']['100']
        hashes[a+'@100']=digest
    for name in names:short.cell(name,rows,hashes)
    long.prepare=lambda:(rows,hashes)
    long.CELLS=[(False,name) for name in names]
    long.REPORT=HERE/'long_receipts.json'
    long.main()
if __name__=='__main__':main()
