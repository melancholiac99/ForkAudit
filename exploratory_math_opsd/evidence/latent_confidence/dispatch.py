"""Workers share file locks and atomically retain completed questions across GPU handoffs."""
import os
import fcntl
import sys
import time
from experiment import *

def main():
    mode=sys.argv[1];out=RAW/mode;out.mkdir(exist_ok=True)
    m=read(RAW/'manifest.json');tok=AutoTokenizer.from_pretrained(BASE)
    needed=set(i for i,_,_ in m['train']) & set(m['pilot_extract'] if mode=='pilot' else m['extract'])
    if mode=='full':
        (out/'extract').mkdir(exist_ok=True)
        for source in (RAW/'pilot/extract').glob('*.npz'):
            try:os.link(source,out/'extract'/source.name)
            except FileExistsError:pass
    if not (out/'fit.json').exists():
        if len(list((out/'extract').glob('*.npz')))<len(needed):
            worker('extract',mode,'B',0,1,opportunistic=True)
        while not (out/'fit.json').exists():
            with (out/'fit.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                fs=sorted((out/'extract').glob('*.npz'))
                if not (out/'fit.json').exists() and len(fs)==len(needed):fit(fs,out)
            time.sleep(1)
    for arm in ARMS:
        reader=None
        for stage,expected in [('fixed',33 if mode=='pilot' else 750),('math',33 if mode=='pilot' else 500)]:
            if len(list((out/(stage+'-'+arm)).glob('*.npz')))==expected:continue
            if reader is None:reader=Reader(arm,tok,read(out/'fit.json')['layer_index_zero_based'],np.load(out/'direction.npy'))
            worker(stage,mode,arm,0,1,reader,True)
        del reader;torch.cuda.empty_cache()
if __name__=='__main__':main()
