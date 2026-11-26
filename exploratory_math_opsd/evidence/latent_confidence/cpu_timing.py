"""Time the registered cluster statistics on completed B pilot rows."""
from time import monotonic
import numpy as np
from summarize import metrics
from experiment import RAW, SEED, read, save
m=read(RAW/'manifest.json');out={}
for kind,key,scale in [('fixed','pilot_assess',750/33),('math','pilot_math',500/33)]:
    ids=m[key];n=len(ids)
    rng=np.random.default_rng(SEED)
    def bench(arm,draws):
        weights=rng.multinomial(n,np.full(n,1/n),draws).astype(float)
        start=monotonic()
        metrics(RAW/'pilot',arm,ids,weights,kind)
        return monotonic()-start
    bench('B',5)
    b=bench('B',1000)
    other=bench('T0@100',1000) if kind=='fixed' else b
    out[kind]=dict(draws=1000,seconds_B=b,seconds_trained=other,estimated_seconds=(b+5*other)*10*scale)
save(RAW/'cpu_timing.json',out)
