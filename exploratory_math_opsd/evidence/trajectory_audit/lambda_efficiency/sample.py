"""Fixed seed outcome-stratified selection plus two matched question panels."""
import hashlib,json,random
from collections import defaultdict
from measure import HERE, org, ARMS,cap
def main():
    rng=random.Random(20261124);tok=org.Tokenizer.from_file(str(cap.TOKENIZER))
    blind=[];key={}
    for ds in ('aime24','nonthinking'):
        panel=rng.randrange(30 if ds=='aime24' else 500)
        for model in ARMS:
            obj,_=org.source(ds,model);groups=defaultdict(list)
            for i,q in enumerate(obj['results']):
                for j,g in enumerate(q['generations']):
                    groups[g['finish_reason'], 'right' if g['correct'] else 'missing' if cap.archived_answer(g) is None else 'wrong'].append((i,j))
            picks={rng.choice(items):'stratum' for items in groups.values()}
            picks.setdefault((panel,0),'matched')
            for (i,j),design in picks.items():
                q=obj['results'][i];g=q['generations'][j];t=g['full_generation']
                code=hashlib.sha256(f'eff20261124{ds}/{model}/{i}/{j}'.encode()).hexdigest()[:8]
                enc=tok.encode(t,add_special_tokens=False);offsets=enc.offsets;n=len(offsets)
                spans=[]
                for fraction in (.2,.6,.94):
                    a=max(0,min(n-80,int(n*fraction)-40));b=min(n,a+80)
                    spans.append(dict(start=a,end=b,text=t[offsets[a][0]:offsets[b-1][1]]))
                key[code]=dict(ds=ds,model=model,q=i,j=j,design=design,panel=i==panel and j==0,finish=g['finish_reason'],correct=g['correct'])
                blind.append(dict(id=code,problem=q['problem'],gold=q['ground_truth'],spans=spans))
    blind.sort(key=lambda v:v['id'])
    (HERE/'blind.json').write_text(json.dumps(blind,indent=1,ensure_ascii=False))
    (HERE/'unblind.json').write_text(json.dumps(key,indent=1))
if __name__=='__main__':main()
