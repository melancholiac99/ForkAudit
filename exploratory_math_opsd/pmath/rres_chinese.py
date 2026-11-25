import hashlib,json,regex
from pathlib import Path
from common import ROOT,json_save
OUT=Path(__file__).resolve().parents[1]/'evidence/rres'
HAN=regex.compile(r'\p{Script=Han}')
LATIN=regex.compile(r'\p{Script=Latin}')
def describe(rows):
    return dict(responses=len(rows),with_han=sum(n>0 for n,_,_ in rows),
        with_han_fraction=sum(n>0 for n,_,_ in rows)/len(rows),
        mean_han_character_fraction=sum(n/max(1,k) for n,k,_ in rows)/len(rows),
        with_han_and_latin=sum(n>0 and latin for n,_,latin in rows),
        with_han_and_latin_fraction=sum(n>0 and latin for n,_,latin in rows)/len(rows))
def main():
    result={}
    for m in ('B','R@100','P@100'):
        all_rows=[]
        result[m]={}
        for bench in ('aime24','aime25','hmmt25'):
            p=ROOT/'run/evaluation'/m/bench/'1001.json'
            assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads(p.with_suffix('.receipt.json').read_text())['sha256']
            rows=[]
            for r in json.loads(p.read_text())['results']:
                for g in r['generations']:
                    text=g['full_generation']
                    rows.append((len(HAN.findall(text)),len(text),bool(LATIN.search(text))))
            assert len(rows)==360
            all_rows.extend(rows)
            result[m][bench]=describe(rows)
        result[m]['all']=describe(all_rows)
    OUT.mkdir(parents=True,exist_ok=True)
    json_save(OUT/'han.json',result)
if __name__=='__main__':main()
