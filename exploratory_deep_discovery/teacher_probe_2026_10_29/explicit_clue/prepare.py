from pathlib import Path
import json,hashlib
from transformers import AutoTokenizer

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=BASE.parents[1]
MODEL='/root/autodl-tmp/models/Qwen2.5-7B-Instruct'

def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    points=json.loads((ROOT/'exploratory_deep_discovery/regime_mechanism_2026_10_28/design.json').read_text())['points']
    points={p['id']:p for p in points}
    original=[json.loads(x) for x in (BASE/'requests.jsonl').read_text().splitlines()]
    jobs=[]
    for r in original:
        if r['cohort']!='development' or r['condition']!='H':continue
        p=points[r['point_id']]
        marker='<|im_end|>\n<|im_start|>assistant\n'
        assert r['prefix_text'].count(marker)==1
        evidence=('\n<verified_location_evidence>\n'+p['clue']['text']+
          '\nThis true fact is provided to the teacher for assessment. It is not an extra hint '
          'given to the original student after its action.\n</verified_location_evidence>')
        prefix=r['prefix_text'].replace(marker,evidence+marker)
        preids=tok.encode(prefix,add_special_tokens=False)
        candidates=[];fulls=[]
        for c in r['candidates']:
            fulltext=tok.decode(r['prefix_ids']+c['suffix_ids'])
            assert fulltext.startswith(r['prefix_text'])
            suffix=fulltext[len(r['prefix_text']):]
            fulls.append(tok.encode(prefix+suffix,add_special_tokens=False))
        start=len(preids)
        for ids in fulls:
            i=0
            while i<min(len(preids),len(ids)) and preids[i]==ids[i]:i+=1
            start=min(start,i)
        assert len(preids)-start<=3
        for c,ids in zip(r['candidates'],fulls):
            candidates.append({**c,'suffix_ids':ids[start:]})
        job={**r,'request_id':r['point_id']+'__C','condition':'C','prefix_text':prefix,
             'prefix_ids':fulls[0][:start],'candidates':candidates,'token_boundary_backoff':len(preids)-start}
        del job['request_sha256']
        job['request_sha256']=sha(json.dumps(job,sort_keys=True,ensure_ascii=False).encode())
        jobs.append(job)
    assert len(jobs)==24
    path=OUT/'requests.jsonl'
    assert not path.exists()
    path.write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
    prep=dict(contexts=24,candidate_sequences=120,
        protocol_sha256=sha((OUT/'PROTOCOL.md').read_bytes()),request_file_sha256=sha(path.read_bytes()))
    (OUT/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    print(prep)

if __name__=='__main__':main()
