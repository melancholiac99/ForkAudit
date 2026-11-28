"""One sequential editing agent; every previously unqualified question receives an independent attempt."""
import json
import subprocess
import time
from check import HERE,DATA,read,save,verify,boxes
from expand import pool,done
def main():
    folder=DATA/'editor';folder.mkdir(exist_ok=True)
    initial=read(DATA/'validated-base.json')
    previous=done();verdict=read(DATA/'semantic-base.json')
    base_ok={r['idx'] for r in initial if r['qualified'] and verdict[str(r['idx'])]['pass']}
    jobs=[j for j in pool() if j['idx'] not in previous and j['idx'] not in base_ok]
    accepted=[]
    for start in range(0,len(jobs),16):
        batch=jobs[start:start+16]
        out=folder/f'checked-{start:04}.json'
        if out.exists():
            accepted.extend(read(out));continue
        path=folder/f'source-{start:04}.json'
        save(path,[dict(idx=j['idx'],question=j['problem'],gold=j['Answer'],B_box=boxes(j['negative'])[-1],reference=j['solution'],B=j['negative']) for j in batch])
        answer=folder/f'agent-{start:04}.json'
        task=f"""Open {path.name} and individually solve/check every record. Work as the human mathematical editor for a style-matched wrong-solution experiment. For each reference, identify the concrete erroneous step in B and minimally introduce that exact kind of error in the reference, with consistent downstream propagation to exactly B_box. Keep all lines, Markdown headings, prose and LaTeX notation unless the error requires a change. Remove all occurrences of the original correct boxed answer (including the registered gold); no edit may exceed 25% reference tokens. Reject if B answer is mathematically correct or if you cannot construct a coherent matching error. Do not just change final boxed. Produce one result per idx, with concrete reason and an edits list of [old,new] pairs to apply globally in order to reference, using exact substrings (proper JSON backslash escaping). For rejections supply an empty edits list. Return ONLY valid JSON array of {{"idx":integer,"reason":"...","edits":[["...","..."]]}}. Do not write files or invoke other models or agents. You can read and use Python in the working directory to examine records or test your substitutions. Return all {len(batch)} records."""
        for attempt in range(3):
            log=folder/f'editor-{start:04}.log'
            try:
                with log.open('a') as f:
                    subprocess.run(['/root/.local/bin/codex','exec','--skip-git-repo-check','--ephemeral','-s','read-only','-o',str(answer),task],cwd=folder,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
                raw=answer.read_text().strip()
                if raw.startswith('```'):raw='\n'.join(raw.splitlines()[1:-1])
                decisions={r['idx']:r for r in json.loads(raw)}
                assert set(decisions)=={j['idx'] for j in batch}
                break
            except (ValueError,KeyError,AssertionError,subprocess.SubprocessError):
                if attempt==2:raise
                time.sleep(3)
        manual={}
        for j in batch:
            i=j['idx'];d=decisions[i];s=j['solution']
            if not d['edits']:manual[str(i)]=dict(reason=d['reason']);continue
            for old,new in d['edits']:
                if not old or old not in s:break
                s=s.replace(old,new)
            else:
                manual[str(i)]=dict(text=s,error=d['reason'],evidence=boxes(j['negative'])[-1]);continue
            manual[str(i)]=dict(reason='Exact authored edit failed to apply: '+d['reason'])
        result=verify(batch,manual)
        save(out,result);accepted.extend(result)
        print(start,len(result),sum(r['qualified'] for r in result),flush=True)
    merged={r['idx']:r for r in initial};merged.update({r['idx']:r for r in accepted})
    decisions={str(r['idx']):dict(**{'pass':True},reason=r['error']) for r in accepted if r['qualified']}
    decisions.update({str(i):verdict[str(i)] for i in base_ok})
    assert len(merged)==len(initial) and set(decisions)=={str(i) for i,r in merged.items() if r['qualified']}
    save(DATA/'validated.json',list(merged.values()));save(DATA/'semantic.json',decisions)
    save(HERE/'semantic.json',decisions)
if __name__=='__main__':main()
