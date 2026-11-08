"""Publish only after ALL preregistered primary cells are complete; no early stop."""
import argparse,importlib,os,subprocess,sys,time,traceback
from pathlib import Path
import accelerate_20261108 as A
E,D,H,ROOT=A.E,A.D,A.H,A.ROOT
OUT=H/'primary_complete_20261109'
FIELDS=('base','after','step','full','rescue','one_step','primary','full_gain','interaction')

def inputs(points):
    for p in points:
        for a,c in [('base','00')]+[(a,c) for a in D.PRIMARY_ARMS for c in ('01','10','11')]:
            for u in range(24):yield p,a,c,u,D.run_path(E.DATA,a,c,p['point_id'],u)

def analyze():
    plan,manifest=A.B.preserved();lock=E.locked();points=E.zpoints(lock);raw={}
    replay=importlib.import_module('11_replay')
    for p,a,c,u,path in inputs(points):
        x=E.load(path);E.verify_target_log(x);A.execution_valid(x,path,p,plan,manifest)
        assert len(x['continuations'])==1;v=x['continuations'][0]
        assert v['rep']==u and not v['refeed_div']
        assert v['len']==p['turn']+len(v['actions'])-1 and v['len']<=50
        assert len(v['actions'])==len(v['valids']) and len(v['seeds'])==len(v['actions'])-1
        assert v['seeds']==[replay.seed_of(p['seed'],p['turn'],1,u*100+p['turn']+k)+D.FORMAL_OFFSET for k in range(len(v['seeds']))]
        assert x['factual_outcomes']==[int(v['won'])]
        expected=lock['adapters'][f'{a}_s{u//8}'] if c in ('01','11') else None
        assert x['provenance']['adapter_sha256']==expected
        raw[str(path.relative_to(ROOT))]=E.sha_file(path)
    assert len(raw)==len(points)*7*24
    rows=[]
    for p in points:
        b=E.result_prob('base','00',p,lock)
        row=dict(point_id=p['point_id'],game=p['game'],remaining=p['remaining'],base=b)
        for a in D.PRIMARY_ARMS:row[a]={'00':b,**{c:E.result_prob(a,c,p,lock) for c in ('01','10','11')}}
        rows.append(row)
    games=D.game_contrasts(rows);stats={k:D.bootstrap_game(games,k) for k in FIELDS}
    verdict=D.reading(stats['rescue']['ci95_pp'],stats['primary']['ci95_pp'],len(rows),len(games))
    out=dict(status='PRIMARY_CELLS_COMPLETE_SECONDARY_PENDING',reading=verdict,n_states=len(rows),n_games=len(games),
        records=len(raw),pooled=stats,lock_sha256=E.sha_file(E.LOCK),
        source={str(H/n):E.sha_file(H/n) for n in ['e4d.py','e4d_common.py','primary_complete_20261109.py']},
        scope='All fixed Azero/Dzero 00/01/10/11 cells. No partial sampling, optional stopping, changed threshold, or cancelled secondary work.',
        limitation='Agap/Dgap and 72-point five-action probe are still pending; this is not the complete E4d report.')
    OUT.mkdir(exist_ok=True)
    E.save_json(OUT/'raw_manifest.json',raw);E.save_json(OUT/'analysis.json',out)
    E.save_json(OUT/'verification.json',dict(passed=True,records=len(raw),raw_manifest_sha256=E.sha_file(OUT/'raw_manifest.json'),analysis_sha256=E.sha_file(OUT/'analysis.json'),original_statistics_reused=True))
    lines=['E4d: complete primary cells; secondary comparisons/probes pending',f'{len(rows)} locked states / {len(games)} games / {len(raw)} records',f'Primary reading: {verdict}','']
    for k in FIELDS:
        v=stats[k];lines.append(f"{k}: {v['mean_pp']:+.3f} pp; 95% game-bootstrap CI {v['ci95_pp']}")
    lines.extend(['',out['scope'],out['limitation']]);(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    return out

def publish():
    def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    assert git('branch','--show-current')=='credit-grok'
    assert git('remote','get-url','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    assert not git('diff','--cached','--name-only'),'Other staged changes; postpone publication'
    files=[OUT/n for n in ['analysis.json','verification.json','raw_manifest.json','REPORT.txt']]
    git('add',*[str(p.relative_to(ROOT)) for p in files])
    if git('diff','--cached','--name-only'):git('commit','-m','Report complete E4d primary cells while secondary work continues')
    for attempt in range(3):
        try:
            git('push','origin','HEAD:credit-grok')
            git('fetch','--no-tags','origin','refs/heads/credit-grok:refs/remotes/origin/credit-grok')
            for p in files:
                name=str(p.relative_to(ROOT));assert git('rev-parse','origin/credit-grok:'+name)==git('hash-object',name)
            E.save_json(OUT/'publication.json',dict(verified=True,commit=git('rev-parse','origin/credit-grok')));return
        except Exception:
            with (OUT/'publication_errors.log').open('a') as f:f.write(traceback.format_exc())
            if attempt==2:raise
            time.sleep(20)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');ap.add_argument('--publish',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    try:
        pts=E.zpoints(E.load(E.LOCK))
        while True:
            missing=sum(not p.exists() for *_,p in inputs(pts))
            E.save_json(OUT/'status.json',dict(stage='waiting' if missing else 'analyzing',missing=missing,pid=os.getpid(),time=time.time()))
            if not missing:break
            assert args.wait,'Primary cells are incomplete'
            time.sleep(30)
        out=analyze()
        if args.publish:publish()
        E.save_json(OUT/'status.json',dict(stage='completed',time=time.time()))
        A.N._wake('【E4d主实验完整结果已出】请读 '+str(OUT/'analysis.json')+' 与 verification.json，汇报完整Azero/Dzero主条件；次要对照和探针仍运行，不得提前关机。主报告不替代完整E4d结果。')
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());os.sync()
        A.N._wake('【E4d独立主报告异常】请读 '+str(OUT/'failure.log')+'。只修只读报告；不要停止GPU队列，不放宽门槛。')
        raise
