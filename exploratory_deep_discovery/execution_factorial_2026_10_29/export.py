"""Publishable full outcomes, compressed traces, and deterministic raw examples."""
from pathlib import Path
import shutil
import tarfile
from protocol import HERE, ROOT, read, write, file_hash

def main():
    results=HERE/'results'; d=read(HERE/'design.json')
    integrity=read(results/'integrity.json')
    assert integrity['all_checks_passed'] and integrity['records']==960
    run=HERE/'runs/main_6x8'
    for name in ['run_manifest.json','runtime.json','progress.json']:
        shutil.copy2(run/name,results/name)
    # Preserve all traces, including the actual reused control records, with
    # original project-relative names so the hash manifest can be checked.
    archive=results/'all_960_continuations.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for item in integrity['sources']:
            path=ROOT/item['path']
            assert file_hash(path)==item['sha256']
            tar.add(path,arcname=item['path'])
    cases=[];case_dir=results/'cases';case_dir.mkdir(exist_ok=True)
    controls={(r['point'],r['arm'],r['condition'],r['rep']):ROOT/r['path'] for r in d['controls']}
    for p in d['points']:
        # Fixed from mechanism metadata: information-revealing source arm for
        # common obstacles; original wrong-object-taking arm for branch obstacles.
        if p['old_regime']=='Dead':
            arm=next(a for a,v in p['arms'].items() if v['action']==p['source_action'])
        else:
            arm=next(a for a,v in p['arms'].items() if v['repair_object'] is not None)
        row=dict(point=p['id'],arm=arm,selection='metadata-fixed diagnostic arm, repeat 0',conditions={})
        for c in ['native','completed','repair','repair_completed']:
            if c in ['native','completed']:
                path=controls[(p['id'],arm,c,0)]
            else:
                path=run/'results'/f"{p['id']}__{arm}__{c}__r000.json"
            r=read(path);name=f"{p['id']}__{arm}__{c}__r000.json"
            shutil.copy2(path,case_dir/name)
            row['conditions'][c]=dict(won=r['won'],path='cases/'+name,
                steps=[dict(step=t['step'],action=t['action'],repair=t.get('repair',False),
                            observation=t['observation_after']) for t in r['transitions']])
        cases.append(row)
    write(results/'case_index.json',cases)
    files=[p for p in HERE.rglob('*') if p.is_file() and 'runs' not in p.parts
           and '__pycache__' not in p.parts and not p.name.startswith('shutdown')
           and p.name not in ['bundle_manifest.json','launch.log','launch_pid.json']]
    write(results/'bundle_manifest.json',dict(archive_records=960,
        source_records=integrity['sources'],
        files={str(p.relative_to(HERE)):file_hash(p) for p in sorted(files)},
        note='All 480 new and 480 explicitly reused trajectories are in the compressed archive. '
             'Dry-run trajectories are omitted; test results and logs remain.'))
    print('Exported',len(files),'files; raw archive bytes',archive.stat().st_size)

if __name__=='__main__':
    main()
