"""Verify complete coverage, explicit reuse provenance and unchanged inputs."""
import argparse
from collections import Counter
from pathlib import Path
from protocol import ARMS,CONDITIONS,ROOT,read,write,file_hash,seed_for,verify_inputs
from run import result_key,validate_record


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run_dir',type=Path)
    args=ap.parse_args();m=read(args.run_dir/'run_manifest.json');verify_inputs(m['design'])
    points=[p for p in m['design']['points'] if p['id'] in m['config']['point_ids']]
    hashes={};counts=Counter();formats=Counter();reused=0;total_steps=0;total_tokens=0
    for p in points:
        for a in ARMS:
            for c in CONDITIONS:
                for rep in range(m['config']['reps']):
                    key=result_key(p,a,c,rep);path=args.run_dir/'results'/f'{key}.json'
                    r=read(path);validate_record(r,m['run_fingerprint'],key)
                    assert not r['dry'] and r['forced_action']==p['arms'][a]['action']
                    assert r['final_episode_step']<=50 and r['generated_steps']==len(r['transitions'])-1
                    assert (p['clue']['text'] in r['first_prompt']) == (c=='completed')
                    assert r['transitions'][0]['action']==p['arms'][a]['action']
                    assert r['transitions'][0]['observation_after']==p['arms'][a]['observation_after']
                    if 'source_record' in r:
                        src=r['source_record'];original=read(ROOT/src['path'])
                        assert file_hash(ROOT/src['path'])==src['sha256']
                        assert original['transitions']==r['transitions'] and original['first_prompt']==r['first_prompt']
                        assert original['won']==r['won'];reused+=1
                    else:
                        for t in r['transitions'][1:]:
                            assert t['seed']==seed_for(p['index'],a,rep,t['step']-1)
                    counts[(p['id'],a,c)]+=1
                    total_steps+=r['generated_steps'];total_tokens+=r['generated_tokens']
                    for t in r['transitions'][1:]:
                        formats[t['fmt']]+=1
                    hashes[str(path.relative_to(args.run_dir))]=file_hash(path)
    expected=len(points)*10*m['config']['reps']
    assert len(hashes)==expected and set(counts.values())=={m['config']['reps']}
    result=dict(status='complete',records=expected,cells=len(counts),repeats_per_cell=m['config']['reps'],
        reused=reused,new=expected-reused,source_records_identical=True,original_inputs_unchanged=True,
        all_condition_placements_match=True,generated_steps_including_reuse=total_steps,
        generated_tokens_including_reuse=total_tokens,formats=dict(formats),
        run_fingerprint=m['run_fingerprint'])
    write(args.run_dir/'integrity.json',result)
    write(args.run_dir/'results_manifest.json',hashes)
    print(result)


if __name__=='__main__':
    main()
