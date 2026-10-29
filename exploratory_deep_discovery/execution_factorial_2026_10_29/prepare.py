"""Freeze six predetermined cases and check repair legality using CPU only."""
import copy
import re
from protocol import (HERE, ROOT, VERSION, ARMS, read, write, digest, file_hash,
                      verify_inputs, verify_environment, post_facts, repair_action)

SOURCE = ROOT / 'exploratory_deep_discovery/regime_mechanism_2026_10_28'
IDS = ['s005_t31', 's006_t24', 's021_t15', 's078_t36', 's010_t29', 's073_t10']

def main():
    old = read(SOURCE / 'design.json')
    verify_inputs(old)
    points = [copy.deepcopy(next(p for p in old['points'] if p['id'] == pid)) for pid in IDS]
    checks = verify_environment(points)
    import alfenv
    env = None
    inputs = dict(old['inputs'])
    inputs[str((SOURCE/'design.json').relative_to(ROOT))] = file_hash(SOURCE/'design.json')
    controls = []
    try:
        for p in points:
            pk = read(ROOT / p['peek_path'])
            for arm in ARMS:
                a = p['arms'][arm]
                peek = next(x for x in pk['peeks'] if x['action'] == a['action'])
                facts = post_facts(pk, peek)
                held = [m.group(1) for f in facts
                        if (m := re.match(r'^holds\([^,]+,\s*([a-z]+ \d+)\s*:\s*object\)$',f))]
                wrong = [o for o in held if o.rsplit(' ',1)[0] != p['target']]
                assert len(wrong) <= 1
                a['repair_object'] = wrong[0] if wrong else None
                action = repair_action(a['admissible_after'], a['repair_object'])
                a['immediate_repair'] = action
                row = dict(point=p['id'], arm=arm, held=held, repair_object=a['repair_object'],
                           immediate_repair=action)
                if action:
                    game = str(alfenv.ALFWORLD_DATA + '/' + p['game_file'])
                    if env is None:
                        env = alfenv.AlfGame(game)
                    else:
                        env.load(game)
                    _, _, _, div = env.refeed(p['prefix'], p['expected_prefix_observations'])
                    assert not div
                    env.step(a['action'])
                    obs, adm, won, done = env.step(action)
                    assert 'You move ' in obs and not won and not done, obs
                    assert not any(x.startswith(f"move {a['repair_object']} to ") for x in adm)
                    row.update(repair_response=obs, admissible_after_repair=sorted(adm))
                checks.append(row)
                for condition in ['native','completed']:
                    for rep in range(8):
                        path = SOURCE/'runs/main_24x16/results'/f"{p['id']}__{arm}__{condition}__r{rep:03d}.json"
                        r = read(path)
                        assert r['status'] == 'complete' and not r['dry']
                        assert r['forced_action'] == a['action'] and r['clue'] == p['clue']['text']
                        controls.append(dict(path=str(path.relative_to(ROOT)), sha256=file_hash(path),
                                             point=p['id'], arm=arm, condition=condition, rep=rep))
    finally:
        if env is not None:
            env.close()
    design = dict(protocol=VERSION, status='exploratory fixed before outcomes', points=points,
        reps=8, inputs=inputs, controls=controls, new_continuations=480,
        protocol_document_sha256=file_hash(HERE/'PROTOCOL.md'),
        repair='First legal release of originally held non-target-category object; at most once; costs one step',
        cue_timing='Once immediately after original action, before repair; normal history expiry')
    design['design_sha256'] = digest(design)
    write(HERE/'design.json', design)
    write(HERE/'preflight.json', dict(checks=checks, original_branches=30,
        flagged_branches=sum(a['repair_object'] is not None for p in points for a in p['arms'].values()),
        immediate_release_branches=sum(a['immediate_repair'] is not None for p in points for a in p['arms'].values()),
        controls=len(controls)))
    print(read(HERE/'preflight.json') | {'checks': 'saved'}, flush=True)

if __name__ == '__main__':
    main()
