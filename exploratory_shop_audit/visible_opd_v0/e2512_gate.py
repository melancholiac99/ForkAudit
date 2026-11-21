"""The control-arm decision only accepts sampled T_opd_eval receipts from the new layout."""
from .e2512_routing import ROUTING,owners
from .d_data import PARENT
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read


def compare_counts(p_success,b_success,episodes):
    if not 0<=p_success<=episodes or not 0<=b_success<=episodes or episodes==0:
        raise ValueError('Invalid success counts')
    return 'RUN_CONTROLS' if p_success>b_success else 'NOT_RUN'


def decision(layout,p_endpoint):
    if p_endpoint<100:
        return dict(decision='RUN_CONTROLS',why='P stopped under 25.7 before P@100; the P@100 rule has no observation',p_endpoint=p_endpoint)
    assert p_endpoint==100
    expected={(t['goal_index'],i) for t in read(PARENT/'asin_split_manifest.json')['T_opd_eval'] for i in range(4)}
    assert len(expected)==800
    counts={};files={}
    for arm,k in [('B',0),('P',100)]:
        trials={}
        checkpoint='B@0' if arm=='B' else read(layout.folder(k,arm)/'optimizer.json')['safetensors_sha256']
        for rank in range(4):
            path=layout.root/'evaluation'/f'{k:03}'/arm/f'rank{rank}'/'gate_T_opd_eval_sample.json'
            r=read(path);files[f'{arm}_{rank}']=digest_file(path)
            assert (r['split'],r['decode'],r['step'],r['arm'],r['rank'])==('T_opd_eval','sample',k,arm,rank)
            assert r['routing_sha']==digest_file(ROUTING) and r['checkpoint']==checkpoint
            for goal,rep,win in r['trials']:
                assert owners()[goal]==rank and (goal,rep) not in trials and win in (0,1)
                trials[goal,rep]=win
        assert set(trials)==expected
        counts[arm]=sum(trials.values())
    return dict(decision=compare_counts(counts['P'],counts['B'],800),p_endpoint=100,counts=counts,episodes_per_arm=800,
                pool='T_opd_eval',decode='sample',sources=files,routing_sha=digest_file(ROUTING))
