"""Read-only cross-check of server outputs against CSVs and published rounded results."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

H=Path(__file__).resolve().parent;ROOT=H.parent;OUT=H/'server_20261112'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    summary=json.loads((OUT/'summary.json').read_text());execution=json.loads((OUT/'execution.json').read_text())
    for p,h in execution['source_hashes'].items():assert sha(ROOT/p)==h,p
    assert (summary['n_points'],summary['n_games'],summary['n_arms'],summary['n_replays_per_arm'])==(6016,308,5,3)
    primary=list(csv.DictReader((OUT/'points.csv').open()));holdout=list(csv.DictReader((OUT/'replicate_holdout.csv').open()))
    assert len(primary)==1586 and summary['zero_gap_count']==4430
    assert len({r['point_id'] for r in primary})==len(primary)
    assert len({(r['point_id'],r['heldout_rep']) for r in holdout})==len(holdout)==4147
    for rows,bins,expected_counts in [(primary,summary['gap_bins'],[894,462,230]),(holdout,summary['noise_holdout_bins'],[2818,1329])]:
        levels=sorted({round(float(r['gap']),10) for r in rows});assert len(levels)==len(bins)
        assert [b['n'] for b in bins]==expected_counts
        for level,b in zip(levels,bins):
            selected=[r for r in rows if round(float(r['gap']),10)==level];assert len(selected)==b['n']
            for key in ['gap','projected_diff','signed_ratio','clipped_ratio']+(['heldout_diff'] if rows is holdout else []):
                vals=np.array([float(r[key]) for r in selected]);assert np.isfinite(vals).all()
                assert np.isclose(vals.mean(),b[key+'_mean'],rtol=0,atol=1e-12),(level,key)
        for r in rows:
            g,d,ratio=float(r['gap']),float(r['projected_diff']),float(r['clipped_ratio'])
            assert 0<g<=1 and -1<=d<=1 and 0<=ratio<=1
            assert abs(ratio-np.clip(d/g,0,1))<1e-12
            assert abs(float(r['signed_ratio'])-d/g)<1e-12
            assert int(r['correct_sign'])==int(d>0)
            if rows is holdout:assert int(r['heldout_rep']) in (0,1,2) and float(r['heldout_diff']) in (-1,0,1)
    for key,metric in [('rank_association','clipped_ratio'),('projected_difference_association','projected_diff')]:
        r=spearmanr([round(float(x['gap']),10) for x in primary],[float(x[metric]) for x in primary]).statistic
        assert abs(r-summary[key]['rho'])<1e-12
    assert len(summary['folds'])==5 and sum(f['heldout_points'] for f in summary['folds'])==6016
    assert sum(f['heldout_games'] for f in summary['folds'])==308
    assert all(f['train_games']+f['heldout_games']==308 and f['train_points']+f['heldout_points']==6016 for f in summary['folds'])
    checks={
        'MSE':(round(summary['oof']['mse'],5),.03961),
        'mean_baseline_MSE':(round(summary['oof']['baseline_mse'],5),.04504),
        'action_baseline_MSE':(round(summary['oof']['action_only_mse'],5),.04312),
        'ratio_rho':(round(summary['rank_association']['rho'],3),.091),
        'projected_difference_rho':(round(summary['projected_difference_association']['rho'],3),.201),
        'gap_bin_predictions':([round(b['projected_diff_mean'],4) for b in summary['gap_bins']],[.0050,.0141,.0337]),
        'gap_bin_clipped_ratios':([round(b['clipped_ratio_mean'],4) for b in summary['gap_bins']],[.0431,.0352,.0390]),
        'independent_holdout_gaps':([round(b['heldout_diff_mean'],4) for b in summary['noise_holdout_bins']],[.0529,.2792])}
    matches={k:dict(server=a,published=b,match=a==b) for k,(a,b) in checks.items()}
    result=dict(passed=True,checks=['source files unchanged','fixed cohort/counts','CSV unique IDs and valid ranges','summary bins independently recomputed','rank coefficients independently recomputed','five-fold coverage','comparison with published rounded results'],published_comparison=matches,all_published_rounded_values_match=all(v['match'] for v in matches.values()),artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='verification.json'},verifier_sha256=sha(Path(__file__)))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
