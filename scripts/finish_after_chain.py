"""Monitor the a'_u + fp32 queue; archive and shut down, or queue recovery review."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from finish_phase1c import ROOT, save_archive, shutdown
from watch_phase1 import identity


def validate():
    import numpy as np
    import pandas as pd
    selection = json.loads((ROOT / 'eval/au_points.json').read_text())['points']
    expected = {(t, int(u)): a for t, u, a in selection}
    seen = set()
    for p in (ROOT / 'data/replay_au').glob('*/*.json'):
        r = json.loads(p.read_text())
        key = (r['traj_id'], r['turn'])
        assert key in expected and key not in seen, p
        seen.add(key)
        assert r['alternatives'] == [expected[key]], p
        assert r['factual_outcomes'] == [] and len(r['alt_outcomes']) == 1, p
        assert len(r['alt_outcomes'][0]) == 6, p
        cs = r['continuations']
        assert len(cs) == 6 and {(c['role'], c['rep']) for c in cs} == {(2, i) for i in range(6)}, p
        for c in cs:
            assert c['refeed_div'] == 0 and c['len'] <= 50, p
            assert c['actions'][0] == expected[key], p
            assert len(c['actions']) == len(c['valids']) == c['len'] - r['turn'] + 1, p
            assert c['won'] or c['len'] == 50, p
            assert int(c['won']) == r['alt_outcomes'][0][c['rep']], p
    assert seen == set(expected), f'a_u incomplete: {len(seen)}/{len(expected)}'
    baseline = pd.read_parquet(ROOT / 'data/scores.parquet')
    score = pd.read_parquet(ROOT / 'data/scores_fp32.parquet')
    chosen = pd.read_parquet(ROOT / 'data/scores_chosen_fp32.parquet')
    cols = ['traj_id', 'turn', 'action']
    assert not score.duplicated(cols).any()
    assert set(map(tuple, baseline[cols].values)) == set(map(tuple, score[cols].values))
    values = ['logp_sum', 'logp_mean', 'logp_sum_notask', 'logp_mean_notask']
    assert np.isfinite(score[values].to_numpy()).all()
    assert score.n_tokens.gt(0).all() and score.scorer.eq('hfprefix').all()
    pts = set(map(tuple, baseline[['traj_id', 'turn']].values))
    assert not chosen.duplicated(['traj_id', 'turn', 'variant']).any()
    assert set(map(tuple, chosen[['traj_id', 'turn', 'variant']].values)) == {
        (t, u, v) for t, u in pts for v in ('emptythink', 'realized', 'hindsight')}
    assert np.isfinite(chosen.loc[chosen.available, ['logp_sum', 'logp_mean']].to_numpy()).all()
    prov = json.loads((ROOT / 'data/scores_fp32.provenance.json').read_text())
    assert prov['n_rows'] == len(score) and prov['n_points'] == len(pts)
    return {'au_points': len(seen), 'au_continuations': len(seen) * 6,
            'scored_points': len(pts), 'score_rows': len(score)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--thread', required=True)
    args = ap.parse_args()
    with (ROOT / 'logs/after_chain_finish.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('Armed for a_u replay and fp32 scoring; runner', args.pid, flush=True)
        while identity(args.pid) == args.start:
            time.sleep(30)
        status = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'runner_pid': args.pid}
        try:
            status['audit'] = validate()
            status['result'] = 'both_tasks_complete_and_verified'
        except Exception:
            status['result'] = 'review_required'
            status['error'] = traceback.format_exc()
        p = ROOT / 'logs/after_chain_shutdown_status.json'
        p.write_text(json.dumps(status, indent=2))
        os.sync()
        try:
            status['archive'] = str(save_archive(ROOT, status))
        except Exception:
            status['result'] = 'review_required'
            status['archive_error'] = traceback.format_exc()
        p.write_text(json.dumps(status, indent=2))
        os.sync()
        if status['result'] == 'review_required':
            message = (
                '【两任务关机前异常审核】用户授权：a_u续跑和FP32评分都完成后保存并校验代码、结果、日志，'
                'sync后执行/usr/bin/shutdown；出错先唤醒本会话尝试修复，无法恢复则保存错误日志并关机。'
                '请读logs/after_chain_shutdown_status.json、logs/au_replay.log、logs/p2_score_fp32.log，'
                '检查scripts/run_after_chain.sh。只恢复这两个任务，不新增实验，不重跑库7–10。'
                '不得删除实验数据。若恢复队列，重新挂载scripts/finish_after_chain.py到新runner的PID和start；'
                '若无法恢复，保存错误和现有文件、sync后执行/usr/bin/shutdown。'
            )
            try:
                result = subprocess.run(['codex', 'queue', '--thread', args.thread,
                                         '--message', message], cwd=ROOT, timeout=45)
                if result.returncode == 0:
                    print('Recovery review queued; reviewer must finish or shut down', flush=True)
                    return
                status['queue_error'] = f'exit {result.returncode}'
            except Exception:
                status['queue_error'] = traceback.format_exc()
            status['result'] = 'recovery_unavailable_errors_and_originals_saved'
        shutdown(status, 'after_chain_shutdown_status.json')


if __name__ == '__main__':
    main()
