"""Export a completed ten-update P checkpoint from the serial Stage E run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(root, step, dest):
    if step <= 0 or step % 10:
        raise ValueError('Expected a positive ten-update checkpoint')
    stamp = f'{step:03}'
    receipt_file = root/'arms/P/updates'/f'{stamp}.json'
    receipt = read(receipt_file)
    folder = root/'pilot'/f'{step:02}'/'P'
    optimizer = read(folder/'optimizer.json')
    assert receipt['arm'] == 'P' and receipt['update'] == step
    assert optimizer['after_sha'] == receipt['after_sha']
    behavior_file = root/'behavior'/f'{stamp}.json'
    probe_file = root/'monitor'/f'{stamp}.json'
    behavior = read(behavior_file)
    assert behavior['endpoints'] == {'B': 0, 'P': step}
    settings = read(root/'settings.json')
    endpoints = []
    for rank in range(4):
        rec = read(root/'evaluation'/stamp/'P'/f'rank{rank}'/'complete.json')
        assert (rec['count'], rec['rank'], rec['checkpoint'], rec['k'], rec['arm']) == (
            500, rank, optimizer['safetensors_sha256'], step, 'P')
        assert rec['routing_sha'] == settings['routing_sha']
        endpoints.append(rec)
    probe = read(probe_file)['P']
    assert probe['stats']['all']['positions'] == 4448 and probe['versions'] == [step-10, step]
    fits = [read(root/'refresh'/stamp/f'fit_{rank}.json') for rank in range(4)]
    assert all(f['actor_sha'] == optimizer['after_sha'] for f in fits)
    assert len({f['history_sha'] for f in fits}) == 1

    steps = []
    sources = [receipt_file, behavior_file, probe_file, folder/'optimizer.json', root/'settings.json']
    for k in range(1, step+1):
        p = root/'pilot'/f'{k:02}'/'P'
        m = read(p/'measurement.json')
        s = read(p/'train_statistics.json')
        assert s['arm'] == m['arm'] == 'P' and s['update'] == m['update'] == k
        assert not steps or steps[-1]['after_sha'] == s['before_sha']
        row = {name:s[name] for name in ('update','before_sha','after_sha','optimizer_steps','legal_rate','records','scored','projection_version')}
        for precision in ('bf16','fp32_reanchored'):
            row.update({f'{precision}_{name}':v for name,v in m[precision].items()})
        steps.append(row)
        sources.extend([p/'train_statistics.json',p/'measurement.json'])
    assert sum(s['optimizer_steps'] for s in steps) == optimizer['cumulative_steps']
    assert steps[-1]['after_sha'] == optimizer['after_sha']
    baseline_legal = read(root/'adoption/P.json')['initial_training_legal_rate']
    early = step >= 20 and baseline_legal-max(steps[-11]['legal_rate'],steps[-1]['legal_rate']) >= .1-1e-12
    assert receipt['stopped'] == early

    dest.mkdir(parents=True, exist_ok=True)
    for name, source in [('behavior.json',behavior_file),('common_probe.json',probe_file)]:
        (dest/name).write_bytes(source.read_bytes())
    with (dest/'training.tsv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(steps[0]), dialect='excel-tab',lineterminator='\n')
        w.writeheader()
        w.writerows(steps)
    history = {}
    for k in range(0,step+1,10):
        path = root/'behavior'/f'{k:03}.json'
        history[k] = read(path)['summary']
        sources.append(path)
    labels = ['B@0'] + [f'P@{k}' for k in range(10,step+1,10)]
    md = [f'# 正式 E · P@{step}', '',
          f'已完成 {step} 次更新，累计 {optimizer["cumulative_steps"]} 个 AdamW 小步。训练采用单种子。', '',
          '## 全要求购买率', '',
          '每个评测集有 200 个任务。采样每题 4 条轨迹，贪心每题 1 条。差值为百分点，95% 区间按 ASIN 聚类配对 bootstrap 计算。', '',
          '| 评测集 | 解码 | '+' | '.join(labels)+f' | P@{step} − B@0（95% 区间，百分点） |',
          '|---|---|'+'---:|'*(len(labels)+1)]
    for pool in ('T_opd_eval','V'):
        for mode,desc,n in [('sample','采样',800),('greedy','贪心',200)]:
            counts = []
            for k,s in history.items():
                v = s[f'{"P" if k else "B"}/{pool}/{mode}']
                assert v['tasks'] == 200 and v['episodes'] == n
                counts.append(round(v['all_requirements']*n))
            diff = behavior['paired'][f'P-B/{mode}/asin/{pool}']['all_requirements']
            assert abs((counts[-1]-counts[0])/n-diff['estimate']) < 1e-10
            lo,hi = diff['ci95']
            values = [f'{v/n:.3%} ({v}/{n})' for v in counts]
            md.append(f'| {pool} | {desc} | '+' | '.join(values)+
                      f' | {diff["estimate"]*100:+.3f} [{lo*100:+.3f}, {hi*100:+.3f}] |')
    md.extend(['','## 更新前后固定目标 KL','',
               'ΔKL 为更新前 KL 减去更新后 KL。正值表示本次更新使学生更接近当步目标。每段均值按十次更新等权计算。','',
               '| 更新 | BF16 ΔKL 为正 | BF16 平均 ΔKL | FP32 重锚 ΔKL 为正 | FP32 平均 ΔKL |','|---|---:|---:|---:|---:|'])
    for start in range(0,step,10):
        group = steps[start:start+10]
        vals = []
        for key in ('bf16_delta_kl','fp32_reanchored_delta_kl'):
            series = [r[key] for r in group]
            vals.extend([f'{sum(v>0 for v in series)}/10',f'{mean(series):+.6f}'])
        md.append(f'| {start+1}–{start+10} | '+' | '.join(vals)+' |')
    decision = probe['stats']['decision_click_option']
    md.extend(['','## 固定探针与训练轨迹','',
               f'共同前缀 4,448 个位置的 BF16 全词表 KL(π₀‖πₜ) 为 {probe["stats"]["all"]["kl"]:.6f}。'
               f'click 和规格位置的平均熵为 {decision["entropy"]:.6f}，平均 top-1 概率为 {decision["top1"]:.4%}。'
               f'投影器版本 {step-10} 与 {step} 的修正余弦为 {probe["projector_cosine"]:.6f}，均方差为 {probe["projector_mse"]:.6f}。','',
               f'训练轨迹合法动作率为 {steps[-1]["legal_rate"]:.2%}，对应的第 0 步参考为 {baseline_legal:.2%}。'
               +('已触发 §25.7 停止条件。' if early else '本次监测通过 §25.7 继续条件。'), '',
               '[行为评测与配对区间](behavior.json) · [固定探针各类别](common_probe.json) · [逐步训练读数](training.tsv) · [检查点与数据校验](provenance.json)'])
    (dest/'README.md').write_text('\n'.join(md)+'\n')
    provenance = dict(run=root.name,section=settings['section'],seed=settings['seed'],update_receipt=receipt,
                      optimizer_steps=optimizer['cumulative_steps'],adapter_sha256=optimizer['safetensors_sha256'],
                      evaluation=endpoints,projector_file_sha256=[f['projector_sha'] for f in fits],
                      history_sha256=fits[0]['history_sha'],training_legal_reference=baseline_legal,
                      sources={str(path.relative_to(root)):digest(path) for path in sorted(set(sources))})
    (dest/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('step',type=int)
    parser.add_argument('dest',type=Path)
    args = parser.parse_args()
    export(args.root,args.step,args.dest)
