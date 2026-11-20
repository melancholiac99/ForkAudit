"""Publish aggregate evidence after the frozen D behavior evaluation."""
import shutil
import sys

import torch

from .post_d_common import ARMS, CASES, POST, case_dir, folder, required_stage
from .post_d_gradients import geometry
from .s22_report import tsv
from .s23_cross import EVIDENCE
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, read_rows, save


def snapshot_report():
    required_stage()
    rows = []
    for k in range(1, 11):
        for arm in ARMS:
            row = read(POST / 'snapshot' / f'{k:02}_{arm}.json')
            assert (row['update'], row['arm']) == (k, arm)
            assert row['valid_candidate_logits'] >= row['positions'] >= row['records'] > 0
            rows.append(row)
    tsv(EVIDENCE / 'snapshot_audit.tsv', rows)
    summary = dict(cells=len(rows), passed=all(r['passed'] for r in rows),
                   valid_logits=sum(r['valid_candidate_logits'] for r in rows),
                   max_difference=max(r['max_abs_logit_diff'] for r in rows),
                   max_probability_difference=max(r['probability_max_abs_diff'] for r in rows),
                   unmatched=[(r['update'], r['arm']) for r in rows if not r['passed']])
    save(POST / 'snapshot_summary.json', summary)
    if not summary['passed']:
        raise RuntimeError(f"Inspect source inputs and numeric paths for {summary['unmatched']}")
    return rows, summary


def reset_report():
    rows = []
    for k, arm in CASES:
        path = case_dir(k, arm)
        fp, meta = read(path / 'fp32.json'), read(path / 'optimizer.json')
        hist = read(folder(k, arm) / 'optimizer.json')
        assert (fp['update'], fp['arm']) == (k, arm)
        assert meta['before_sha'] == hist['before_sha'] == fp['initial_sha']
        assert meta['reset_step_before'] == 0 and meta['reset_step_after'] == hist['optimizer_steps']
        assert meta['minibatch_sha256'] == hist['minibatch_sha256']
        assert meta['historical_step_before'] + meta['reset_step_after'] == hist['cumulative_steps']
        assert meta['target_sha256'] == read(folder(k, arm) / 'target_manifest.json')['target_sha256']
        original, reset = fp['historical'], fp['reset']
        assert original == read(folder(k, arm) / 'fp32.json')['fixed_target']
        assert abs(original['kl_before'] - reset['kl_before']) < 1e-10
        row = dict(update=k, arm=arm, kl_before=reset['kl_before'])
        for label, endpoint in [('original', original), ('reset', reset)]:
            assert abs(endpoint['delta_kl'] - endpoint['alignment'] + endpoint['movement']) < 1e-11
            for field in ('alignment', 'movement', 'delta_kl', 'kl_after'):
                row[f'{label}_{field}'] = endpoint[field]
        row.update(alignment_change=reset['alignment']-original['alignment'],
                   recovered_positive=original['alignment'] <= 0 < reset['alignment'],
                   historical_steps_at_start=meta['historical_step_before'],
                   optimizer_steps=meta['reset_step_after'], initial_sha=fp['initial_sha'],
                   reset_after_sha=fp['reset_sha'], minibatch_sha256=meta['minibatch_sha256'],
                   target_sha256=meta['target_sha256'])
        rows.append(row)
    tsv(EVIDENCE / 'reset_adam_replay.tsv', rows)
    return rows


def gradient_report():
    rows = []
    for arm in ARMS:
        previous = None
        for k in range(1, 11):
            path = POST / 'gradients' / arm / f'{k:02}'
            meta = read(path / 'manifest.json')
            assert meta['weight_sha'] == read(folder(k, arm) / 'optimizer.json')['before_sha']
            assert (meta['update'], meta['arm']) == (k, arm)
            assert digest_file(path / 'gradient.pt') == meta['gradient_file_sha256']
            gradient = torch.load(path / 'gradient.pt', weights_only=True, mmap=True, map_location='cpu')
            assert sum(p.numel() for p in gradient.values()) == meta['parameters']
            if previous is not None:
                values = geometry(previous, gradient)
                assert values['cosine'] is None or abs(values['cosine']) <= 1 + 1e-12
                rows.append(dict(arm=arm, previous_update=k-1, update=k, **values,
                                 previous_gradient_sha=previous_sha, gradient_sha=meta['gradient_file_sha256']))
            previous, previous_sha = gradient, meta['gradient_file_sha256']
    tsv(EVIDENCE / 'adjacent_gradient_cosines.tsv', rows)
    return rows


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|',
            *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]]


def final():
    snapshots, anchor = snapshot_report()
    replays = reset_report()
    gradients = gradient_report()
    counts = {a:sum(r['recovered_positive'] for r in replays if r['arm'] == a) for _, a in CASES[:3]}
    negative = {a:sum(r['original_alignment'] <= 0 for r in replays if r['arm'] == a) for a in counts}
    allocations = read_rows(POST / 'allocation.jsonl')
    report = dict(snapshot=anchor, reset_cells=len(replays), gradient_pairs=len(gradients),
                  nonpositive_original_alignment=negative, recovered_positive_alignment=counts,
                  gpu_hours=sum(a['seconds']*a['gpus'] for a in allocations)/3600, allocated_stages=allocations,
                  stage_E_started=False)
    save(EVIDENCE / 'post_d_checks.json', report)
    lookup = {(r['arm'],r['update']): r['fp32_kl_before'] for r in snapshots}
    cosine = {(r['arm'],r['update']): r['cosine'] for r in gradients}
    lines = ['# 阶段 D 结束后的三项诊断', '', '## 轮初快照与更新前 KL', '',
             f"已逐一比对 40 个组次，共 {anchor['valid_logits']:,} 个有效候选 logits。"
             f"目标锚点与原轮初 actor 的最大绝对差为 {anchor['max_difference']:.3g}，"
             f"相应候选概率与归档学生概率最大绝对差为 {anchor['max_probability_difference']:.3g}。"
             "四组的权重哈希、原始前缀、候选与掩码均通过核对。"
             "第 1 步使用独立归档前向以及原始学生概率交叉重建，"
             "第 2–10 步直接比对构造目标时保存的 BF16 logits。", '',
             *table(['步','R','Rβ','P','P-shuffled'],
                    [[k, *[f'{lookup[a,k]:.7f}' for a in ARMS]] for k in range(1,11)]),
             '', '上表为 FP32 更新前 KL。每步抽取新的任务，目标强度允许随批次变化。'
             '每个组次的 BF16 目标强度、权重哈希和候选误差见[逐步快照核对](snapshot_audit.tsv)。',
             '', '## 重置 AdamW 的单轮反事实重放', '',
             '每次独立恢复原轮初权重，沿用原目标、全局权重、随机划分、学习率与裁剪。对齐量与移动量均由 FP32 前向测量。',
             '', *table(['步','组','原对齐量','重置对齐量','原移动量','重置移动量','原 ΔKL','重置 ΔKL'],
                     [[r['update'],r['arm'],*[f"{r[v]:+.7f}" for v in
                       ('original_alignment','reset_alignment','original_movement','reset_movement','original_delta_kl','reset_delta_kl')]]
                      for r in replays]),
             '', f"原对齐量非正的 {sum(negative.values())} 次重放中，{sum(counts.values())} 次在重置后转正。"
             "重置同时清除两个矩估计和偏差修正步数，因此该对照只识别完整优化器状态的影响。"
             "[重放数据与输入签名](reset_adam_replay.tsv)", '', '## 原轮初的相邻批平均梯度', '',
             *table(['相邻步','R','Rβ','P','P-shuffled'],
                    [[f'{k-1} → {k}',*[f'{cosine[a,k]:+.5f}' if cosine[a,k] is not None else 'NA' for a in ARMS]]
                     for k in range(2,11)]),
             '', '各轮将完整批的梯度按原任务和轨迹权重累加，裁剪前取值，余弦以 FP64 计算。'
             '相邻轮的任务、轨迹和 actor 都会变化。此表描述两轮梯度的实际一致性。'
             '[梯度范数、内积及哈希](adjacent_gradient_cosines.tsv)', '',
             '[阶段 D 的训练与行为结果](STAGE_D.md) · [诊断汇总](post_d_checks.json)']
    (EVIDENCE / 'POST_D_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    dest = POST / 'reports'
    dest.mkdir(exist_ok=True)
    from .post_d_supplement import report as supplement, OUTPUTS
    extra=supplement()
    names = ('snapshot_audit.tsv','reset_adam_replay.tsv','adjacent_gradient_cosines.tsv',
             'post_d_checks.json','POST_D_DIAGNOSTICS.md',*OUTPUTS)
    hashes = {}
    for name in names:
        shutil.copyfile(EVIDENCE / name,dest / name)
        hashes[name] = digest_file(dest / name)
        assert hashes[name] == digest_file(EVIDENCE / name)
    save(dest / 'delivery.json', dict(cells=40, reset_cases=9, gradient_pairs=36, reanchored_history=extra['history_cells'],
         reanchored_reset=extra['reset_cells'],behavior_models=extra['behavior_models'],public_sha256=hashes))


if __name__ == '__main__':
    if sys.argv[1] == 'snapshot':
        snapshot_report()
    elif sys.argv[1] == 'final':
        final()
    else:
        raise ValueError('Expected snapshot or final')
