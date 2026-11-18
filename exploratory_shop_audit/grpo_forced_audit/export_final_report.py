"""Publish saved final aggregates with explicit recovery and clock limitations."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import subprocess

from .common import DEFAULT_RUN, ROOT, digest_file, read, read_rows, save, utc
from .completion_shutdown import completion_gate


def aggregates(value):
    if isinstance(value, dict):
        return {k: aggregates(v) for k, v in value.items()
                if k != 'records' and not k.endswith('_records') and k != 'variance_by_task'}
    if isinstance(value, list):
        return [aggregates(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output
    reasons = completion_gate(run)
    if reasons:
        raise RuntimeError(reasons)
    if output.exists():
        raise FileExistsError('Preserve prior exports; choose a new directory')
    source = read(run / 'metrics.json')
    updates = Counter(r['update'] for r in read_rows(run / 'training_metrics.jsonl'))
    ledger = read_rows(run / 'cost_ledger.jsonl')
    negative = [{k: r.get(k) for k in ['stage', 'session_id', 'time', 'gpu_hours']}
                for r in ledger if r.get('gpu_hours', 0) < 0]
    audit = {'exported_at': utc(), 'completion_gate_reasons': reasons,
             'completion': read(run / 'campaign_status.json'),
             'final_checkpoint': read(run / 'final_checkpoint.json'),
             'duplicate_update_record_counts': {k: v for k, v in updates.items() if v > 1},
             'cost_verified': not negative, 'negative_cost_entries': negative,
             'cost_note': 'Raw wall-clock ledger retained; negative durations show clock jumps. The recorded total is not verified GPU occupancy. No guessed correction is applied.',
             'counts': {}}
    inputs = [run / n for n in ['metrics.json', 'REPORT.md', 'campaign_status.json',
              'final_checkpoint.json', 'splits.json', 'budget_freeze.json',
              'registration.json', 'cost_ledger.jsonl', 'training_metrics.jsonl', 'RECOVERY.md']]
    table = []
    for label in ['base', 'final']:
        for mode in ['greedy', 'sample']:
            folder = run / 'evaluation' / label / mode
            inputs += [folder / 'episodes.jsonl', folder / 'steps.jsonl']
            summary = source['evaluation'][label + '_' + mode]
            audit['counts'][label + '_' + mode] = summary['all_unweighted']['episodes']
            for metric in ['reward', 'strict_success', 'purchase']:
                item = summary['weighted_all'][metric]
                table.append({'model': label, 'decode': mode, 'metric': metric,
                              'estimate': item['estimate'], 'ci95_low': item['ci95'][0],
                              'ci95_high': item['ci95'][1]})
    for label in ['audit_base', 'audit_post']:
        for name in ['metrics.json', 'states.jsonl', 'candidates.jsonl',
                     'natural/episodes.jsonl', 'bank_a/episodes.jsonl', 'bank_b/episodes.jsonl']:
            inputs.append(run / label / name)
        audit['counts'][label] = {name: len(read_rows(run / label / name)) for name in
            ['states.jsonl', 'candidates.jsonl', 'natural/episodes.jsonl',
             'bank_a/episodes.jsonl', 'bank_b/episodes.jsonl']}
    output.mkdir(parents=True)
    save(output / 'metrics.json', aggregates(source))
    save(output / 'delivery_checks.json', audit)
    save(output / 'provenance.json', {
        'run_id': run.name, 'source_commit_at_export': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'generator_sha256': digest_file(Path(__file__)),
        'input_sha256': {str(p.relative_to(run)): digest_file(p) for p in inputs},
        'raw_records_and_weights': 'Retained on server, not included in Git.'})
    with (output / 'evaluation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ['# 最终交付：100 步 GRPO 与 E1 审计', '',
             '100步训练、基础/最终模型各200条贪心与800条采样评测、E1回放均完成；完整性检查通过。到100步上限停止，未满足预定平台期条件，不自动开展下一轮训练。', '',
             '固定V集属于开发/验证集，不是最终独立留出测试集。下表按冻结易难总体比例加权；区间见evaluation.csv。周期验证每题一次的成功率不可与此表混用。', '',
             '| 模型 | 解码 | 连续奖励 | 完全成功率 |', '|---|---|---:|---:|']
    for label in ['base', 'final']:
        for mode in ['greedy', 'sample']:
            v = source['evaluation'][label + '_' + mode]['weighted_all']
            lines.append(f'| {label} | {mode} | {v["reward"]["estimate"]:.4f} | {v["strict_success"]["estimate"]:.2%} |')
    lines += ['', '交付限制：', '',
              '- 训练有效性检查通过只证明保存的数值、权重更新和前检条件通过；训推概率差异仍未解决，不能据此断言不影响训练。',
              '- checkpoint40恢复曾未通过逐位重建，随后经研究者授权续训；详见下附恢复说明。后续重启从50和90恢复，日志保留重复51/91/92更新记录，不把它们合并为连续无中断的100次更新证据；最高逻辑步号为100。',
              '- GPU时间账本含负时长，显示系统时钟跳变。原始报告的累计GPU小时未核验，不用于成本比较或计费承诺。异常条目见delivery_checks.json；原始账本不改写。',
              '- 以下原始报告中的训练健康、断点恢复和累计成本须结合上述限制阅读。原始数据和模型在服务器，GitHub仅含汇总与来源哈希。', '',
              '## 保存数据生成的原始报告', '', (run / 'REPORT.md').read_text(),
              '## 恢复说明', '', (run / 'RECOVERY.md').read_text(),
              '## 复算与导出', '',
              '从服务器保存数据运行 `report.final_report(run)` 可重算分析；运行 `python -m exploratory_shop_audit.grpo_forced_audit.export_final_report --output 新目录` 可重新导出。不会重新采集轨迹。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print({'output': str(output), 'counts': audit['counts'], 'cost_verified': audit['cost_verified']})


if __name__ == '__main__':
    main()
