"""Saved-data evidence, feedback proxy, optional probe, figures and report."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.analyze_saved import (estimate, replay_metrics, shown, task_equal,
                                                 weighted_correlation)
from exploratory_shop_audit.data import stable_seed
from exploratory_shop_audit.predictive_probe import run_probe
from exploratory_shop_audit.derive_environment import item_text
from exploratory_shop_audit.runtime import read_rows


def feedback_proxy(run, steps, maxima, values, metric):
    points = json.loads((run / 'state_manifest.json').read_text())['points']
    point_by_id = {p['state_id']: p for p in points}
    histories = defaultdict(set)
    for point in points:
        for step in steps:
            if step['trajectory_id'] == point['trajectory_id'] and step['step'] <= point['step']:
                histories[point['state_id']] |= shown(step['before'])
    rows = []
    for c in read_rows(run / 'formal_forks/candidates.jsonl'):
        sid, tid, cid = c['state_id'], c['task_id'], c['candidate']
        if c['parsed_action'][0] not in ['search', 'click'] or c['terminal']:
            continue
        ids = shown(c['one_step']['state'])
        if not ids:
            continue
        quality = max(maxima[(tid, a)]['r_star'][metric] for a in ids)
        before_ids = histories[sid]
        before = max((maxima[(tid, a)]['r_star'][metric] for a in before_ids), default=None)
        # Empty prior catalog has no defined maximum; do not manufacture 0.
        gain = max(0.0, quality - before) if before is not None else None
        row = {'state_id': sid, 'task_id': tid, 'candidate': cid, 'display_quality': quality,
               'previous_best': before, 'cumulative_gain': gain,
               'new_products': len(ids - before_ids), 'shown_products': len(ids),
               'value_A': values.get((sid, cid, 'A')), 'value_B': values.get((sid, cid, 'B'))}
        rows.append(row)
    correlations = {}
    for feature in ['display_quality', 'cumulative_gain']:
        for bank in ['A', 'B']:
            grouped = defaultdict(list)
            for row in rows:
                if row[feature] is not None and row['value_' + bank] is not None:
                    grouped[row['state_id']].append(row)
            centered = []
            for sid, group in grouped.items():
                if len(group) < 2:
                    continue
                x_mean = np.mean([r[feature] for r in group]); y_mean = np.mean([r['value_' + bank] for r in group])
                centered.extend((r['task_id'], sid, r[feature] - x_mean, r['value_' + bank] - y_mean) for r in group)
            correlations[feature + '_' + bank] = {'correlation': weighted_correlation(centered),
                'tasks': len({r[0] for r in centered}), 'candidate_rows': len(centered)}
    return {'label': 'privileged displayed-product attainable-score proxy; not all feedback information value',
            'eligible_candidate_rows': len(rows), 'total_candidates': len(read_rows(run / 'formal_forks/candidates.jsonl')),
            'quality': task_equal(rows, lambda r: r['display_quality'], metric + 'feedback_quality'),
            'gain': task_equal(rows, lambda r: r['cumulative_gain'], metric + 'feedback_gain'),
            'correlations': correlations, 'records': rows}


def visibility_summary(run):
    pairs = read_rows(run / 'visibility_pairs.jsonl')
    episode_groups = defaultdict(list)
    for pair in pairs:
        episode_groups[pair['trajectory_id']].append(pair)
    result = {'pairs': len(pairs), 'manual_review_pairs': len(read_rows(run / 'visibility_manual_review.jsonl')),
              'definition': 'Product-specific text blocks only, excluding task/query/other products and assistant generations; lexical proxies, not ground truth.',
              'label_counts': {}}
    for name in ['historical_label', 'current_input_label', 'historical_normalized_label', 'current_input_normalized_label']:
        result['label_counts'][name] = dict(Counter(r[name] for r in pairs))
        for label in ['support_proxy', 'conflict_proxy', 'unknown']:
            per_episode = [{'task_id': rows[0]['task_id'], 'fraction': np.mean([r[name] == label for r in rows])} for rows in episode_groups.values()]
            result[name + '_' + label] = task_equal(per_episode, lambda r: r['fraction'], name + label)
    result['history_evidence_lost'] = sum(p['history_lost_from_current'] for p in pairs)
    result['history_lost_with_truncation'] = sum(p['history_lost_from_current'] and p['truncation_present'] for p in pairs)
    return result


def fmt(value):
    if value is None: return 'NA'
    if isinstance(value, dict):
        if value['estimate'] is None: return f'NA（任务数={value["n_tasks"]}）'
        lo, hi = value['ci95']
        return f'{value["estimate"]:.4f} [{lo:.4f}, {hi:.4f}]（任务数={value["n_tasks"]}）'
    return f'{value:.4f}'


def make_figures(run, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = run / 'figures'
    destination.mkdir(exist_ok=True)
    native = metrics['reward_metrics']['reward']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([r['range'] for r in native['groups']['records']], bins=20)
    ax.set(xlabel='Within-task reward range (4 trajectories)', ylabel='Tasks')
    fig.tight_layout(); fig.savefig(destination / 'reward_ranges.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    for offset, (name, label) in enumerate([('reward', 'Native Reward v3'), ('no_budget_gate', 'No budget gate')]):
        gaps = metrics['reward_metrics'][name]['funnel']['gaps']
        ax.bar(x + offset * .35, [g['estimate'] or 0 for g in gaps], width=.35, label=label)
    ax.set_xticks(x + .175, ['1 - surfaced', 'surfaced - opened', 'opened - item', 'item - realized'])
    ax.set_ylabel('Task-equal mean gap'); ax.legend(); fig.tight_layout()
    fig.savefig(destination / 'funnel.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for offset, (name, label) in enumerate([('reward', 'Native'), ('no_budget_gate', 'No budget gate')]):
        reliability = metrics['reward_metrics'][name]['replay']['reliability']
        for i, depth in enumerate(['1', '3', '6']):
            v = reliability['rho'].get(depth)
            interval = reliability.get('rho_ci95', {}).get(depth)
            if v is not None:
                location = i + offset * .1
                ax.scatter([location], [v], color=f'C{offset}', label=label if i == 0 else None)
                if interval: ax.plot([location, location], interval, color=f'C{offset}', alpha=.7)
    ax.axhline(.7, color='gray', linestyle='--'); ax.set_xticks(range(3), ['1', '3', '6'])
    ax.set(xlabel='Continuations averaged', ylabel='Cross-covariance reliability (unclipped)')
    ax.legend(); fig.tight_layout(); fig.savefig(destination / 'reliability.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    overlaps = [p['jaccard'] for r in read_rows(run / 'formal_forks/state_results.jsonl') for p in r['query_overlaps']]
    ax.hist(overlaps, bins=np.linspace(0, 1, 11)); ax.set(xlabel='Search result-set Jaccard', ylabel='Candidate pairs (descriptive)')
    fig.tight_layout(); fig.savefig(destination / 'query_overlap.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reuse-probe', action='store_true', help='Reuse completed probe for report-only regeneration')
    args = parser.parse_args()
    run = args.run.resolve()
    metrics = json.loads((run / 'metrics.json').read_text())
    episodes = [r for r in read_rows(run / 'episodes.jsonl') if r['phase'] == 'formal']
    completed = {r['trajectory_id'] for r in episodes}
    steps = [r for r in read_rows(run / 'steps.jsonl') if r['trajectory_id'] in completed]
    maxima = {(r['task_id'], r['asin']): r for r in read_rows(run / 'r_star.jsonl')}
    tasks = {r['task_id']: r for r in json.loads((run / 'tasks.json').read_text())['formal']}
    states = list({r['state_id']: r for r in read_rows(run / 'formal_forks/state_results.jsonl')}.values())
    metrics['visibility'] = visibility_summary(run)
    # Scope current-item proxies separately from forgotten evidence about any
    # previously displayed product. Inspect actual retained user observations.
    goals = json.loads((run / 'snapshots/goals_frozen.json').read_text())
    history_observations = defaultdict(list)
    history_products = defaultdict(set)
    lost_records = []
    for row in steps:
        eid, tid = row['trajectory_id'], row['task_id']
        history_observations[eid].append(row['before']['observation'])
        history_products[eid] |= shown(row['before'])
        if not row['input']['removed_pairs']:
            continue
        goal = goals[int(tid.split(':')[1])]
        requirements = [str(x) for x in goal.get('expected_core_functions', [])]
        requirements += [str(x.get('value', '')) for x in goal.get('required_options_by_key', {}).values()]
        for asin in sorted(history_products[eid]):
            historical = '\n'.join(item_text(obs, asin) for obs in history_observations[eid])
            current = '\n'.join(item_text(m['content'], asin) for m in row['input']['messages'] if m['role'] == 'user')
            for requirement in requirements:
                if requirement and requirement in historical and requirement not in current:
                    lost_records.append({'trajectory_id': eid, 'task_id': tid, 'step': row['step'],
                        'asin': asin, 'requirement': requirement, 'historical_product_text': historical,
                        'retained_product_text': current})
    metrics['truncation']['all_historical_products_lost_requirement_proxy_count'] = len(lost_records)
    metrics['truncation']['unique_lost_episode_product_requirement_pairs'] = len({(r['trajectory_id'], r['asin'], r['requirement']) for r in lost_records})
    (run / 'truncation_lost_evidence.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in lost_records))
    replay_steps = read_rows(run / 'formal_forks/replay_steps.jsonl')
    metrics['truncation']['replay'] = {'decisions': len(replay_steps),
        'truncated_decisions': sum(bool(r['input']['removed_pairs']) for r in replay_steps),
        'removed_tokens': sum(r['input']['removed_tokens'] for r in replay_steps),
        'max_input_tokens': max([r['input']['token_count'] for r in replay_steps], default=0)}
    (run / 'truncation_audit.json').write_text(json.dumps(metrics['truncation'], ensure_ascii=False, indent=2) + '\n')
    metrics['price_sensitivity'] = {
        'reward_difference': task_equal(episodes, lambda r: r['no_budget_gate'] - r['reward'], 'price_difference'),
        'changed_reward_fraction': task_equal(episodes, lambda r: abs(r['no_budget_gate'] - r['reward']) > 1e-8, 'price_fraction'),
        'purchase_price_unverifiable': sum(r['status'].get('reward_detail', {}).get('hard_gates', {}).get('budget', {}).get('status') == 'unverifiable' for r in episodes)}
    metrics['query_overlap'] = {'states': sum(bool(r['query_overlaps']) for r in states),
        'jaccard': task_equal(states, lambda r: np.mean([p['jaccard'] for p in r['query_overlaps']]) if r['query_overlaps'] else None, 'jaccard'),
        'identical_set_fraction': task_equal(states, lambda r: np.mean([p['identical'] for p in r['query_overlaps']]) if r['query_overlaps'] else None, 'identical_queries')}
    metrics['candidate_coverage'] = {'states': len(states), 'undefined_states': sum(not r['defined'] for r in states),
        'valid_draft_mass': task_equal(states, lambda r: r['valid_drafts'] / 16, 'valid_mass'),
        'covered_draft_mass': task_equal(states, lambda r: r['covered_drafts'] / 16, 'covered_mass'),
        'omitted_valid_mass': task_equal(states, lambda r: r['omitted_valid_drafts'] / 16, 'omitted_mass')}
    state_seconds = sum(r['total_seconds'] for r in states)
    candidate_count = sum(r['candidate_count'] for r in states)
    crossbank = metrics['reward_metrics']['reward']['replay']['crossbank_records']
    repeated = sum(r['A_select_B_evaluate'] > 0 and r['B_select_A_evaluate'] > 0 for r in crossbank)
    repeated_005 = sum(r['A_select_B_evaluate'] > .05 and r['B_select_A_evaluate'] > .05 for r in crossbank)
    metrics['fork_cost'] = {'actual_seconds_with_gpu_resident': state_seconds,
        'prefix_restore_seconds': sum(r['restore_seconds'] for r in states),
        'candidate_estimates_including_exact_terminal': candidate_count,
        'seconds_per_candidate_estimate': state_seconds / candidate_count if candidate_count else None,
        'both_crossbank_gains_positive_states': repeated,
        'both_crossbank_gains_gt_005_states': repeated_005,
        'seconds_per_both_positive_opportunity': state_seconds / repeated if repeated else None,
        'seconds_per_both_gt_005_opportunity': state_seconds / repeated_005 if repeated_005 else None,
        'warning': 'Empirical crossbank reproducibility criteria; not proof of true beneficial actions.'}
    for metric in ['reward', 'no_budget_gate']:
        replay, values = replay_metrics(run, metric)
        metrics['reward_metrics'][metric]['feedback_proxy'] = feedback_proxy(run, steps, maxima, values, metric)
        metrics['reward_metrics'][metric]['target_unattainable_tasks'] = [tid for tid in sorted({e['task_id'] for e in episodes})
            if maxima[(tid, tasks[tid]['target_asin'])]['r_star'][metric] < 1 - 1e-8]
    if not args.reuse_probe or 'probe' not in metrics:
        metrics['probe'] = run_probe(episodes, steps, metrics['reward_metrics']['reward']['groups'])
    metrics['runtime_errors'] = read_rows(run / 'errors.jsonl')
    (run / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    make_figures(run, metrics)
    native = metrics['reward_metrics']['reward']; diagnostic = metrics['reward_metrics']['no_budget_gate']
    rel = native['replay']['reliability']
    rules = metrics['stop_rules']
    def verdict(name):
        result = rules[name]['triggered']
        if result is None: return '未触发但证据不足（该阈值不可判定）'
        if result:
            return '触发止损：不继续该方向在当前设置下的训练或扩展'
        if name == 'stop_baseline':
            return '未触发止损；值得下一轮小规模训练对照，但尚无降方差或策略增益证据；本轮未启动训练'
        return '未触发但失败救援证据不足；本轮不自动启动训练'
    lines = ['# 购物环境探索测量报告', '',
        '**环境：社区 Environment v2.1 / Reward v3，35步原生终止；不是旧ShopSimulator奖励的复现。**', '',
        '## 一页结论', '',
        f'- 规模：{metrics["formal_tasks"]}题、{metrics["formal_episodes"]}条正式轨迹；{len(states)}个分叉点，{native["replay"]["defined_states"]}个可定义。采集完成标记：{metrics["collection_complete"]}。',
        f'- 环境验收：20×2策略试运行、20前缀双进程恢复、20隔离购买对分已通过；全库239,925规格值解析检查通过。',
        f'- 实际累计成本：{metrics["costs"]["gpu_hours"]:.4f} GPU小时（含初始化、失败启动与试运行），上限16；预算冻结为200题＋60状态。',
        f'- 预算门敏感性：去预算门后奖励改变比例 {fmt(metrics["price_sensitivity"]["changed_reward_fraction"])}；均值差 {fmt(metrics["price_sensitivity"]["reward_difference"])}。',
        f'- 历史截断：{metrics["truncation"]["truncated_decisions"]}/{metrics["truncation"]["decisions"]}决策；{metrics["truncation"]["truncated_episodes"]}回合；移除{metrics["truncation"]["removed_tokens"]}token。',
        f'- CTHA：无奖励差异组 {fmt(native["groups"]["equal"])}，全失败组 {fmt(native["groups"]["all_failed"])}。{verdict("ctha")}。',
        f'- 停手：决策步可购买覆盖 {fmt(native["stop"]["coverage"])}；实际继续操作机会比例 {fmt(native["stop"]["opportunity"])}。{verdict("stop_baseline")}。',
        f'- 回放：主估计可用状态 {rel["states"]}，S={fmt(rel["S"])}, N={fmt(rel["N"])}, rho3={fmt(rel["rho"].get("3"))}；可识别={rel["identifiable"]}。{verdict("replay_noise")}。',
        '', '以上区间为按任务聚类bootstrap 2000次的95%区间，先回合再任务等权。诊断性多指标不是确认性检验。未触发阈值不等于方法有效。', '',
        '## 奖励差异、停手和漏斗', '',
        f'奖励差异任务组 {native["groups"]["mixed_groups"]}；严格高低轨迹对 {native["groups"]["strict_pairs"]}。零分同分组 {fmt(native["groups"]["zero_equal"])}；负分同分组 {fmt(native["groups"]["negative_equal"])}；全部满分组 {fmt(native["groups"]["full_equal"])}。', '',
        f'探针状态：`{metrics["probe"]["status"]}`。仅作预测性检查，详见metrics.json，不推断行为因果或训练收益。', '',
        f'停手改善比例 {fmt(native["stop"]["improvement"])}；退步比例 {fmt(native["stop"]["regression"])}；回合覆盖 {fmt(native["stop"]["episode_coverage"])}。不可购买状态写NA，分母为零不补零。', '',
        '| 漏斗项 | 原生奖励 | 去预算门诊断 |', '|---|---|---|']
    for i, name in enumerate(['1−Phi_surf', 'Phi_surf−Phi_open', 'Phi_open−R_item', 'R_item−R']):
        lines.append(f'| {name} | {fmt(native["funnel"]["gaps"][i])} | {fmt(diagnostic["funnel"]["gaps"][i])} |')
    lines += ['', f'未纳入漏斗的回合：原生{len(native["funnel"]["excluded"])}，诊断{len(diagnostic["funnel"]["excluded"])}；逐条原因保留。目标商品最高原生分不足1的任务{len(native["target_unattainable_tasks"])}。最大值来自原生评分函数穷举，不使用1000组合截断。', '',
        '![组内奖励极差](figures/reward_ranges.png)', '![双口径漏斗](figures/funnel.png)', '',
        '## 回放、候选集和信息', '', '| 续跑次数 | 原生rho及95%区间 | 去预算门rho及95%区间 |', '|---|---|---|']
    for depth in ['1', '3', '6']:
        def reliability_cell(r):
            return f'{fmt(r["rho"].get(depth))}; {r.get("rho_ci95", {}).get(depth)}'
        lines.append(f'| {depth} | {reliability_cell(rel)} | {reliability_cell(diagnostic["replay"]["reliability"])} |')
    lines += ['', f'两库交叉选择相对原动作收益 {fmt(native["replay"]["crossbank_gain"])}。主估计只含至少两个非终止候选的状态，S不裁零；包含确定终止臂、草稿频次加权及100次随机拆分仅另列。', '',
        f'查询结果Jaccard {fmt(metrics["query_overlap"]["jaccard"])}；集合完全相同率 {fmt(metrics["query_overlap"]["identical_set_fraction"])}。候选覆盖草稿质量 {fmt(metrics["candidate_coverage"]["covered_draft_mass"])}。这些是候选集条件下读数，不是完整策略概率。', '',
        f'S/F差异：库B {fmt(native["replay"]["information_control"]["B"])}；库A {fmt(native["replay"]["information_control"]["A"])}。解析或输入失败请求{native["replay"]["information_control"]["invalid_requests"]}个，未静默修复。一步反馈清洗和原文保留在候选及judge请求文件中。', '',
        '![续跑信度](figures/reliability.png)', '![检索集合重合](figures/query_overlap.png)', '',
        f'每候选估值实耗 {fmt(metrics["fork_cost"]["seconds_per_candidate_estimate"])} 秒（含确定终止臂）；两方向交叉库收益均为正的点{repeated}个，均超过.05的点{repeated_005}个。各机会成本及分离的前缀恢复成本见metrics.json；零机会分母写NA。', '',
        '## 可见证据与限制', '',
        f'商品专属证据配对{metrics["visibility"]["pairs"]}条，人工待复核抽样{metrics["visibility"]["manual_review_pairs"]}条；逐字与规范化匹配分支持、显式否定冲突、未知。任务文本和其他商品的词不计入该商品证据。历史证据当前丢失{metrics["visibility"]["history_evidence_lost"]}条，其中伴随截断{metrics["visibility"]["history_lost_with_truncation"]}条。', '',
        '该环境的预算来自任务显式文本，实际采用variant价格；不能沿用旧版“随机价格不可见”的结论。R_no_budget_gate仅对同一实际购买取消预算硬门，保留类别、软偏好及其余原生评分，非购买终止奖励不变。它不等于旧线性去价格分。', '',
        '后台商品质量代理仅表示一步展示商品中可达的最高分，不是一步反馈全部信息价值的上界。后台评分从未进入策略或S/F输入。负奖励、主动停手和原生循环终止均保留；系统失败另记errors.jsonl。', '',
        '## 复算', '',
        '仅依赖保存数据：`python exploratory_shop_audit/analyze.py --run RUN_DIR`。精确商品评分原语由 `derive_environment.py` 使用冻结环境生成到r_star.jsonl；报告与图表复算无需环境或模型。', '',
        '原始轨迹、模型输入token、后台明细、候选草稿、A/B续跑、S/F请求、冻结表和失败证据保存在运行目录。GitHub仅提交代码、报告与可审阅的汇总/验证证据，不提交权重或排除的原始回放。', '']
    index = lines.index('## 奖励差异、停手和漏斗')
    lines[index:index] = ['下一步判断：优先考虑停手基线的小规模训练对照；CTHA保留为待验证方向，先区分购买完成预测与失败救援信号。ALFWorld低信度现象未在本样本主口径复现，停止追加以低信度为动机的回放研究。本轮没有训练、32B评判或main合并。', '']
    if metrics['probe']['status'] == 'COMPLETED_PREDICTIVE_ONLY':
        probe = metrics['probe']; comparisons = probe['comparisons']
        index = next(i for i, line in enumerate(lines) if line.startswith('探针状态：')) + 1
        lines[index:index] = ['', f'前8步探针排序准确率 {fmt(comparisons["full"]["accuracy"])}；动作／进度对照 {fmt(comparisons["action_progress"]["accuracy"])}；最终有效购买单变量 {fmt(comparisons["final_valid_purchase_diagnostic"]["accuracy"])}。相对100次组内打乱的差 {fmt(probe["shuffled_control"]["paired_task_accuracy_difference"])}。', '',
            f'两条轨迹均购买时，探针准确率 {fmt(comparisons["full"]["both_purchased_accuracy"])}，严格对{comparisons["full"]["both_purchased_pairs"]}个。前8步都未购买的子集只有{comparisons["full"]["early_purchase"]["False"]["pairs"]}个严格对，不能据此宣称全失败救援。78个商品大类留一验证不是ALFWorld原任务类型复现。', '']
    index = lines.index('## 可见证据与限制') + 1
    lines[index:index] = ['', 'S/F容量偏离：16个尚未执行的完整请求超过8192输入，按信息对照生成前登记的SHOP-009，使用同checkpoint原生32768总上下文补执行。策略与续跑仍为8192；S/F前文相同，未删除F反馈，已有选择和解析失败未重跑。', '',
        f'以下可见性配对只针对当前打开商品。另对全部历史展示商品核查截断：丢失要求文本证据代理{len(lost_records)}条（逐决策计数），明细见truncation_lost_evidence.jsonl；不能把当前商品证据未丢失解释为整段历史没有损失。正式续跑截断{metrics["truncation"]["replay"]["truncated_decisions"]}/{len(replay_steps)}个决策。社区公开key_attributes与原版页面不同，不能直接迁移旧环境的信息限制结论。', '']
    (run / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'probe': metrics['probe']['status'], 'report': str(run / 'REPORT.md')}))


if __name__ == '__main__':
    main()
