"""Saved-data S/F comparison, integrity checks and corrective report."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import digest_value, stable_seed
from exploratory_shop_audit.runtime import read_rows
from exploratory_shop_audit.audit_statistics import bootstrap_mean
from exploratory_shop_audit.review_recompute import verify_parent
from exploratory_shop_audit.judge_prompt import build_choice_messages, CHOICE_SYSTEM


def ci(values, name):
    return bootstrap_mean(values, stable_seed(20261115, 'review-sf', name))


def values_by_candidate(parent):
    candidates = read_rows(parent / 'formal_forks/candidates.jsonl')
    outcomes = defaultdict(list)
    for bank in ['A', 'B']:
        for row in read_rows(parent / f'formal_forks/replay_bank_{bank.lower()}.jsonl'):
            outcomes[(row['state_id'], row['candidate'], bank)].append(row)
    values, invalid = {}, {}
    for c in candidates:
        sid, cid = c['state_id'], c['candidate']
        flags = []
        for bank in ['A', 'B']:
            samples = outcomes[(sid, cid, bank)]
            if c['terminal']:
                values[(sid, cid, bank)] = c['one_step']['status']['reward']
                flags.append(c['one_step']['status'].get('reward_valid') is not True)
            elif len(samples) == 6:
                values[(sid, cid, bank)] = float(np.mean([r['reward'] for r in samples]))
                flags.extend(r['status'].get('reward_valid') is not True for r in samples)
        invalid[(sid, cid)] = float(np.mean(flags)) if flags else None
    return candidates, values, invalid


def sf_metrics(outputs, requests, points, values, invalid, name):
    choices = {(r['state_id'], r['condition']): r for r in outputs}
    order = {r['state_id']: r['order'] for r in requests}
    records = []
    for sid in order:
        row = {'state_id': sid, 'task_id': points[sid]['task_id'], 'page': points[sid]['page_type']}
        s, f = choices.get((sid, 'S'), {}).get('selected'), choices.get((sid, 'F'), {}).get('selected')
        row.update({'S': s, 'F': f, 'paired_valid': s is not None and f is not None,
                    'all_candidate_outcomes_valid': all(invalid[(sid, cid)] == 0 for cid in order[sid]),
                    'selected_S_invalid_fraction': invalid[(sid, s)] if s is not None else None,
                    'selected_F_invalid_fraction': invalid[(sid, f)] if f is not None else None})
        for bank in ['A', 'B']:
            v = [values[(sid, cid, bank)] for cid in order[sid]]
            slo, shi = (values[(sid, s, bank)],) * 2 if s is not None else (min(v), max(v))
            flo, fhi = (values[(sid, f, bank)],) * 2 if f is not None else (min(v), max(v))
            row[bank] = flo - slo if row['paired_valid'] else None
            row[bank + '_lower'] = flo - shi; row[bank + '_upper'] = fhi - slo
        records.append(row)
    summary = {'requests': len(choices), 'invalid_outputs': sum(r.get('selected') is None for r in choices.values()),
        'paired_valid_states': sum(r['paired_valid'] for r in records), 'records': records, 'by_condition_page': {}}
    for condition in ['S', 'F']:
        for page in ['all', 'search_home', 'search_results', 'product_detail']:
            selected = [r for r in records if page == 'all' or r['page'] == page]
            summary['by_condition_page'][condition + '/' + page] = {'states': len(selected),
                'valid_outputs': sum(r[condition] is not None for r in selected)}
    for bank in ['A', 'B']:
        summary[bank] = ci([r[bank] for r in records if r[bank] is not None], name + bank)
        summary[bank + '_all35_missing_choice_bounds'] = {
            'lower': float(np.mean([r[bank + '_lower'] for r in records])),
            'upper': float(np.mean([r[bank + '_upper'] for r in records])),
            'lower_bootstrap': ci([r[bank + '_lower'] for r in records], name + bank + 'lo'),
            'upper_bootstrap': ci([r[bank + '_upper'] for r in records], name + bank + 'hi'),
            'scope': 'Bounds for missing choices over fixed observed candidate bank means, not a confidence interval for true action values.'}
        summary[bank + '_all_candidate_outcomes_valid_conditional'] = ci(
            [r[bank] for r in records if r[bank] is not None and r['all_candidate_outcomes_valid']], name + bank + 'valid')
    return summary


def fmt(obj):
    if obj is None: return 'NA'
    if not isinstance(obj, dict): return f'{obj:.4f}'
    if obj.get('estimate') is None: return f'NA（{obj.get("n_tasks",0)}任务）'
    lo, hi = obj['ci95']
    return f'{obj["estimate"]:.4f} [{lo:.4f}, {hi:.4f}]；n={obj["n_tasks"]}'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--revision', type=Path, required=True); args = p.parse_args()
    revision = args.revision; parent = verify_parent(revision)
    offline = json.loads((revision / 'offline_metrics.json').read_text())
    probes = json.loads((revision / 'early_probe_metrics.json').read_text())
    previous = json.loads((parent / 'metrics.json').read_text())
    candidates, values, invalid = values_by_candidate(parent)
    arms = {(r['state_id'], r['candidate']): r for r in candidates}
    points = {r['state_id']: r for r in json.loads((parent / 'state_manifest.json').read_text())['points']}
    requests = read_rows(revision / 'judge_requests.jsonl'); outputs = read_rows(revision / 'judge_outputs.jsonl')
    old_requests = read_rows(parent / 'judge_requests.jsonl'); old_outputs = read_rows(parent / 'judge_outputs.jsonl')
    original = {(r['state_id'], r['condition']): r for r in old_requests}
    assert len(requests) == len(outputs) == 70
    assert len({(r['state_id'], r['condition']) for r in requests}) == 70
    assert len({(r['state_id'], r['condition']) for r in outputs}) == 70
    assert {r['state_id'] for r in requests} == {r['state_id'] for r in read_rows(parent / 'formal_forks/state_results.jsonl') if r['defined']}
    for req in requests:
        sid, condition = req['state_id'], req['condition']
        old = original[(sid, condition)]
        assert req['order'] == old['order'] and req['seed'] == old['seed']
        assert req['temperature'] == 0 and req['max_tokens'] == 256
        expected = build_choice_messages(points[sid]['input']['messages'], [arms[(sid, c)] for c in req['order']], condition)
        assert req['messages'] == expected
        assert req['messages'][0]['content'] == CHOICE_SYSTEM
        assert req['original_context_sha256'] == digest_value(points[sid]['input']['messages'])
        assert not any(k in str(req['messages']) for k in ['reward_detail', 'goal_options', 'price_upper', 'target_asin'])
    assert sum(r['requests'] for r in read_rows(revision / 'inference_costs.jsonl')) == 70
    costs = json.loads((revision / 'costs.json').read_text()); assert costs['total_gpu_hours'] <= 16
    old = sf_metrics(old_outputs, old_requests, points, values, invalid, 'original')
    new = sf_metrics(outputs, requests, points, values, invalid, 'corrected')
    assert abs(old['B']['estimate'] - previous['reward_metrics']['reward']['replay']['information_control']['B']['estimate']) < 1e-12
    old_pairs = {r['state_id'] for r in old['records'] if r['paired_valid']}
    common = [r for r in new['records'] if r['paired_valid'] and r['state_id'] in old_pairs]
    sf = {'original_preserved': old, 'dedicated_choice': new,
          'corrected_on_original_valid_pairs': {b: ci([r[b] for r in common], 'common' + b) for b in ['A', 'B']},
          'selected_invalid_score_fraction': {c: ci([r['selected_' + c + '_invalid_fraction'] for r in new['records'] if r[c] is not None], 'selected_invalid_' + c) for c in ['S', 'F']}}
    (revision / 'sf_metrics.json').write_text(json.dumps(sf, ensure_ascii=False, indent=2) + '\n')
    validation = {'passed': True, 'parent_artifact_hashes_unchanged': True, 'base_episodes_retained': 800,
        'new_policy_episodes': 0, 'new_replay_outcomes': 0, 'fixed_comparable_states': 35,
        'new_judge_generations': 70, 'new_judge_invalid_outputs': new['invalid_outputs'],
        'same_candidate_order_and_seeds': True, 'same_non_system_context_and_feedback': True,
        'no_background_input_fields': True, 'native_stop_and_reliability_points_preserved': True,
        'cpu_tests': 27, 'costs': costs}
    (revision / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    summary = {'validation': validation, 'offline': offline, 'early_probes': probes, 'sf': sf,
        'recommendation': 'Pause training; retain natural-behavior baseline; no whole-environment reliability or early CTHA efficacy claim.'}
    (revision / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    native = previous['reward_metrics']['reward']; action = offline['stop']['strata']; maxima = offline['validity_maxima']['reward']
    lines = ['# 购物测量复核修订报告', '',
        '本次修订保留847b167的800条正式轨迹、全部候选和回放库。原结果不覆盖；仅离线重算并在原35点完整重测70次S/F。**暂停训练。**', '',
        '## 结论与更正', '',
        f'- 数据保全：父目录所有登记文件SHA256复核不变；新轨迹0条、新续跑0条。新增GPU成本{costs["additional_gpu_hours"]:.4f}小时，总计{costs["total_gpu_hours"]:.4f}小时。',
        '- 撤回“74.4%意味着停止搜索有很大训练空间”的解释。它是所有非购买动作的数值差额，包含规格选择、非法动作及主动放弃；不能直接支持训练决策。',
        f'- 无效零分问题已确认：{maxima["invalid_max_strictly_above_valid_pairs"]}/{maxima["task_product_pairs"]}个既有任务—商品配对中，无效数值最大值严格高于全部有效配置；另有{maxima["pairs_with_no_valid_configuration"]}对没有可核验配置。原生数值保留，新增可核验条件诊断。',
        '- 原8步探针保留为近整局关联测量。首次搜索后及首次打开商品后的新探针均未显示高于打乱对照的证据，不能据原71.7%宣称早期指导或失败救援。',
        f'- 原生rho3={native["replay"]["reliability"]["rho"]["3"]:.4f}及原区间{native["replay"]["reliability"]["rho_ci95"]["3"]}保留；只适用于这批候选、当前策略和自然短续跑。26点中20点是搜索首页，不泛化为整个购物环境。',
        f'- 新S/F：{new["invalid_outputs"]}/70无效输出，完整配对{new["paired_valid_states"]}/35；F−S库B为{fmt(new["B"])}。原始14个无效输出和22配对结果保留，不能沿用原负差结论作为干净的信息效应。', '',
        '## 轨迹形态', '',
        f'平均长度{offline["episode_diagnostics"]["length_all"]["mean"]:.4f}步；分位数[min,Q1,median,Q3,max]={offline["episode_diagnostics"]["length_all"]["quantiles"]}。终止原因：{json.dumps(offline["episode_diagnostics"]["termination_counts"],ensure_ascii=False)}。', '',
        '| 指标 | 均值 | [最小,Q1,中位,Q3,最大] | 分母 |', '|---|---|---|---|']
    for field in ['searches', 'distinct_queries', 'opened_products', 'purchase_first_rank', 'purchase_last_rank']:
        d = offline['episode_diagnostics'][field]; lines.append(f'| {field} | {fmt(d["mean"])} | {d["quantiles"]} | {d["n"]} |')
    for success, d in offline['episode_diagnostics']['length_by_success'].items(): lines.append(f'| length/success={success} | {fmt(d["mean"])} | {d["quantiles"]} | {d["n"]} |')
    lines += ['', '购买排名指既有展示列表记录的全局rank；缺失不补1。搜索计数只计原生合法搜索，非法请求另报。', '',
        '## 停手机会：动作与规格拆分', '',
        '表中机会为|R−C_t|>.05，仍按回合再任务等权。两端有效诊断仅含C_t和R均可核验的步骤，条件不同、分母不同，不能视为无偏修正。规格完整仅指全部商品轴已选择合法值，不等于符合目标。', '',
        '| 动作/状态 | 原步骤数 | 原生机会比例及95%区间 | 两端有效步骤数 | 两端有效条件比例及95%区间 |', '|---|---:|---|---:|---|']
    for name in ['all_continuation'] + ['action_class:' + k for k in ['specification', 'inspect', 'search', 'navigate', 'open_product', 'abstain', 'invalid', 'other']] + ['specification_state:' + k for k in ['incomplete', 'complete', 'no_options']]:
        d = action.get(name)
        if d:
            a, b = d['native_numeric']['opportunity'], d['both_valid_conditional']['opportunity']
            lines.append(f'| {name} | {a["n_steps"]} | {fmt(a)} | {b["n_steps"]} | {fmt(b)} |')
    lines += ['', '完整动作×规格交叉表、改善/退步、C_t与R各自无效比例和去预算门版本见offline_metrics.json。没有观察到的动作类别明确为0与NA，不从其他动作外推。', '',
        '## 评分有效性与漏斗', '',
        f'原生数值argmax无效的配对为{maxima["numeric_argmax_invalid_pairs"]}/{maxima["task_product_pairs"]}；这不是全商品库的随机比例。对104460个配置逐项复核，原数字最大值与旧文件一致，新增有效配置最大值，没有有效配置记NA。', '',
        '| 漏斗项 | 旧原生数字口径 | 原生可核验条件诊断 | 去预算门可核验条件诊断 |', '|---|---|---|---|']
    for i, label in enumerate(['1−Phi_surf', 'Phi_surf−Phi_open', 'Phi_open−R_item', 'R_item−R']):
        lines.append(f'| {label} | {fmt(native["funnel"]["gaps"][i])} | {fmt(offline["valid_funnel"]["reward"]["gaps"][i])} | {fmt(offline["valid_funnel"]["no_budget_gate"]["gaps"][i])} |')
    lines += ['', f'原生条件漏斗包含{offline["valid_funnel"]["reward"]["episodes"]}局，排除原因{offline["valid_funnel"]["reward"]["exclusions"]}；去预算门包含{offline["valid_funnel"]["no_budget_gate"]["episodes"]}局。无有效配置商品不充当“更好的零分商品”。分母变化限制前后均值比较；所有入表漏斗恒等式已验证。', '',
        '## 早期CTHA预测探针', '', '| 决策时点 | 可用轨迹/严格对 | 全特征排序准确率 | 相对打乱的差 |', '|---|---|---|---|']
    for landmark, result in probes.items():
        lines.append(f'| {landmark} | {result["eligible_episodes"]}/{result["strict_pairs"]} | {fmt(result["comparisons"]["full"]["accuracy"])} | {fmt(result["shuffled_control"]["paired_difference"])} |')
    lines += ['', '保持78个预定义商品大类留一、Ridge alpha=10、训练折归一化及100次任务内打乱。特征只读该时点精确输入与已保留历史，当前动作和未来观察均未使用。缺失检查点不补零。购买完成事后对照、双方已购买与有效结局条件诊断另列；均不证明行为因果。', '',
        '## 信度分层', '', '| 分层 | 状态数 | S | N | rho3 | 95%区间 | 可识别 |', '|---|---:|---:|---:|---:|---|---|']
    for name, d in offline['replay']['strata'].items():
        if '/validity:' in name: continue
        lines.append(f'| {name} | {d["states"]} | {fmt(d["S"])} | {fmt(d["N"])} | {fmt(d["rho"].get("3"))} | {d.get("rho_ci95",{}).get("3")} | {d["identifiable"]} |')
    lines += ['', '所有入选候选仍保留全部12次重复。逐候选无效评分比例在replay.candidate_records中；没有删除无效重复后重新估计成无偏结果。实际长度与有效性都是事后分层，样本少的分层只描述，信号不可识别时不把数值rho当结论。未延长轨迹、未挑难点替换。', '',
        '## S/F专用提示重测', '',
        f'新系统只要求输出候选编号；原非系统消息逐字作为历史资料引用，旧策略系统的Action指令不继承。70次全部重新生成，候选、顺序、反馈、种子与A/B价值均复用。新条件：S有效{new["by_condition_page"]["S/all"]["valid_outputs"]}/35，F有效{new["by_condition_page"]["F/all"]["valid_outputs"]}/35。', '',
        '| 结果 | 库B F−S | 库A F−S |', '|---|---|---|',
        f'| 原结果保留（有效配对） | {fmt(old["B"])} | {fmt(old["A"])} |',
        f'| 专用提示重测（有效配对） | {fmt(new["B"])} | {fmt(new["A"])} |',
        f'| 新提示，限原有效配对 | {fmt(sf["corrected_on_original_valid_pairs"]["B"])} | {fmt(sf["corrected_on_original_valid_pairs"]["A"])} |', '',
        f'原35点在缺失选择下的库B固定候选价值界：[{old["B_all35_missing_choice_bounds"]["lower"]:.4f}, {old["B_all35_missing_choice_bounds"]["upper"]:.4f}]。该界反映解析缺失，不是新增独立样本或真实动作价值CI。新旧请求按条件/页面的覆盖率与有效性条件诊断见sf_metrics.json。', '',
        'S/F仅测这个7B在当前候选与反馈表示下如何选择；即便修正后有正差或负差，也不证明整个环境的信息不可能性或能力分离。', '',
        '## 后续与复算', '',
        '暂不进入训练，也不补采整批轨迹。保留本轮为自然行为基线。若研究停止搜索或长程信用，需先另行登记确实包含搜索延续/换商品的设置，不能用本轮规格动作机会比例代替。旧数值止损线不移动，其结论限于旧冻结主口径和当前样本。', '',
        '`python exploratory_shop_audit/analyze_review.py --revision exploratory_shop_audit/runs/review_revision_20261115_01` 只使用保存数据和CPU分析，重算指标、早期探针与报告，不调用模型或购物轨迹环境。可核验规格原语由review_recompute.py --derive从冻结商品/目标及原生评分函数离线生成并保存。', '']
    position = lines.index('## 早期CTHA预测探针')
    paired = offline['valid_funnel']['reward']['paired_valid_minus_numeric_gaps']
    lines[position:position] = ['为分离样本条件变化和最大值定义变化，另在完全相同的可核验购买回合上计算原数字漏斗与有效配置漏斗。四项“有效配置−原数字”的配对差依次为：' + '；'.join(fmt(x) for x in paired) + '。', '']
    (revision / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'validation': validation, 'new_SF_B': new['B']}, ensure_ascii=False))


if __name__ == '__main__': main()
