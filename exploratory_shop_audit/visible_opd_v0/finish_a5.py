"""Write the deidentified A.5 report and attach its evidence index."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .acceptance import digest, rows
from .packets import MATCH, POLICIES

NAMES={'none':'无包','full':'完整包','product':'商品包','evidence':'当页证据包','placeholder':'完整包占位文本'}

def quantiles(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p10_p90=np.quantile(a,[.1,.9]).tolist())

def extra_diagnostics(run):
    """Describe the actual available pairs, preserving action-class/task coverage."""
    lengths=defaultdict(list)
    for p in [*run.glob('smoke_batch_*/score_inputs.jsonl'), *run.glob('formal_batch_*/score_inputs.jsonl')]:
        for r in rows(p):
            for m,ref in MATCH.items():
                real,empty=r['views'][m],r['views'][ref]
                if real['reason'] or empty['reason']:continue
                lengths[m].append(real['length']-empty['length'])

    repaired={(r['goal'],r['rep'],r['step']):r['classes'] for r in rows(run/'parser_repair.jsonl')}
    delta=defaultdict(list)
    kl_nonzero=defaultdict(list)
    for p in [*run.glob('scores_*.jsonl'),*run.glob('formal_scores_*.jsonl')]:
        for r in rows(p):
            labels=np.array(repaired.get((r['goal'],r['rep'],r['step']),r['classes']))
            valid=np.array(r['candidate_mask'],dtype=bool)
            for mode,score in r['scores'].items():
                for label in set(labels):
                    selected=labels==label
                    k='/'.join([mode,r['page'],label])
                    kl_nonzero[k].extend(np.asarray(score['kl'])[selected]>1e-5)
                    if mode in MATCH and MATCH[mode] in r['scores']:
                        diff=np.asarray(score['d'])-np.asarray(r['scores'][MATCH[mode]]['d'])
                        delta[k].extend(diff[selected][valid[selected]])
    return dict(
        paired_packet_minus_placeholder_token_length={m:quantiles(v) for m,v in lengths.items()},
        kl_fraction_above_1em5_by_position={k:dict(positions=len(v),fraction=float(np.mean(v))) for k,v in kl_nonzero.items()},
        candidate_centered_content_delta={k:quantiles(v) for k,v in delta.items()},
    )

def finish(run):
    dest=run/'formal_report'
    load=lambda name:json.loads((dest/name).read_text())
    evidence=load('acceptance.json')
    assert evidence['passed']
    results=load('preflight_utility_report.json')
    selection=load('packet_selection.json')
    summary=load('summary.json')
    alignment=load('score_alignment.json')
    funnel=json.loads((run/'cpu_replay_checks/funnel.json').read_text())
    assert sum(g['episodes'] for g in funnel.values())==1500
    assert sum(sum(g['steps'].values()) for g in funnel.values())==evidence['steps']
    (dest/'extra_diagnostics.json').write_text(json.dumps(extra_diagnostics(run),indent=2)+'\n')
    (dest/'funnel.json').write_text(json.dumps(funnel,indent=2)+'\n')
    files=['acceptance.json','input_hashes.json','summary.json','score_alignment.json','preflight_utility_report.json','preflight_content_coverage.json','privilege_delta_report.json','template_effect_report.json','packet_token_counts.json','extra_diagnostics.json','funnel.json','packet_selection.json']
    artifacts={name:digest(dest/name) for name in files}
    for name in ('formal_authorization.json','splits_hash.json','teacher_packet_manifest.json','parser_repair_manifest.json','preflight_cost_ledger.jsonl','env_manifest.json','PROTOCOL.md'):
        artifacts['../'+name]=digest(run/name)
    (dest/'report_manifest.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    winner=selection['selected']
    desc='选定'+NAMES[winner] if winner else '本轮未选出特权包'
    n=summary['tasks']
    lines=[
        '# VC-OPD v0.4 正式特权包预检（A.5）','',
        f"冻结的 {n} 个任务和 {n} 个不同目标商品已完成五组配对实验。一次性判定为 **{selection['status']}**，{desc}。共计 {summary['episodes']:,} 条完整轨迹、{summary['steps']:,} 个环境动作。各组在每题上采样两次，另做一次贪心生成。",'',
        '| 老师输入 | 采样全要求购买（/200） | 相对无包（百分点，95% CI） | 贪心全要求购买（/100） | 采样严格成功率 | 采样平均 Reward v3 |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in POLICIES:
        s=results[mode+'_sample']; g=results[mode+'_greedy']; lift=s['all_requirements_vs_none']; a,b=lift['ci95']
        lines.append(f"| {NAMES[mode]} | {round(200*s['all_requirements'])} | {100*lift['estimate']:+.1f} [{100*a:+.1f}, {100*b:+.1f}] | {round(100*g['all_requirements'])} | {s['strict']:.1%} | {s['reward']['mean']:.3f} |")
    lines+=['', '采样差值先在每题内平均，再对任务作 10,000 次配对 bootstrap。每题对应独立目标商品，按任务和按商品聚类在此相同。全要求购买同时计入 gold 和满足全部条件的替代商品。','',
            '| 真实包 | 可评 click/规格位置 | 同时超过内容阈值 | 有偏移任务 | 判定 |','|---|---:|---:|---:|---|']
    for m,x in selection['decisions'].items():
        lines.append(f"| {NAMES[m]} | {x['positions']:,} | {x['hits']:,} ({x['hits']/max(1,x['positions']):.1%}) | {x['shifted_tasks']}/{x['tasks']} | {x['status']} |")
    ref=results['full_sample']['all_requirements_vs_placeholder']
    lines+=['',f"完整包相对同格式占位文本的采样购买率差为 {ref['estimate']*100:+.1f} 个百分点（95% CI {ref['ci95'][0]*100:+.1f} 至 {ref['ci95'][1]*100:+.1f}）。商品包与当页证据包的同格式占位对照只用于 token 评分，行为层面的差值仍包含模板效应。",'',
            '行动内容判定使用学生 top-8 加实际采样 token 的条件分布，并比较同一输出前缀上的真实包与对应占位包。至少 10% 合格位置须同时达到采样 token 中心化差绝对值 >0.02 和内容 KL >1e-5，还须覆盖至少 10 题。行为门使用采样全要求购买率的配对区间下界 >0。','',
            f"无包学生的 {alignment['scored_steps']:,} 个决策和 {alignment['response_tokens']:,} 个响应 token 全部对齐。重复空包评分的最大误差为 {alignment['max_empty_error']:.3g}。保留了 8 题试跑中对裸动作解析的逐步修订记录。CPU 独立回放复核了全部 1,500 条轨迹的终止原因、终局奖励与逐步合法性。",'',
            f"本次追加使用 {evidence['formal_gpu_hours']:.3f} / 2.500 GPU 小时。含首次 8 题的总用量为 {summary['gpu_hours']:.3f} GPU 小时。实验停在 A.5；本报告未评估投影器效果或蒸馏后策略泛化。",'',
            '正式运行目录内的 report_manifest.json 记录报告和原始输入清单的 SHA256。完整 reward 类型分布、合法动作率、搜索/打开商品/规格/购买漏斗、KL 与候选质量的页面和 token 分类统计随附。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    finish(p.parse_args().run)
