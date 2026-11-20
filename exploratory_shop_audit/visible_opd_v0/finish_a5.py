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
    exposures=defaultdict(Counter)
    for p in [*run.glob('smoke_batch_*/score_inputs.jsonl'), *run.glob('formal_batch_*/score_inputs.jsonl')]:
        for r in rows(p):
            for m,view in r['views'].items():
                exposures[m][view['reason'] or 'available']+=1
                cap=1024 if m in ('evidence','placeholder_evidence') else 192
                if view['reason'] is None and view['length']>cap:raise ValueError('packet exceeded applicable cap')
                if m in ('evidence','placeholder_evidence') and r['page']!='search_results' and view['reason']!='PACKET_UNAVAILABLE':raise ValueError('off-page evidence')
            for m,ref in MATCH.items():
                real,empty=r['views'][m],r['views'][ref]
                if real['reason'] or empty['reason']:continue
                lengths[m].append(real['length']-empty['length'])

    repaired={(r['goal'],r['rep'],r['step']):r['classes'] for r in rows(run/'parser_repair.jsonl')} if (run/'parser_repair.jsonl').exists() else {}
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
        student_packet_exposures={m:dict(counts=x,total=sum(x.values()),overflow_rate=x['PACKET_OVERFLOW']/sum(x.values()),overflow_given_available_page=x['PACKET_OVERFLOW']/max(1,sum(x.values())-x['PACKET_UNAVAILABLE'])) for m,x in exposures.items()},
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
    diagnostics=extra_diagnostics(run)
    (dest/'extra_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (dest/'funnel.json').write_text(json.dumps(funnel,indent=2)+'\n')
    files=['acceptance.json','input_hashes.json','summary.json','score_alignment.json','preflight_utility_report.json','preflight_content_coverage.json','privilege_delta_report.json','template_effect_report.json','packet_token_counts.json','extra_diagnostics.json','funnel.json','packet_selection.json']
    artifacts={name:digest(dest/name) for name in files}
    for name in ('formal_authorization.json','splits_hash.json','teacher_packet_manifest.json','token_parser_manifest.json','preflight_cost_ledger.jsonl','env_manifest.json','PROTOCOL.md'):
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
            '行动内容判定使用学生 top-8 加实际采样 token 的条件分布，并比较同一输出前缀上的真实包与对应占位包。至少 10% 合格位置须同时达到采样 token 中心化差绝对值 >0.02 和内容 KL >1e-5，还须覆盖至少 10 题。§16 行为门要求采样全要求购买率提升至少 10 个百分点且配对区间下界 >0。','',
            f"无包学生的 {alignment['scored_steps']:,} 个决策和 {alignment['response_tokens']:,} 个响应 token 全部对齐。重复空包评分的最大误差为 {alignment['max_empty_error']:.3g}。本次按 §16 模板重跑全部 100 题。CPU 独立回放复核了全部 1,500 条轨迹的终止原因、终局奖励与逐步合法性。",'',
            f"本次追加使用 {evidence['formal_gpu_hours']:.3f} / 2.500 GPU 小时。其中新 §16 运行（全部 100 题）用量为 {summary['gpu_hours']:.3f} GPU 小时，旧版首次预检 0.190 GPU 小时在追加额度之外。实验停在 A.5；本报告未评估投影器效果或蒸馏后策略泛化。",'',
            '正式运行目录内的 report_manifest.json 记录报告和原始输入清单的 SHA256。完整 reward 类型分布、合法动作率、搜索/打开商品/规格/购买漏斗、KL 与候选质量的页面和 token 分类统计随附。']
    e=diagnostics['student_packet_exposures']['evidence']
    lines+=['', f"在无包学生的真实状态上，证据包的 1024-token 上限溢出 {e['counts'].get('PACKET_OVERFLOW',0)}/{e['total']} 步（{e['overflow_rate']:.2%}）；在有对应页面资料的决策步中溢出率为 {e['overflow_given_available_page']:.2%}。非结果页标 NA。各真实包与占位包的溢出计数、分母及成对长度差见 extra_diagnostics.json。",
            '', '本次修订在旧 8 题预检披露及第一轮正式采集启动后登记。新版所有条件按同一冻结任务和种子重跑，旧样本未合入本表。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    finish(p.parse_args().run)
