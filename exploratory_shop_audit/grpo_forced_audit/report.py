"""CPU-only final report; missing stages never become completed experiments."""
from collections import defaultdict
from .common import *
from .analysis import evaluate_saved,interval
from .checks import environment
from .budget import spent,uncapped


def platform(run):
    points=sorted((run/'evaluation').glob('checkpoint_*_sample.json'))
    if len(points)<3:return False
    last=points[-3:]
    steps=[int(p.stem.split('_')[1]) for p in last]
    if steps[-1]<30 or steps[1]-steps[0]!=10 or steps[2]-steps[1]!=10:return False
    rows=[read(p)['hard']['metrics'] for p in last]
    for key,limit in [('reward',.03),('strict_success',.05),('L_click_plus_spec',.05)]:
        values=[r[key]['estimate'] for r in rows]
        if any(v is None for v in values) or max(values)-min(values)>limit:return False
    coverage=[r['funnel_coverage']['estimate'] for r in rows]
    return min(coverage)>=coverage[0]


def decision_d(summary,healthy,plateau):
    if not healthy:return '实现有效性未通过，不作方法结论'
    if not plateau:return '预算内尚未稳定'
    ci=summary['hard']['metrics']['L_click_plus_spec']['interval']
    if ci is None:return '证据不足：有效漏斗不可定义'
    lo,hi=ci
    if hi<.10:return '条件点击／规格残差已小；仍须检查总体表现、停手损失和覆盖变化'
    if lo>.20:return '点击／规格仍有明确残差'
    if lo>.10 and hi<.20:return '中间地带'
    return '证据不足：区间跨预定阈值'


def decision_e(metric):
    ci=metric['interval']
    if ci is None:return '证据不足'
    if ci[0]>.05:return '当前候选集合存在明确实际收益'
    if ci[1]<.05:return '未达到本轮最小实用收益线'
    return '证据不足：区间覆盖0.05'


def paired(base,final,field):
    def grouped(records):
        out=defaultdict(list)
        for r in records:
            value=r.get(field)
            if field=='L_stop':value=r['funnel'][2] if r['funnel'] is not None else None
            if field=='coverage':value=r['funnel'] is not None
            if value is not None:out[r['task_id']].append(float(value))
        return {k:sum(v)/len(v) for k,v in out.items()}
    a,b=grouped(base),grouped(final)
    return interval([b[k]-a[k] for k in a.keys()&b.keys()],'paired:'+field,.975)


def final_report(run):
    rpc=environment(run,'final_analysis')
    summaries={}
    try:
        for label in ['base','final']:
            for decode in ['greedy','sample']:
                folder=run/'evaluation'/label/decode
                if not (folder/'episodes.jsonl').exists():continue
                summary=evaluate_saved(run,read_rows(folder/'episodes.jsonl'),read_rows(folder/'steps.jsonl'),lambda i,a:rpc.call(cmd='maximum',goal_index=i,asin=a),label+decode)
                save(run/'evaluation'/f'final_{label}_{decode}.json',summary);summaries[label+'_'+decode]=summary
    finally:rpc.close()
    health=read_rows(run/'weight_updates.jsonl'); restored=(run/'training/restore_rank_0.json').exists() and read(run/'training/restore_rank_0.json').get('passed') is True
    advantages=read_rows(run/'advantage_checks.jsonl')
    healthy=bool(health) and all(r['changed'] and r['finite'] for r in health) and restored and any(r['finite'] and r['nonzero_fraction']>0 for r in advantages) and read(run/'preflight_report.json')['passed']
    expected_tasks={t['task_id'] for t in read(run/'splits.json')['V']}
    repeats=read(run/'budget_freeze.json')['final_repeats']
    complete=True
    for key in ['base_sample','final_sample','base_greedy','final_greedy']:
        records=summaries.get(key,{}).get('all_unweighted',{}).get('records',[])
        counts=defaultdict(int)
        for row in records:counts[row['task_id']]+=1
        complete &= set(counts)==expected_tasks and all(n==(repeats if key.endswith('sample') else 1) for n in counts.values())
    result={'completed_final_evaluation':bool(complete),
        'gpu_hours':spent(run),'training_healthy':healthy,'checkpoint_restore_verified':restored,'platform':platform(run),
        'evaluation':summaries,'training_updates':max([r['update'] for r in read_rows(run/'training_metrics.jsonl')],default=0),
        'source':read(run/'registration.json'),'cleaning':read(run/'cleaning_summary.json')}
    recovery=run/'recovery_0042/continuation_authorization.json'
    if recovery.exists():
        result['checkpoint_recovery']=read(recovery)
    if 'final_sample' in summaries:
        result['D_decision']=decision_d(summaries['final_sample'],healthy,result['platform'])
    if 'base_sample' in summaries and 'final_sample' in summaries:
        base=summaries['base_sample']['hard']['records'];final=summaries['final_sample']['hard']['records']
        changes={k:paired(base,final,k) for k in ['reward','strict_success','purchase','coverage','L_stop']};result['paired_hard_changes']=changes
        adverse=any(changes[k]['interval'] and changes[k]['interval'][1]<0 for k in ['strict_success','purchase','coverage']) or bool(changes['L_stop']['interval'] and changes['L_stop']['interval'][0]>0)
        result['adverse_other_metrics']=adverse
        if adverse and result.get('D_decision','').startswith('条件点击'):result['D_decision']='条件点击／规格损失降低，但其他指标恶化；不宣称配置已解决'
    result['audits']={}
    for label in ['audit_base','audit_post']:
        if (run/label/'metrics.json').exists():
            m=read(run/label/'metrics.json');result['audits'][label]={'metrics':m,'decisions':{p:decision_e(v['gain']) for p,v in m['pages'].items()}}
    result['training_curve']={p.stem:read(p) for p in sorted((run/'evaluation').glob('checkpoint_*.json'))}
    result['historical']=read(run/'historical_reanalysis/metrics.json')
    result['screening']=read(run/'screening_summary.json')
    result['preflight']=read(run/'preflight_report.json')
    result['pilot']=read(run/'pilot_summary.json')
    result['next_step']='本轮尚未完成，不作下一轮方法建议'
    if complete:
        d=result.get('D_decision','')
        e=result['audits'].get('audit_post',{}).get('decisions',{})
        if d.startswith('条件点击') and not result.get('adverse_other_metrics'):
            result['next_step']='当前配置点击/规格的方法余量小；更难配置须另行登记，本轮不切换'
        elif d=='点击／规格仍有明确残差' and any(v.startswith('当前候选') for v in e.values()):
            result['next_step']='建议下一轮对达线环节进行方法对照；本轮不自动启动'
        elif d=='点击／规格仍有明确残差' and e and all(v.startswith('未达到') for v in e.values()):
            result['next_step']='暂不开发当前单步强制分叉方法；仅限本轮取点、候选和续跑策略'
        else:result['next_step']='记录中等残差、未稳定或证据不足；不自动扩轮'
    save(run/'metrics.json',result)
    lines=['# 购物 GRPO 与强制臂审计', '',f'实际更新数：{result["training_updates"]}；累计GPU小时：{result["gpu_hours"]:.4f}；预算：'+('研究者已解除GPU小时上限。' if uncapped(run) else '24 GPU小时。'),
           f'训练有效性：{healthy}；断点恢复：{restored}；操作性平台：{result["platform"]}。',
           f'清洗保留：{result["cleaning"]["retained"]}/{result["cleaning"]["train"]}。',
           'D判定：'+result.get('D_decision','尚未完成终点评测，不作判定'),'']
    for label,data in result['audits'].items():
        for page,m in data['metrics']['pages'].items():lines.append(f'{label}/{page}：A选B评 {m["gain"]}；rho3 {m["reliability"]["rho3"]}；{data["decisions"][page]}。')
    historical=result['historical'];hm=historical['funnel']['metrics']
    lines+=['',f'C历史复算：{historical["tasks"]}题、{historical["episodes"]}条轨迹；有效漏斗{historical["funnel"]["funnel_episodes"]}条。',
        '四段损失（任务等权）：'+str({k:hm[k] for k in ['L_search','L_click','L_stop','L_spec']}),'',
        '| 更新 | 难题奖励 | 严格成功 | 搜索损失 | 点击损失 | 规格损失 | 漏斗覆盖 |',
        '| - | - | - | - | - | - | - |']
    for key,summary in result['training_curve'].items():
        if not key.endswith('_sample'):continue
        step=int(key.split('_')[1]);m=summary['hard']['metrics']
        values=[m[k]['estimate'] for k in ['reward','strict_success','L_search','L_click','L_spec','funnel_coverage']]
        lines.append('| '+str(step)+' | '+' | '.join('NA' if x is None else f'{x:.4f}' for x in values)+' |')
    lines+=['','预测“搜索损失先降、点击与规格后降”按上述完整曲线描述；不从单一终点推断先后顺序。',
        '下一步：'+result['next_step'],'','旧800条仅按通过清洗的任务复算，见historical_reanalysis；不是新接口基线。完整逐题、漏斗覆盖、总体表现和配对变化见metrics.json。',
            '信度高不构成停止回放的理由；基础E0不能替代训练后E1。尚未找到准确的ALFWorld学生可见份额实现，不以文本代理替代。',
            '本轮到此停止，不自动启动新信用方法、SFT、S/F、换环境或扩预算。']
    if recovery.exists():
        lines+=['','恢复限制：原41/42更新未保存完整状态，精确重建哈希未通过。研究者明确授权从checkpoint40续训；原记录保存在recovery_0042/pre_continuation_archive，当前日志记录恢复后轨迹。41–43复用原缓存，42/43的生成策略属于故障前谱系；保留原行为概率，不宣称逐位同轨或严格on-policy恢复。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result
