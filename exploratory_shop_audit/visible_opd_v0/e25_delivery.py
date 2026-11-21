"""Write aggregate, reproducible stage-E artifacts and preserve all registered endpoints."""
from .e25_layout import Layout
from .e25_probe import RUN,PUBLIC
from .s22_report import tsv
from exploratory_shop_audit.grpo_forced_audit.common import read,read_rows,save,digest_file

def markdown(l,completion,behavior,costs):
    step=completion['last_step']
    endpoint=behavior[str(step)]
    lines=['# 阶段 E 结果','',
        '本次是单种子探索性实验。R、R-matched（如设立）及 P 均从同一初始快照独立训练。'
        '各步以当步学生为锚，沿修正方向反复推进。P 按冻结窗口和当前快照重评分并进行 ASIN 隔离的四折刷新。','',
        '| 组 | 终点更新 | 停止状态 | 学习率 |','|---|---:|---|---:|']
    for a,k in completion['endpoints'].items():
        lines.append(f"| {a} | {k} | {completion['stop_labels'][a]} | {l.config['rates'][a]:.8g} |")
    for a in completion.get('not_run',[]):
        lines.append(f"| {a} | not run | not run | {l.config['rates'][a]:.8g} |")
    if l.name=='formal_2512':
        gate=read(l.root/'controls_gate.json')
        lines+=['','采用 §25.12 的四引擎固定任务布局及新 B@0 基线，顺序为 P、R-matched、R。']
        if 'counts' in gate:
            c=gate['counts'];n=gate['episodes_per_arm']
            lines+=[f"T_opd_eval 采样全要求购买为 P {c['P']}/{n}、B {c['B']}/{n}，控制组决策 {gate['decision']}。"]
        else:lines+=['P 在第 100 步前按 §25.7 停止，依登记解释继续运行控制组。']
    lines+=['','## 主要行为比较','',
        '全要求购买率之差，按预登记主比较列出。每个池 200 题，采样每题 4 条，贪心每题 1 条。'
        '单位为百分点，括号内为配对商品簇 95% bootstrap 区间。','',
        '| 比较 | 解码 | T_opd_eval | V | V 未见商品 |','|---|---|---:|---:|---:|']
    for comparison in endpoint['primary']:
        for mode in ('sample','greedy'):
            cells=[]
            for pool in ('T_opd_eval','V','V_unseen'):
                key=f'{comparison}/{mode}/asin/{pool}'
                if key not in endpoint['paired']:
                    cells.append('not run' if any(comparison.endswith('-'+a) for a in completion.get('not_run',[])) else 'NA（空层）')
                    continue
                v=endpoint['paired'][key]['all_requirements']
                lo,hi=v['ci95'];cells.append(f"{100*v['estimate']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
            lines.append('| '+comparison+' | '+mode+' | '+' | '.join(cells)+' |')
    lines+=['','[逐检查点行为、熵及合法率](stage_e_behavior_by_checkpoint.tsv) · '
        '[完整终点的其他比较、任务簇和商品簇区间、训练域与验证域差距](stage_e_final_behavior.json)','',
        '## 移动量与训练监测','',
        '[逐更新 BF16 与 FP32 重锚分解](stage_e_steps.tsv) · '
        '[全词表移动量、类别熵、top-1 概率、跨精度 KL 和相邻投影器变化](stage_e_monitors.tsv) · '
        '[每类修正强度、覆盖和候选质量](stage_e_category_metrics.json)']
    matching=l.config.get('matching_2511')
    if matching:
        lines+=['',f"§25.11 第二次校准 ρ″ = {matching['rho_double_prime']:.5f}，状态为 {matching['label']}，"
            f"正式 R-matched 学习率冻结为 {matching['frozen_lr']:.9g}。"
            "正式比较仅纳入第二次校准后从第 0 步独立启动的 R-matched。"]
    check=l.root/'rematch_at_10.json'
    if check.exists():
        match=read(check)
        lines+=['',f"正式 E 第 10 步 sqrt(m_R-matched/m_P) = {match['ratio']:.4f}，"
            +('落在' if match['within_band'] else '落在区间以外，约定区间为')+' [0.8, 1.25]。此读数仅报告。']
    lines+=['','阶段 D 仍以原登记保留 R-β 与 P-shuffled 的结果。'
        'D@10 中，P 相对 P-shuffled 的采样全要求购买率差在 T_opd_eval 为 +6.62 [+4.25, +9.00] 个百分点，'
        '在 V 为 +7.12 [+5.00, +9.38] 个百分点。'
        '[阶段 D 原报告](../section23_20261120/STAGE_D.md)','',
        f"正式 E 已计入的四卡阶段用量为 {costs[l.name]:.2f} GPU 小时。"
        +('匹配校准另用 %.2f GPU 小时。'%costs['calibration'] if 'calibration' in costs else ''),
        '', '[成本预估与匹配依据](PREFLIGHT.md) · [终点快照签名与停止状态](stage_e_completion.json)']
    if 'shared_2511' in costs:
        lines+=['', f"§25.11 两次校准及正式 E 的共享四卡环节按整次计费。"
            f"第二次校准独占环节 {costs.get('calibration_2511',0.):.2f} GPU 小时，"
            f"正式 R/P 与第二次校准共同执行的环节 {costs['shared_2511']:.2f} GPU 小时。"]
    (PUBLIC/'STAGE_E.md').write_text('\n'.join(lines)+'\n')

def final_report(l=None):
    l=Layout('formal') if l is None else l
    completion=read(l.root/'complete.json')
    steps=[];monitors=[];staging={}
    for a,last in completion['endpoints'].items():
        for k in range(1,last+1):
            directory=l.folder(k,a);s=read(directory/'train_statistics.json')
            m=read(directory/'measurement.json');opt=read(directory/'optimizer.json')
            assert opt['before_sha']==s['before_sha'] and opt['after_sha']==s['after_sha']
            raw={f'{prec}_{field}':m[prec][field] for prec in ('bf16','fp32_reanchored') for field in ('kl_before','kl_after','delta_kl','alignment','movement')}
            steps.append(dict(arm=a,update=k,legal_rate=s['legal_rate'],records=s['records'],scored=s['scored'],
                 before_sha=s['before_sha'],after_sha=s['after_sha'],optimizer_steps=s['optimizer_steps'],
                 clip_count=s['clip_count'],projector_version=s['projection_version'],**raw))
            staging[f'{a}/{k}']=dict(classes=s['by_class'],candidate_mass=s.get('candidate_mass'))
            if k%10==0:
                v=read(l.root/'monitor'/f'{k:03}.json')[a]
                for group,x in v['stats'].items():
                    monitors.append(dict(arm=a,update=k,group=group,**x,projector_mse=v['projector_mse'],projector_cosine=v['projector_cosine']))
    tsv(PUBLIC/'stage_e_steps.tsv',steps);tsv(PUBLIC/'stage_e_monitors.tsv',monitors)
    save(PUBLIC/'stage_e_category_metrics.json',staging)
    behavior={str(k):read(l.root/'behavior'/f'{k:03}.json') for k in range(0,completion['last_step']+1,10)}
    save(PUBLIC/'stage_e_final_behavior.json',behavior[str(completion['last_step'])])
    # Keep all intermediate paired intervals locally. Publish the point estimates by checkpoint.
    table=[]
    for k,r in behavior.items():
        for key,x in r['summary'].items():
            if x['tasks']:
                a,pool,mode=key.split('/')
                table.append(dict(update=k,arm=a,checkpoint=r['endpoints'][a],pool=pool,mode=mode,
                    **{f:x[f] for f in ['tasks','episodes','reward','all_requirements','strict','decision_entropy','legal_rate','length']}))
    tsv(PUBLIC/'stage_e_behavior_by_checkpoint.tsv',table)
    actual=read_rows(RUN/'allocation.jsonl');costs={}
    for r in actual:costs[r['run']]=costs.get(r['run'],0.)+r['elapsed']*r['allocated_gpus']/3600
    check=l.root/'rematch_at_10.json'
    checkpoint_signatures={}
    for a,k in completion['endpoints'].items():
        sha=digest_file(l.state(k,a)/'adapter/adapter_model.safetensors')
        assert sha==read(l.folder(k,a)/'optimizer.json')['safetensors_sha256']
        checkpoint_signatures[a]=sha
    save(PUBLIC/'stage_e_completion.json',dict(completion=completion,checkpoint_sha256=checkpoint_signatures,actual_allocated_gpu_hours=costs,
        formal_settings=l.config,step10_matching=read(check) if check.exists() else None,
        offline_pilot_trigger=read(RUN/'offline/complete.json')['p_kl_pilot_after_E'],description='单种子探索性；沿修正方向反复推进。'))
    markdown(l,completion,behavior,costs)
    save(l.root/'local_delivery.json',dict(public_sha256={p.name:digest_file(p) for p in sorted(PUBLIC.iterdir()) if p.is_file()}))
