"""Render measured pre-E controls and explicit assumptions behind the allocation estimate."""
from .e25_probe import PUBLIC,RUN,ARMS,GROUPS
from exploratory_shop_audit.grpo_forced_audit.common import read

def render():
    m=read(RUN/'movement.json');cost=read(RUN/'gpu_budget.json');d=read(RUN/'lora_deadzone.json')
    names=(*ARMS,'null')
    lines=['# 阶段 E 启动前核验','',
           f"使用阶段 D 第 1 步的 32 个任务、64 条 B 轨迹、282 条动作记录与全部 4,448 个有效位置。"
           f"全词表移动量的 BF16 比值 ρ = {m['rho']:.6f}。匹配带为 [0.8, 1.25]。",'',
           '| 组 | BF16 KL | FP32 KL |','|---|---:|---:|']
    for a in names:
        lines.append(f"| {a} | {m['scores']['bf16'][a]['all']['kl']:.9g} | {m['scores']['fp32'][a]['all']['kl']:.9g} |")
    if m['needs_matched']:
        lines+=['',f"增设 R-matched，十步校准的起始学习率为 {m['initial_matched_lr']:.9g}。"]
    lines+=['','[两种精度按六类位置的移动量、质量和位置数](movement.json)','',
        '## BF16 LoRA 舍入','',
        '使用同一批原生前缀的动作预测位置，按原任务及轨迹位置权重汇总全部 LoRA 层。表中失活率为原增量非零的元素中，实际增量为零的加权比例。','',
        '| 层类型 | R 范数比 | P 范数比 | R 余弦 | P 余弦 | R 失活率 | P 失活率 |','|---|---:|---:|---:|---:|---:|---:|']
    for typ,v in d['R'].items():
        p=d['P'][typ];lines.append(f"| {typ} | {v['norm_ratio']:.4f} | {p['norm_ratio']:.4f} | {v['cosine']:.4f} | "
            f"{p['cosine']:.4f} | {v['zero_given_nonzero_weighted']:.2%} | {p['zero_given_nonzero_weighted']:.2%} |")
    b=cost['benchmark']
    lines+=['','[层数、原始元素计数与实际前向核对](lora_deadzone.json)','',
        '## 同快照在线刷新与预算','',
        f"以 D 第 10 步 P 的同一快照重算最近十步的 {b['tasks']} 个历史任务、{b['records']} 条动作记录，"
        f"得到 {b['positions']:,} 个拟合位置。四卡重算学生表示及真实包/匹配占位标签用时 {b['seconds_score']} 秒，"
        f"四个按 ASIN 隔离的 rank-8 Huber 拟合用时 {b['seconds_fit']} 秒，合计 {b['allocated_gpu_hours']:.2f} GPU 小时。",'',
        f"冻结窗口为最近 {cost['window_updates']} 次已完成更新。线性外推的累计刷新成本为 {cost['selected_refresh_gpu_h']:.1f} GPU 小时，"
        f"全历史方案为 {cost['all_history_refresh_gpu_h']:.1f} GPU 小时。",'',
        f"正式 E 预计 {cost['E_gpu_h']:.1f} GPU 小时，另计匹配校准 {cost['calibration_gpu_h']:.1f} GPU 小时。"
        f"四卡持续占用且各组跑满时合计约 {cost['wall_hours_if_four_reserved']:.1f} 小时。"
        '估算包含阶段内空闲卡、逐步 FP32 测量、训练、检查点全词表监测及行为评测，依赖 D 的吞吐和实际轨迹长度。','',
        '[分项预算、刷新实测与签名](gpu_budget.json)']
    if (RUN/'offline/complete.json').exists():
        o=read(RUN/'offline/complete.json');v=o['global_result'];ci=v['ci95']
        lines+=['','## 离线候选','',
            f"在 512 个 T_cal 任务的外层留出预测上，Huber + b_c 的加权前向 KL 为 {v['huber']:.6f}，"
            f"分布空间投影器为 {v['distribution']:.6f}，改善 {v['gain']:+.6f}，任务簇 95% 区间 [{ci[0]:+.6f}, {ci[1]:+.6f}]。"
            +('达到 P-KL 试点条件。' if o['p_kl_pilot_after_E'] else '未触发 P-KL 试点。'),'',
            '[分类结果、外层训练内选定的收缩系数](distribution_projector.json) · [当前预测器的题外可预测份额，按动词与语法拆分](format_predictable_share.json)']
    (PUBLIC/'PREFLIGHT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':render()
