"""Render the post hoc ending diagnostic as a readable report and five-update CSV."""
import csv

from .action_ending_train import DEST
from exploratory_shop_audit.grpo_forced_audit.common import read


def metric(v, name):
    if not v['positions']:
        return None,None,None
    if name=='log':
        x = v['log_stop_ratio']
        if not x['n']:
            return None,None,None
        means,ci,i = x['mean'],x['ci95'],0
    else:
        means,ci,i = v['mean'],v['ci95'],name
    return (means[i],*(ci[i] if ci else (None,None)))


def formatted(v, name, percent=False):
    m,l,h = metric(v,name)
    if m is None:
        return '—'
    scale = 100 if percent else 1
    digits = 2 if percent else 4
    point = f'{scale*m:.{digits}f}'
    return point if l is None else f'{point} [{scale*l:.{digits}f}, {scale*h:.{digits}f}]'


def main():
    t = read(DEST/'section26_action_ending.json')
    b = read(DEST/'tcal_action_ending.json')
    labels = ('q_stop','actor_stop','mean_log_stop_ratio','fraction_q_stop_lt_actor_stop','q_continue','actor_continue')
    indices = (0,1,'log',2,3,4)
    with (DEST/'ending_windows.csv').open('w') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['group','first_update','last_update','reply_cohort','position','support_filter','n','tasks','n_log',
                         *[f'{col}_{x}' for col in labels for x in ('estimate','lower95','upper95')]])
        for w in t['windows']:
            for cohort,types in w['cohorts'].items():
                for role,stats in types.items():
                    for subset in ('all','both_candidates_present'):
                        v=stats if subset=='all' else stats.get(subset)
                        if v is not None:
                            writer.writerow([w['arm'],w['start'],w['end'],cohort,role,subset,v['positions'],v.get('tasks',0),
                                             v.get('log_stop_ratio',{}).get('n',0),
                                             *[value for index in indices for value in metric(v,index)]])
    lines=['# 第一条命令结束时，目标怎样改变收尾概率？','',
           '2026-11-24。§26 之后的描述性 CPU 诊断。训练部分使用每步存下的目标 q 和同一步更新前的 actor，两者在同一个候选集 C 上比较。','',
           '## §26 训练回复','',
           '每个实际采样位置等权。下表为全程汇总，方括号为按任务整簇重抽 10,000 次的 95% 区间。概率和位置比例均以 % 表示。log 比用自然对数，正值表示目标提高了该位置收尾候选的概率。','',
           '| 组 | 实际回复 | 位置 | 位置数 / 可取 log 数 | 平均 q(收尾) % | 平均 actor(收尾) % | 平均 log 比 | q < actor 占比 % |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for arm,s in t['overall'].items():
        for cohort,name in [('legal','合法'),('next_command','续写下一条命令')]:
            for role,label in [('bracket','首个命令右括号'),('following','右括号后的下一位置')]:
                v=s['cohorts'][cohort][role]
                cells=[arm,name,label,f"{v['positions']} / {v['log_stop_ratio']['n']}",formatted(v,0,True),
                       formatted(v,1,True),formatted(v,'log'),formatted(v,2,True)]
                lines.append('| '+' | '.join(cells)+' |')
    lines+=['','**实采右括号与换行合成一个 token，并实际换行续写另一条命令的回复**。下表直接在产生该 token 的位置比较两种写法。收尾和续写均取原 C 上的概率质量。','',
            '| 组 | 位置数 / 可取 log 数 | q(无换行闭括号) % | actor(无换行闭括号) % | 平均 log 比 | q < actor 占比 % | q(闭括号加换行) % | actor(闭括号加换行) % |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in t['overall'].items():
        v=s['cohorts']['newline_next_command']['bracket_sampled_fused_newline']
        cells=[arm,f"{v['positions']} / {v['log_stop_ratio']['n']}",formatted(v,0,True),
               formatted(v,1,True),formatted(v,'log'),formatted(v,2,True),formatted(v,3,True),formatted(v,4,True)]
        lines.append('| '+' | '.join(cells)+' |')
    coverage=[]
    for arm,s in t['overall'].items():
        v=s['cohorts']['newline_next_command']['after_fused_newline']
        coverage.append(f"{arm} 为 {v['log_stop_ratio']['n']}/{v['positions']}")
    lines+=['','在换行已经生成后的下一位置，EOS 仍在候选集中的覆盖数为 '+'，'.join(coverage)+'。此时应优先读上表的右括号分叉处；后一步的 log 比只覆盖仍有 EOS 的位置。']
    lines+=['','| 组 | 执行回复 | 找到命令右括号 | 无命令 / 缺匹配右括号 | 续写另一条命令（换行 / 同行） | 其中环境判合法 | 有位置但无目标的位置数 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for arm,s in t['overall'].items():
        c=s['coverage']
        lines.append(f"| {arm} | {c['replies']} | {c['located']} | {c.get('no_command',0)} / {c.get('unclosed_first_command',0)} | {c.get('next_command_newline',0)} / {c.get('next_command_same_line',0)} | {c['legal_and_next_command']} | {c['structural_na_positions']} |")
    lines+=['','每组每个五步窗均按同一口径给出四项指标与区间。另外列出明确换行续写的子集、两类候选同时可见的子集，以及右括号是否已携带换行。[五步窗口 CSV](ending_windows.csv) · [完整训练统计及清单哈希](section26_action_ending.json)','',
            '## 阶段 B 的 T_cal','',
            '以任务、动作步、回复 token 位置和候选 ID 对齐两个存档。两个标签按无包学生概率在原 C 上中心化。旧 Huber 使用旧内容标签的 ASIN 折外预测 × b_c，首块 KL 使用新完整包目标的 ASIN 折外预测 × λ_c，系数均取该位置的冻结类别。表内每项依次为候选均值和严格正号比例。此前的 tcal_closing_bracket.json 选取搜索参数中有右括号候选的位置；本表只取实际生成到首条命令结尾的位置。','',
            '| 位置 | 共同候选数 | 完整包 − 无包 | 完整包 − 占位包 | 旧 Huber × b_c | 首块 KL × λ_c |',
            '|---|---:|---:|---:|---:|---:|']
    for label,x in [('两处合计',b['overall']),('右括号',b['positions'][0]),('下一位置',b['positions'][1]),
                    ('右括号中实采换行（括号位置子集）',next(r for r in b['positions'] if r['position']=='bracket_sampled_fused_newline'))]:
        values=[f'{v:+.5f} / {100*p:.1f}%' for v,p in zip(x['mean'],x['positive_share'])]
        lines.append('| '+' | '.join([label,str(x['n']),*values])+' |')
    lines+=['',f"原 512 题共 {b['replies']:,} 条回复、34,825 个评分位置。{b['overall']['n']:,} 个收尾候选可以逐 ID 比较，其中 {b['identical_support_only']['n']:,} 个所在行的完整候选集也相同。只在旧评分或首块评分出现的收尾候选分别为 {b['old_only_candidates']}、{b['new_only_candidates']} 个。[均值、中位数、正号比例、区间和同支持集对照](tcal_action_ending.json)",'',
            '## 位置与分母','',
            '- 使用原始回复 token 的字节，从第一条形如 verb[arg] 的命令找到配对的 ASCII 右括号，处理参数内嵌套的方括号。若没有匹配右括号则单列，若下一位置没有采样 token 则只计括号位置。续写组要求后面出现带左括号的新命令，允许换行、字面量反斜杠加 n、同行或后续命令行上的 Action:。合法性取执行时的环境口径，报告交集。','',
            '- 右括号位置保持当前 token 中右括号前的参数字节完全相同。闭括号后无换行（允许水平空白）的候选归“收尾”，闭括号后带换行的归“续写”。例如 token 60 为单独的右括号，921 为右括号加换行，2533 为右括号加两个换行。单独右括号仍可能在下一步接换行，因此此处只测闭括号写法。','',
            '- 在右括号后的下一位置，收尾候选是生成配置的 EOS（151645、151643）。若上一 token 已含换行，下一位置仍测 EOS；续写侧辅助读数另计换行或可能开始新命令的 token。CSV 同时给出 bracket、following 和按已采样括号后缀拆开的行，名称以 after_ 开头的行均为 following 的子集，bracket_sampled_ 开头的行是括号位置按实采后缀的子集。','',
            '- 概率是原掩码 C 上该类所有候选的概率之和，保留其他候选的质量。收尾不在 C 时两侧均为零，计入两项概率均值和严格小于的比例；此时 log 比无定义，只在正质量位置报告并给出分母。区间保留同任务内的重复轨迹及位置。比较的是训练时的候选条件概率；按已经生成的回复分组，只能作描述。','',
            '复现时在原 torch 环境中设置 CUDA_VISIBLE_DEVICES 为空，运行 exploratory_shop_audit.visible_opd_v0 下的 action_ending_train、action_ending_tcal、action_ending_report 三个 Python 模块。输入来自提交 84388d9e 的存档体系，JSON 记录输入校验值及重抽样种子。']
    fused=[t['overall'][a]['cohorts']['newline_next_command']['bracket_sampled_fused_newline'] for a in ('R-gap','P-gap')]
    lines[3:3]=['',
        '在实际生成“闭括号加换行”并续写下一条命令的回复上，目标相对当步 actor 的无换行闭括号平均 log 比，'
        +f"R-gap 为 **{formatted(fused[0],'log')}**，P-gap 为 **{formatted(fused[1],'log')}**。"
        +'这项测量按已生成的轨迹分组，单凭它无法确定重复命令增多的原因。']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
