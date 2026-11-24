"""Write human-readable diagnostic tables from checked machine reports."""
import argparse
import csv

from .gap_cpu_actions import TYPES
from .first_command_entropy import DEST, MODELS
from exploratory_shop_audit.grpo_forced_audit.common import read


def table(lines,cols,rows):
    lines.extend(['','| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|'])
    lines.extend('| '+' | '.join(map(str,r))+' |' for r in rows)


def estimate(x,scale=1,digits=4):
    m=x['mean']*scale
    l,h=(v*scale for v in x['ci95'])
    return f'{m:.{digits}f} [{l:.{digits}f}, {h:.{digits}f}]'


def entropy():
    d=read(DEST/'tcal_entropy.json')
    groups=d['primary_stage_B_support']
    names=['无包学生','完整包老师','首块 KL × λ','旧 Huber × b']
    lines=['# T_cal 候选分布的熵和最高概率','',
           '原 512 个任务、34,825 个评分位置。每行使用同一候选集比较四个分布。全体位置中，首块 KL 投影的平均熵高于无包学生；差异最大的位置之一是首条命令的右括号。同一位置的旧 Huber 投影提高了最高概率。']
    table(lines,['类别','位置数',*names],[[key,x['n'],*[estimate(x['entropy_nats'][m]) for m in MODELS]] for key,x in groups.items()])
    lines.extend(['','上表单位为 nats。下表为各目标的最高候选概率减同位置学生最高概率，单位为百分点；每个分布各自取最大值。两表的方括号均为按任务整簇重抽 10,000 次的 95% 区间。'])
    table(lines,['类别','位置数',*names[1:]],[[key,x['n'],*[estimate(x['delta_top1'][m],100,3) for m in MODELS[1:]]] for key,x in groups.items()])
    lines+=['','bracket 和 following 分别是第一条命令右括号所在 token 和紧接着的下一 token。两行使用全部可定位位置，包括候选集没有 EOS 的行，且与冻结的词法类别重叠。各位置等权。','',
            '主表沿用阶段 B 原 C 和最后一层无包学生表示。完整包老师由存档 s 加居中的 full−none 还原；Huber 使用原折外输出；首块 KL 在同一 C 上用该商品对应的留出折冻结权重计算。两种投影均使用各自固定的类别收缩。另在首块原 C 上对称重算四个分布，两个存档有 '+f"{d['same_candidate_sets']:,}"+' 个位置的完整候选集相同。每类的成对熵差、原始 top-1 和第二套支持集的结果见[完整数据](tcal_entropy.json)。']
    (DEST/'T_CAL_ENTROPY.md').write_text('\n'.join(lines)+'\n')


def behavior():
    d=read(DEST/'behavior.json')
    cells=d['cells']
    lines=['# 连写时仅执行第一条完整命令的行为回放','',
           '同一题配对 §26 原值，自由交互沿用 T_opd_eval、V 各 200 题，采样种子 2201–2204，每题另做一次贪心。公共结果页恢复原 B 贪心首搜的完整历史与原环境状态，并逐题校验。已归档的起始前缀不计入新执行动作分母。','',
           '每个单元在题内先平均，再按 ASIN 整簇成对重抽 10,000 次。表中购买率以 % 表示，差值为新减原的百分点，方括号为 95% 区间。']
    for mode,label in [('free','自由交互'),('page','公共结果页续跑')]:
        lines+=['','## '+label]
        rows=[]
        for x in cells:
            if x['interaction']!=mode:continue
            l,h=x['paired_ci95']
            cut=f"{x['truncated_legal']}/{x['truncations']}" if x['truncations'] else '0/0'
            rows.append([x['model'],x['dataset'],'采样' if x['decode']=='sample' else '贪心',x['tasks'],
                         f"{100*x['original']:.2f}",f"{100*x['replay']:.2f}",
                         f"{100*x['paired_difference']:+.2f} [{100*l:+.2f}, {100*h:+.2f}]",
                         f"{100*x['legal_rate']:.2f}",cut])
        table(lines,['模型','数据集','解码','题数','原值 %','新值 %','差值 [区间]','执行动作合法率 %','截断后合法 / 截断数'],rows)
    lines+=['','每条回复按 §26 环境交互。对于从命令行或 Action: 开始、第一条配对括号已闭合且后面有另一条完整命令的回复，仅将第一条送入环境。同行连写和换行均计入，命令参数内部允许嵌套方括号。单条回复或只有未完成后缀时沿用原解析。完整生成文本留在模型历史中，下一观察来自环境实际执行的结果。非法首条照常被环境拒绝。','',
            '下表的非法计数为改动后全部新执行动作中的计数。逐格的总合法率、截断命令合法率、首次截断之后其余动作合法率及缺席编号的 gold/部分匹配交叉项见[行为数据](behavior.json)。']
    table(lines,['模型','交互','数据集','解码','无法解析','动作类型不可用','缺席编号','不可用选项','导航不存在','其他'],
          [[x['model'],x['interaction'],x['dataset'],x['decode'],*[x['illegal_types'][k] for k in TYPES]] for x in cells])
    lines+=['','[逐题配对计数（CSV 压缩包）](behavior_task_pairs.csv.gz) · [候选分布熵与最高概率](T_CAL_ENTROPY.md)']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
    keys=['model','interaction','dataset','decode','tasks','asins','episodes','original','replay','paired_difference',
          'ci95_low','ci95_high','actions','legal','legal_rate','truncations','truncated_legal','truncated_legal_rate',
          'subsequent_actions','subsequent_legal','subsequent_legal_rate',*TYPES]
    with (DEST/'behavior_cells.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n',extrasaction='ignore')
        writer.writeheader()
        for x in cells:
            writer.writerow(dict(**x,**x['illegal_types'],ci95_low=x['paired_ci95'][0],ci95_high=x['paired_ci95'][1]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('report',choices=['entropy','behavior'])
    if p.parse_args().report=='entropy':entropy()
    else:behavior()
