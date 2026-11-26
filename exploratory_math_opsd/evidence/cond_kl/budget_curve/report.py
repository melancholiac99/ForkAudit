import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
x=json.loads((here/'results.json').read_text())
ks=[v//1024 for v in x['budgets']]
lines=['# MATH-500 回答中的结束标签与预算曲线','',
       '非思考独立评测为 4k 每题四条、16k 每题两条。按生成文本里是否出现字面量 `</think>` 分组，长度为归档生成 token 数。空分层记 NA。','','|预算|模型|出现率|含标签 条数 / 准确率 / 平均长度|无标签 条数 / 准确率 / 平均长度|','|---|---|---:|---:|---:|']
def group(v):
    if not v['count']:return '0 / NA / NA'
    return f"{v['count']} / {v['accuracy_pct']:.2f}% / {v['tokens']:.1f}"
def marker(b,a,v):
    if v is None:return f'|{b}|{a}|缺独立归档|—|—|'
    return f"|{b}|{a}|{v['prevalence_pct']:.2f}%|{group(v['present'])}|{group(v['absent'])}|"
for key,v in x['marker'].items():
    arm,budget=key.split('/')
    lines.append(marker(budget,arm,v))
lines+=['', 'R-cond 的 4k 替代读数取自 16k 原回答重建的前缀，每题两条。','',
        '|来源|模型|出现率|含标签 条数 / 准确率 / 平均长度|无标签 条数 / 准确率 / 平均长度|','|---|---|---:|---:|---:|',
        marker('16k→4k','R-cond@100',x['R-cond_4k_reconstructed_from_16k']),
        '','## 归档前缀的准确率（%）','','![Accuracy vs token budget](curve.png)','','|模型|'+'|'.join(f'{v}k' for v in ks)+'|','|---|'+'---:|'*len(ks)]
fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
for arm,values in x['curve'].items():
    lines.append('|'+arm+'|'+'|'.join(f'{v:.1f}' for v in values)+'|')
    ax.plot(ks,values,marker='o',label=arm,linewidth=2.8 if 'boundary' in arm else 1.5)
ax.set(xlabel='Generated-token budget (1k = 1,024)',ylabel='MATH-500 accuracy (%)',xticks=ks)
ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8,ncol=2);fig.savefig(here/'curve.png',dpi=170);plt.close(fig)
def fmt(v):return f'{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]'
lines+=['','预算以内保留整条回答；超预算的归档文本用原分词器编码、严格取前缀、解码后执行官方末个 boxed 提取与判分。没有完整末框按官方规则记错。原生 token ID 未归档。','',
        '## 边界组减对照（百分点，题级配对 95% 区间）','','|预算|B 非思考|B 思考|R|','|---|---:|---:|---:|']
for b,v in x['paired'].items():lines.append(f"|{int(b)//1024}k|"+'|'.join(fmt(v[a]) for a in ('B','B thinking','R@100'))+'|')
lines+=['','## λ=0 减对照（百分点，题级配对 95% 区间）','','|预算|R|边界组|B 思考|','|---|---:|---:|---:|']
for b,v in x['lambda0_paired'].items():
    lines.append(f"|{int(b)//1024}k|"+'|'.join(fmt(v[a]) for a in ('R@100','R-clip-boundary@100','B thinking'))+'|')
lines+=['','## 独立 4k 核验','','|模型|16k→4k|实跑 4k|重建减实跑（百分点）|','|---|---:|---:|---:|']
for arm,v in x['validation'].items():lines.append(f"|{arm}|{v['reconstructed_4k_pct']:.2f}%|{v['actual_4k_pct']:.2f}%|{fmt(v['reconstructed_minus_actual_pp'])}|")
lines+=['','区间按相同的 500 道题配对重抽 10,000 次，种子 20261124。4k 独立评测与 16k 归档每题采样次数不同，核验使用各自在题内的均值。']
(here/'REPORT.md').write_text('\n'.join(lines)+'\n')
