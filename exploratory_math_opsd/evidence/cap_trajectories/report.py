"""Aggregate archived measurements with the trace-specific review ledger."""
import csv
import json
from pathlib import Path
from collections import Counter

D=Path(__file__).resolve().parent
def read(name):
    return json.loads((D/name).read_text())
rows=read('capped.json')
lens=read('lengths.json')
reviews=read('review_evidence.json')
notes={tuple([ds,m,int(q),int(j)]):(labs.split(),note) for ds,m,q,j,labs,note in csv.reader((D/'readings.tsv').open(),delimiter='\t')}
assert len(notes)==len(reviews)==39
def pct(a,b):
    return f'{100*a/b:.1f}%'
def table(headers):
    return ['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
def line(*a):
    return '| '+' | '.join(map(str,a))+' |'

s=['# R 与 λ=1.5 的长度触顶轨迹审计','',
'事后探索。四份归档的 SHA-256 均与评测回执一致。分母为回答，AIME24 为 30 题 × 12，MATH-500 为 500 题 × 2。样本之间共享题目。仅使用已保存的文本、官方判分函数与原分词器。','',
'## 全量计数与预算','',*table(['模式 / 模型','触顶 / 全部','触顶 token / 全部预算','触顶长度范围','未触顶 min / Q1 / 中位 / Q3 / P95 / max','触顶最终答对 / 有答但错 / 缺答','有 explicit 正解','正解后 token 合计'])]
for key,x in lens.items():
    ds,m=key.split()
    rr=[r for r in rows if (r['ds'],r['arm'])==(ds,m)]
    c=x['groups']['length'];u=x['groups']['stop'];n=len(rr);right=sum(r['correct'] for r in rr);missing=sum(r['missing'] for r in rr)
    hits=[r['first_correct'] for r in rr if r['first_correct']]
    s.append(line(key,f"{n}/{n+u['count']}",f"{c['total']:,} / {pct(c['total'],c['total']+u['total'])}",f"{c['min']}–{c['max']}",' / '.join(f"{u[k]:,.0f}" for k in ('min','p25','median','p75','p95','max')),f'{right} / {n-right-missing} / {missing}',len(hits),sum(x['remaining_reencoded'] for x in hits)))
s+=['','缺答沿用归档格式标志和缺少 boxed 的哨兵。对有答案的触顶记录，已用官方 grade_answer 复核最终分数。显式答案用原 first / box / grade_answer，逐个候选继续查到最早匹配 gold 的位置。没有被该提取器识别的口头结论、裸数、变量赋值可能漏检；中间量也可能恰好等于 gold。详见下文推导核查。token 位置是对归档文本重新分词的一基位置，剩余量按同一分词计算。归档生成长度与重分词长度分别保存。','',
'## 触顶回答的进度统计','',
'每个单元格依次为四个等 token 数区间。重复 5-gram 指此前出现过的原分词器五元组；段落按空行分隔、折叠空白、在段尾归档。其余列按每千 token 计。数学表达以 LaTeX 命令和二元运算符计数；反思词为 Wait、But、Hmm、Alternatively。角色标记为显式角色分隔符或行首角色名；语言切换为相继识别到的不同语言提示词（英、西、越、法、中、俄），另附脚本切换。词表代理会漏掉同一文字体系内的无标志切换。数学符号、反思词或重复率单独均不足以判断推理是否停滞。','',
*table(['模式 / 模型','重复 5-gram %','重复段落 %','数学','反思','角色','语言提示切换','文字体系切换'])]
for key in lens:
    ds,m=key.split()
    rr=[r for r in rows if (r['ds'],r['arm'])==(ds,m)]
    a=[sum((Counter(r['quarters'][k]) for r in rr),Counter()) for k in range(4)]
    vals=[]
    for k,den,f in [('repeated_5gram','grams',100),('repeated_paragraphs','paragraphs',100),('math','tokens',1000),('reflection','tokens',1000),('role','tokens',1000),('language_switch','tokens',1000),('script_switch','tokens',1000)]:
        vals.append(' / '.join(f'{f*v[k]/v[den]:.2f}' for v in a))
    s.append(line(key,*vals))
s+=['','## 人工审阅与各类预算','',
'AIME24 触顶全查；MATH 各模型按归档顺序、Python Random(20261124) 均匀无放回抽 8 条，不按题目、正误或重复率筛选。表中出现率以人工审阅回答为分母。预算采用粗分段估计：原文按字符四等分，各段结合中点、相邻位置和结尾判读，再把边界换算成原分词器 token 数。P 为尚在推进的有效中间步骤，N 为重复尝试且无可见进展，D 为失去实质内容的语言循环，U 为证据不足或同段混合。D 段也计入“无进展”列。一次新枚举、发现算错或尝试不同证明可以计作推进。出现率可重叠；预算是段级估计，不是逐 token 人工标签。正确答案后继续的预算独立以首次 explicit 正解末尾计算，与各段可重叠。','',
*table(['模式 / 模型','审阅 n','推进：条数 (%) / token','无进展（含失调）：条数 (%) / token','语言失调：条数 (%) / token','不确定：条数 (%) / token','显式正解后：条数 (%) / token','审阅 token 总数'])]
detail=[]
for key in lens:
    ds,m=key.split();cases=[v for v in reviews if v['id'][:2]==[ds,m]];n=len(cases);counts=Counter();budget=Counter();alltok=0
    for v in cases:
        labs,note=notes[tuple(v['id'])];b=v['char_quarter_token_bounds'];alltok+=b[-1]
        for code in 'PNDU':
            tags={code,'D'} if code=='N' else {code}
            counts[code]+=any(l in tags for l in labs)
            budget[code]+=sum(y-x for x,y,l in zip(b,b[1:],labs) if l in tags)
        if v['first_correct']:
            counts['C']+=1;budget['C']+=v['first_correct']['remaining_reencoded']
        ctrl=v['control']
        desc=f"{ctrl['model']} #{ctrl['sample']}，{ctrl['tokens']:,} token，{'对' if ctrl['correct'] else '错'}" if ctrl else '两模型同题均无未触顶样本'
        detail.append(line(ds,m,v['question_index'],v['id'][2],v['id'][3],'/'.join(labs),note,desc))
    s.append(line(key,n,*[f"{counts[k]} ({pct(counts[k],n)}) / {budget[k]:,}" for k in 'PNDUC'],f'{alltok:,}'))
s+=['','## 逐条对照','',
'题序为归档零基索引；MATH 并行引擎的原 problem_id 会重复。对照取同题同模型首个未触顶样本，若无则用另一模型，再查 B 同题归档，仍缺则明示。此对照描述同题可行路径，未按成功筛选。[原文与位置](review_evidence.json)同时给出各进度片段、正确答案上下文及对照末尾。','',
*table(['模式','模型','题序','原 ID','采样','分段','判断依据','同题未触顶']),*detail,'',
'归档校验、触顶和未触顶的均值与分位数见 [lengths.json](lengths.json)。全部 220 条的题目、最终得分、缺答、首次正确和生成进度指标见 [capped.json](capped.json)。复现顺序为 cap_audit.py、supplement.py、report.py；人工判断固定于 readings.tsv。']
(D/'REPORT.md').write_text('\n'.join(s)+'\n'+(D/'interpretation.md').read_text())
