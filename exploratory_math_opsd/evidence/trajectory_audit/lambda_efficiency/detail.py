"""Render detailed segment and semantic tables without rereading outputs from a model."""
import csv,json
from collections import Counter
from pathlib import Path
H=Path(__file__).resolve().parent
def load(n):return json.loads((H/n).read_text())
def row(xs):return '| '+' | '.join(map(str,xs))+' |'
s=['# 全量自动分段统计','',
'按全体 / 前 / 中 / 后 / 最后 1,024 token 列出。覆盖率和重复份额均按词元或实例数加权。最长连续匹配为该组在该段内观察到的最大 token 数。最后一段重叠。','']
for ds,arms in load('summary.json').items():
    for m,v in arms.items():
        s+=['',f'## {ds} · {m}','','| 指标 | 全 | 前 | 中 | 后 | 尾 1,024 |','|---|---:|---:|---:|---:|---:|']
        seg=v['segments']
        for key,label,den,factor in [
            *[(f'cov{k}',f'重复 {k}-gram 覆盖 %','n',100) for k in (4,8,16)],
            ('longest_max','最长连续匹配',None,1),
            *[(f'rep_{k}',f'重复 {k} %',k,100) for k in ('sentence','paragraph','expression','equation','calculation')],
            *[(k,k+' / 千 token','n',1000) for k in ('expression','equation','calculation','reflect','role','offtopic_cue','lang_switch','script_switch','control','double_control','format_run')]]:
            s.append(row([label,*[f'{z[key]*factor/(max(z[den],1) if den else 1):.3f}' for z in seg]]))
(H/'SEGMENTS.md').write_text('\n'.join(s).rstrip()+'\n')
data=load('blind.json');meta=load('unblind.json')
annot={k:(codes.split(),note) for k,codes,note in csv.reader((H/'blind_readings.tsv').open(),delimiter='\t')}
assert {v['id'] for v in data}==set(annot)==set(meta)
s=['# 语义审阅索引','','每条取早、中、晚三个窗口；区间为重分词后的零基半开 token 范围。短轨迹的重叠窗口不能相加估算总 token。抽样刻意覆盖稀少的错误和缺答单元，以下计数不可外推总体。完整引文见 blind.json。','',
'| 模式 / 组 | 轨迹数 | P | V | C | N | F | E | O | A | U |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for ds in ('aime24','nonthinking'):
    for m in ('B','R-lambda0@100','R-lambda05@100','R@100','R-lambda15@100'):
        ids=[k for k,v in meta.items() if v['ds']==ds and v['model']==m]
        cn=Counter(k for i in ids for k in set(annot[i][0]))
        s.append(row([ds+' '+m,len(ids),*[cn[k] for k in 'PVCNFEOAU']]))
s+=['','分类码与定义见 REPORT。下表摘录保留原始语种。','']
for v in data:
    k=v['id'];z=meta[k];codes,note=annot[k]
    s += [f"### {k} · {z['ds']} · {z['model']} · 题序 {z['q']} / 样本 {z['j']}",'','\n'.join(line.rstrip() for line in v['problem'].splitlines()),'',note,'']
    for c,x in zip(codes,v['spans']):
        s.append(f"- {c} · [{x['start']}, {x['end']}) · "+' '.join(x['text'].split()))
    s.append('')
(H/'SEMANTIC.md').write_text('\n'.join(s).rstrip()+'\n')
