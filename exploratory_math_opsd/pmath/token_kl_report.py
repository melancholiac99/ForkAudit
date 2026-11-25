"""Aggregate role-specific vocabulary totals, with explicit disjoint token groups."""
import json,re,unicodedata
from pathlib import Path
import torch
from transformers import AutoTokenizer
from common import MODEL,WORK,json_save
from token_kl import OUT,TARGETS,TEMPERATURES

WORDS='Wait Hmm So But Let Now Therefore Thus Alternatively Actually Okay First Next'.split()
GROUPS=('空白与换行','标点','推理连接词','数字、运算符和 LaTeX 数学命令','其他')
OPS=set('+-*/^=<>%|~&$\\')
NUM=re.compile(r'[+-]?(?:\d+(?:[.,]\d+)*|\.\d+)(?:[eE][+-]?\d+)?')
COMMAND=re.compile(r'[{}$]*(?:\\[A-Za-z]+|\\[^\w\s])(?:[{}$]|\\[A-Za-z]+)*')
def classify(text):
    if text and text.isspace():return 0
    if text in WORDS or (text.startswith(' ') and text[1:] in WORDS):return 2
    body=text.strip()
    if body and (NUM.fullmatch(body) or COMMAND.fullmatch(body) or
                 all(c in OPS or unicodedata.category(c)=='Sm' for c in body)):return 3
    if body and all(unicodedata.category(c).startswith('P') for c in body):return 1
    return 4

def main():
    parts=[torch.load(OUT/f'lane{i}-{j}.pt',weights_only=True) for i in range(4) for j in range(0,375,25)]
    questions=sum(x['questions'] for x in parts)
    positions=sum(x['positions'] for x in parts)
    folds=[sum(x['folds'][f] for x in parts) for f in range(4)]
    assert questions==1500 and folds==[375]*4
    oof=json.loads((WORK/'oof.json').read_text())
    assert positions==oof['positions']
    mass=sum(x['mass'] for x in parts)
    signed=sum(x['signed'] for x in parts)
    assert (signed.sum(-1)>0).all() and (signed.sum(-1)<=mass.sum(-1)).all()
    tok=AutoTokenizer.from_pretrained(MODEL)
    vocab=mass.size(-1)
    tokenizer_size=len(tok)
    labels=[tok.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False) if i<tokenizer_size else '' for i in range(vocab)]
    groups=torch.tensor([classify(t) for t in labels])
    report=dict(positions=positions,questions=questions,fold_counts=folds,scale=oof['scale'],
                max_first_position_archived_centered_label_error=max(x['max_label_error'] for x in parts),
                matched_connector_forms=[t for t in labels if classify(t)==2],
                rule='Vocabulary-ID groups. Strip outer whitespace for punctuation/math. Entire-token Unicode P; math first for decimal numerals, Unicode Sm, ASCII operators, TeX command or backslash tokens. Exact case-sensitive connector spellings with zero or one leading ASCII space; no context-based reassignment of split words/commands. Empty and undecodable token fragments fall under other.',
                temperatures={})
    out=Path(__file__).resolve().parents[1]/'evidence/pmath/token_kl'
    out.mkdir(parents=True,exist_ok=True)
    lines=['# 数学投影器的折外正 KL 贡献','',
           f'{questions:,} 道拟合题，{positions:,} 个回答位置；每折留出 375 题，训练 λ={oof["scale"]:.8f}。每道题用排除它的投影器。','',
           '完整词表上以 q(v)·[log q(v)−log p₀(v)] 计单项贡献，先在每个位置取正部，再按 token ID 累加。每列以本目标的正贡献总和归一。三个目标的 logits 分别为 t₀、s₀+c、t₀−c，其中 c 只在 s₀ 的 top-64 上取 λf。用基础模型 BF16 分数，FP32 加减、归一化和计算 KL，FP64 汇总；未做训练损失的 0.05 截断。','',
           '先按完整 token 解码分类。纯空白含换行；连接词限指定大小写和零或一个前导空格；数字、数学符号和完整 LaTeX 命令优先于 Unicode 标点，单独反斜杠归数学。分类时去掉标点或数学片段外侧空白。切成多段的连接词和命令不在上下文中拼接。']
    for ti,temp in enumerate(TEMPERATURES):
        sub={}
        for a,name in enumerate(TARGETS):
            m=mass[ti,a]
            total=m.sum().item()
            bygroup=[m[groups==i].sum().item()/total for i in range(len(GROUPS))]
            assert abs(sum(bygroup)-1)<1e-9
            sub[name]=dict(positive_mass=total,positive_per_position=total/positions,
                           net_kl_per_position=signed[ti,a].sum().item()/positions,
                           top20=[dict(id=i,token=labels[i],share=m[i].item()/total) for i in m.topk(20).indices.tolist()],
                           groups=dict(zip(GROUPS,bygroup)))
        ratios={g:{a:sub[a]['groups'][g]/sub['R']['groups'][g] for a in ('P','residual')} for g in GROUPS}
        report['temperatures'][str(temp)]={**sub,'ratios_to_R':ratios}
        lines+=['',f'## T = {temp}'+('（训练温度）' if ti==0 else '（单位温度）'),'',
                '| 组别 | R | P | 残差 | P/R | 残差/R |','|---|---:|---:|---:|---:|---:|']
        for g in GROUPS:
            values=[sub[a]['groups'][g] for a in TARGETS]
            lines+=['| '+g+' | '+' | '.join(f'{v:.3%}' for v in values)+f' | {ratios[g]["P"]:.3f} | {ratios[g]["residual"]:.3f} |']
        lines+=['','| 排名 | R token（占比） | P token（占比） | 残差 token（占比） |','|---:|---|---|---|']
        for i in range(20):
            cols=[]
            for a in TARGETS:
                v=sub[a]['top20'][i]
                label=json.dumps(v['token'],ensure_ascii=False).replace('|','&#124;').replace('`','&#96;')
                cols.append(f'`{label}` · {v["share"]:.3%}')
            lines+=['| '+str(i+1)+' | '+' | '.join(cols)+' |']
    json_save(out/'results.json',report)
    (out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
