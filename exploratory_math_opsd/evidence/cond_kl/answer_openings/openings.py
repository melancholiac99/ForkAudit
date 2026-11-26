"""CPU summaries of surface openings in completed MATH-500 archives."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'boundary_answers'))
from audit import archive, org

ARMS = ('B','T0@100','R@100','R-noclip@100','R-cond@100','R-clip-boundary@100',
        'R-clip-interior@100','R-lambda0@100','R-lambda05@100')
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
STYLE = re.compile(r'(?:okay|alright|hmm|let[ \t]+me|so[ \t]+i[ \t]+need|i[ \t]+need[ \t]+to)\b',re.I)
WAIT = re.compile(r'\bwait\b',re.I)
TAGS = re.compile(r'^\s*(?:</?think>\s*)+')


def opening(text, thinking):
    if thinking:
        _,marker,body=text.partition('<think>')
        if not marker:
            raise ValueError('Thinking reference has no opening tag')
        text=body.partition('</think>')[0]
    else:
        text=TAGS.sub('',text)
    word=WORD.search(text)
    if word is None:
        return '(无词)',False
    return word.group().casefold(),bool(STYLE.match(text,word.start()))


def summarize(rows):
    n=len(rows)
    if not n:
        return dict(n=0,accuracy=None,tokens=None,closing_pct=None,wait_per_1000=None)
    tokens=sum(g['generated_tokens'] for g in rows)
    return dict(n=n,accuracy=100*sum(g['correct'] for g in rows)/n,tokens=tokens/n,
                closing_pct=100*sum('</think>' in g['full_generation'] for g in rows)/n,
                wait_per_1000=1000*sum(len(WAIT.findall(g['full_generation'])) for g in rows)/tokens)


def measure(arm,budget,thinking):
    qs,digest=archive(arm,budget,thinking)
    words=Counter();groups={True:[],False:[]}
    leading_close=0
    for q in qs:
        for g in q['generations']:
            txt=g['full_generation']
            word,style=opening(txt,thinking)
            words[word]+=1
            groups[style].append(g)
            leading_close+=txt.lstrip().startswith('</think>')
    n=sum(words.values())
    top=sorted(words.items(),key=lambda p:(-p[1],p[0]))[:15]
    return dict(n=n,archive_sha256=digest,leading_closing_tag=leading_close,
                thinking_opening_pct=100*len(groups[True])/n,
                top_words=[dict(word=w,n=k,pct=100*k/n) for w,k in top],
                groups={label:summarize(groups[key]) for key,label in ((True,'思考式'),(False,'其他'))})


def main():
    assert opening('</think>\n**Okay**, now',False)==('okay',True)
    assert opening('<think>\nLet me verify.</think>\nYep',True)==('let',True)
    assert opening('So, I need this',False)==('so',False)
    assert opening('We should say Okay.',False)==('we',False)
    report={};missing=[]
    for arm in ARMS:
        receipt=org.ROOT/'run/math500-long-seed1001-n2'/('nonthinking-'+arm)/'receipt.json'
        if arm.startswith('R-lambda') and not receipt.exists():
            missing.append(arm);continue
        report[arm+' · 16k']=measure(arm,16384,False)
    report['B 思考 · 32k']=measure('B',32768,True)
    report['R-clip-boundary@100 · 4k']=measure('R-clip-boundary@100',4096,False)
    (HERE/'results.json').write_text(json.dumps(dict(models=report,missing_16k=missing),ensure_ascii=False,indent=2)+'\n')
    lines=['# MATH-500 回答开头','','首词为首个词法单元（字母或数字，可带词内撇号），忽略大小写、开头的控制标签、标点及排版符号。思考式严格从首词匹配 Okay、Alright、Hmm、Let me、So I need、I need to；短语中只允许空格或制表符。并列首词按字母顺序列出。','',
           'B 思考组从第一个 <think> 内抽取开头；所有组的正确性、原生生成长度、结束标签及 Wait 都在整条回答上统计。Wait 为大小写不敏感的完整单词，频率为每个分层的总次数除以原生 token 总数再乘 1,000。非思考 16k 和思考 32k 均为每题两条；4k 为每题四条。每档均为同一组 500 道题。','',
           '|模型与归档|思考式开头占比|开头类型|回答数|准确率|平均长度|含结束标签|Wait / 千 token|','|---|---:|---|---:|---:|---:|---:|---:|']
    for name,x in report.items():
        for kind,r in x['groups'].items():
            metrics='—|—|—|—' if not r['n'] else f"{r['accuracy']:.2f}%|{r['tokens']:.1f}|{r['closing_pct']:.2f}%|{r['wait_per_1000']:.3f}"
            lines.append(f"|{name}|{x['thinking_opening_pct']:.2f}%|{kind}|{r['n']}|{metrics}|")
    lines+=['','## 首词前 15 名','','百分比的分母为该组全部回答。出现不足 15 种时列出全部。','','|模型|首词（占比）|','|---|---|']
    for name,x in report.items():
        lines.append('|'+name+'|'+'，'.join(f"{r['word']} ({r['pct']:.2f}%)" for r in x['top_words'])+'|')
    if missing:lines+=['','尚无完整 16k 回执：'+', '.join(missing)+'。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
