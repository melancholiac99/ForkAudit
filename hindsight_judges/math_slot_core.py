"""CPU-only deterministic MATH data, prompt, cut, verifier and statistics helpers."""
import ast,hashlib,json,re
from fractions import Fraction
from pathlib import Path
import numpy as np
ARMS=('ACTIVE','QUOTE-CONTINUE','QUOTE-REWRITE','INDEPENDENT')
TASK='Solve the problem step by step. Give your final answer in \\boxed{...}.'
DEPS=Path('/root/autodl-tmp/math-slot-deps')
def seed(namespace,qid,index=0):
 return int(hashlib.sha256(f'20261106:{namespace}:{qid}:{index}'.encode()).hexdigest()[:15],16)
def boxed(text):
 found=[]
 for m in re.finditer(r'\\(?:boxed|fbox)\s*\{',text):
  depth=1;end=m.end()
  while end<len(text) and depth:
   depth+=(text[end]=='{')-(text[end]=='}');end+=1
  if depth==0:found.append(text[m.end():end-1])
 return found[-1] if found else None

def parse_answer(text):
 import sys
 if str(DEPS) not in sys.path:sys.path.insert(0,str(DEPS))
 from math_verify import parse
 b=boxed(text)
 expr=('\\boxed{'+b+'}') if b is not None else text
 parsed=parse(expr,fallback_mode='no_fallback',parsing_timeout=3)
 return parsed,('boxed' if b is not None else 'other'),expr

def grade(gold,response,student=None):
 import sys
 if str(DEPS) not in sys.path:sys.path.insert(0,str(DEPS))
 from math_verify import verify
 g,_,_=parse_answer(gold);assert g,'Gold is not parseable'
 p,fmt,expr=parse_answer(response)
 result=dict(parseable=bool(p),format=fmt,answer=expr if fmt=='boxed' else str(p),correct=bool(p and verify(g,p,timeout_seconds=3)))
 if student is not None:
  s,_,_=parse_answer(student);assert s
  result['retains_student_answer']=bool(p and verify(s,p,timeout_seconds=3))
 return result

def _number(s):
 node=ast.parse(s.strip().replace('^','**'),mode='eval')
 def go(n):
  if isinstance(n,ast.Expression):return go(n.body)
  if isinstance(n,ast.Constant) and type(n.value)==int and abs(n.value)<10**12:return Fraction(n.value)
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):return go(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
  if isinstance(n,ast.BinOp):
   a,b=go(n.left),go(n.right)
   if isinstance(n.op,ast.Add):return a+b
   if isinstance(n.op,ast.Sub):return a-b
   if isinstance(n.op,ast.Mult):return a*b
   if isinstance(n.op,ast.Div):return a/b
   if isinstance(n.op,ast.Pow) and b.denominator==1 and abs(b)<=12:return a**int(b)
  raise ValueError('Not a bounded exact rational expression')
 return go(node)

def cuts(text):
 limit=min([m.start() for m in re.finditer(r'\\(?:boxed|fbox)\s*\{',text)] or [len(text)])
 return [m.end() for m in re.finditer(r'\n\s*\n',text) if 30<=m.end()<limit and text[m.end():].strip()]

def propose(text):
 ends=cuts(text)
 if not ends:return None
 suggestions=[]
 # Only isolated inline/display mathematical spans; contextual correctness still needs review.
 for m in re.finditer(r'\$([^$\n]+)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]',text,re.S):
  expression=next(v for v in m.groups() if v is not None).strip()
  if expression.count('=')!=1 or not re.fullmatch(r'[0-9\s+*/^()=\-]+',expression):continue
  try:
   a,b=expression.split('=');left,right=_number(a),_number(b)
   if left!=right and any(x>=m.end() for x in ends):suggestions.append(dict(span=[m.start(),m.end()],quote=m.group(),left=str(left),right=str(right)))
  except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
 end=next((x for x in ends if suggestions and x>=suggestions[0]['span'][1]),min(ends,key=lambda x:abs(x-len(text)/2)))
 return dict(prefix_end=end,prefix=text[:end],candidate_errors=suggestions,status='review_required')

def validate_annotation(text,a):
 assert a['prefix_end'] in cuts(text),'Cut must be a pre-answer paragraph boundary'
 assert a['cohort'] in ('error_observed','error_unknown') and a['reviewer'] and a['reason']
 if a['cohort']=='error_observed':
  assert a.get('error_quote') and a['error_quote'] in text[:a['prefix_end']]
 return text[:a['prefix_end']]

def prompt(tok,problem,arm,prefix=''):
 base=TASK+'\n\n'+problem
 if arm in ('ACTIVE','INDEPENDENT'):messages=[dict(role='user',content=base)]
 else:
  assert arm in ARMS
  request=('Continue from the interruption. Output only the continuation. Do not restart the solution.' if arm=='QUOTE-CONTINUE' else 'You may re-analyze the problem and rewrite the solution from the beginning.')
  messages=[dict(role='user',content=base+'\n\nPartial draft:\n<partial_draft>\n'+prefix+'\n</partial_draft>'),dict(role='user',content=request)]
 rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 if arm=='ACTIVE':rendered+=prefix
 ids=tok.encode(rendered,add_special_tokens=False)
 return dict(messages=messages,active_prefix=prefix if arm=='ACTIVE' else '',input_ids=ids,rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest())

def paired_ci(values):
 a=np.asarray(values,dtype=float)
 if not len(a):return dict(n=0,mean=None,ci95=None)
 rng=np.random.default_rng(20261106)
 means=np.mean(a[rng.integers(len(a),size=(4000,len(a)))],axis=1)
 return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist())

def analysis(rows):
 result={}
 for cohort in ('error_observed','error_unknown','all_descriptive'):
  group=[r for r in rows if cohort=='all_descriptive' or r['cohort']==cohort]
  arms={arm:{metric:paired_ci([100*float(r['grades'][arm][metric]) for r in group]) for metric in ('correct','retains_student_answer')} for arm in ARMS}
  diffs={metric:paired_ci([100*(float(r['grades']['QUOTE-CONTINUE'][metric])-float(r['grades']['ACTIVE'][metric])) for r in group]) for metric in ('correct','retains_student_answer')}
  result[cohort]=dict(questions=len(group),arms=arms,quote_continue_minus_active_pp=diffs)
 return dict(cohorts=result,bootstrap=dict(unit='question',replicates=4000,seed=20261106),interpretation='Main correction inference only error_observed. Presentation plus task framing; not pure role causality or OPD training benefit.')
