"""Action body mask on original sampled tokenization; never fall back to reasoning."""
import re
ACTION=re.compile(r'[<\[]\s*action\s*[>\]](.*?)[<\[]\s*/\s*action\s*[>\]]',re.I|re.S)
def action_mask(tokenizer,ids):
 text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 matches=list(ACTION.finditer(text))
 if not matches:return None
 match=matches[-1];start,end=match.span(1)
 while start<end and text[start].isspace():start+=1
 while end>start and text[end-1].isspace():end-=1
 if start==end:return None
 # Prefix decoding respects the actual generated token IDs, including
 # noncanonical tokenizations that encode(decode(ids)) would change.
 bounds=[0]+[len(tokenizer.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False)) for i in range(1,len(ids)+1)]
 mask=[a<end and b>start for a,b in zip(bounds,bounds[1:])]
 return mask if any(mask) else None
