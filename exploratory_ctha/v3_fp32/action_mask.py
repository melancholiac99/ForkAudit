"""Exact offsets only on canonical tokenizations; unknown boundaries fail closed."""
import re

def byte_offsets(tokenizer,ids,text):
 # GPT-2/Qwen ByteLevel alphabet. Use exact token bytes, never partial UTF-8 decoding.
 decoder=getattr(getattr(tokenizer,'backend_tokenizer',None),'decoder',None)
 if type(decoder).__name__!='ByteLevel':return None
 bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs.copy();n=0
 for b in range(256):
  if b not in bs:bs.append(b);cs.append(256+n);n+=1
 table={chr(c):b for b,c in zip(bs,cs)};special=set(tokenizer.all_special_ids);chunks=[]
 try:
  for i,tok in zip(ids,tokenizer.convert_ids_to_tokens(ids)):
   chunks.append(tok.encode('utf-8') if i in special else bytes(table[c] for c in tok))
  if b''.join(chunks).decode('utf-8',errors='strict')!=text:return None
 except (KeyError,UnicodeError,TypeError):return None
 offsets=[];pos=0
 for chunk in chunks:offsets.append((pos,pos+len(chunk)));pos+=len(chunk)
 return offsets
ACTION=re.compile(r'[<\[]\s*action\s*[>\]](.*?)[<\[]\s*/\s*action\s*[>\]]',re.I|re.S)
def action_mask(tokenizer,ids):
 text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 matches=list(ACTION.finditer(text))
 if not matches:return None
 start,end=matches[-1].span(1)
 while start<end and text[start].isspace():start+=1
 while end>start and text[end-1].isspace():end-=1
 if start==end:return None
 try:encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
 except (TypeError,NotImplementedError,ValueError):return None
 if list(encoded['input_ids'])==list(ids):
  offsets=encoded['offset_mapping']
 else:
  offsets=byte_offsets(tokenizer,ids,text)
  if offsets is None:return None
  start,end=len(text[:start].encode('utf-8')),len(text[:end].encode('utf-8'))
 if len(offsets)!=len(ids):return None
 mask=[a<end and b>start for a,b in offsets]
 return mask if any(mask) else None
