"""Native action spans, including direct command responses."""
import re
from functools import lru_cache

@lru_cache(maxsize=1)
def byte_decoder():
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    return {v:k for k,v in bytes_to_unicode().items()}

def command_parts(tokenizer, ids):
    # Decode byte-BPE spans, including split UTF-8 tokens, against actual IDs.
    reverse=byte_decoder()
    special_ids=set(tokenizer.all_special_ids)
    parts = [bytes(reverse[c] for c in t) if i not in special_ids else t.encode()
             for i,t in zip(ids,tokenizer.convert_ids_to_tokens(ids))]
    data = b''.join(parts)
    m = re.search(rb'(?:^|\n|\\n)\s*Action:\s*(search|click|finish)\[(.*)\]\s*(?:<\|im_end\|>)?\s*$', data, re.S)
    if not m:
        m = re.fullmatch(rb'\s*(search|click|finish)\[(.*)\]\s*(?:<\|im_end\|>)?\s*', data, re.S)
    return parts, m

def categories(tokenizer, ids, public):
    parts, m = command_parts(tokenizer, ids)
    labels = []
    offset = 0
    options = public.get('available_options', {})
    arg = m[2].decode('utf-8',errors='replace').strip().lower() if m else ''
    special_ids=set(tokenizer.all_special_ids)
    for i,b in zip(ids,parts):
        end = offset + len(b)
        label = 'unknown'
        if i in special_ids: label = 'format'
        elif m:
            if end <= m.start(): label = 'thought'
            elif offset < m.end(2) and end > m.start(2):
                label = m[1].decode()
                if label == 'finish': label = 'format'
                elif label == 'click' and public['page_type']=='product_detail' and arg in {str(v).lower() for vals in options.values() for v in vals}: label = 'option'
            else: label = 'format'
        labels.append(label)
        offset = end
    return labels
