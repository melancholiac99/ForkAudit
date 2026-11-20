"""Frozen P-pre templates. Evidence has no access to a goal or a gold product."""
import json
import re

POLICIES = ('none', 'full', 'product', 'evidence', 'placeholder')
MATCH = {'full': 'placeholder', 'product': 'placeholder_product', 'evidence': 'placeholder_evidence'}
FIELDS = {
    'full': [('商品编号', 'goal.asin', 'no'), ('规格', 'goal.goal_options', 'partial'), ('价格上限', 'goal.price_upper', 'partial')],
    'product': [('商品编号', 'goal.asin', 'no'), ('规格', 'goal.goal_options', 'partial'), ('价格上限', 'goal.price_upper', 'partial'), ('商品名称', 'catalog.title', 'partial'), ('属性', 'catalog.attribute', 'partial')],
    'evidence': [('按展示顺序的首个规格轴', 'catalog.customization_options keys for current public.products', 'partial')],
}
HEADER = '补充购物资料'
EMPTY = '未提供'

def packet(mode, public, catalog, goal=None):
    if mode == 'none':
        return {}, None
    kind = next((k for k, v in MATCH.items() if v == mode), mode)
    keys = [f[0] for f in FIELDS[kind]]
    if kind == 'evidence':
        products = public.get('products', [])
        if public['page_type'] != 'search_results' or not products:
            return {}, 'PACKET_UNAVAILABLE'
        if any(p['asin'] not in catalog for p in products):
            return {}, 'PACKET_UNAVAILABLE'
    if mode in MATCH.values():
        return dict.fromkeys(keys, EMPTY), None
    if kind == 'evidence':
        values = [[next(iter(catalog[p['asin']]['customization_options'] or {}), '无') for p in products]]
    elif kind == 'full':
        values = [goal['asin'], goal['goal_options'], goal['price_upper']]
    else:
        p = catalog[goal['asin']]
        values = [goal['asin'], goal['goal_options'], goal['price_upper'], p['title'], p['attribute']]
    return dict(zip(keys, values)), None

def render(tokenizer, native, fields, reason=None):
    """No second history truncation. Record unavailable exposures explicitly."""
    ids = native['token_ids']
    if reason:
        return ids, reason, {}, 0
    messages = native['messages'].copy()
    counts = {}
    length = 0
    if fields:
        lines = [HEADER]
        for k, v in fields.items():
            value = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(',', ':'))
            lines.append(k + '：' + value)
            counts[k] = len(tokenizer.encode(value, add_special_tokens=False))
        block = '\n'.join(lines)
        length = len(tokenizer.encode(block, add_special_tokens=False))
        cap = 1024 if set(fields) == {FIELDS['evidence'][0][0]} else 192
        if length > cap:
            return ids, 'PACKET_OVERFLOW', counts, length
        messages.insert(2, {'role': 'user', 'content': block})
    final = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not fields and final != ids:
        raise ValueError('empty packet changed native ids')
    return (final, None, counts, length) if len(final) <= 8192 else (ids, 'CONTEXT_OVERFLOW', counts, length)

def categories(tokenizer, ids, public):
    # Decode byte-BPE spans, including split UTF-8 tokens, against actual IDs.
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    reverse = {v:k for k,v in bytes_to_unicode().items()}
    parts = [bytes(reverse[c] for c in t) if i not in tokenizer.all_special_ids else t.encode()
             for i,t in zip(ids,tokenizer.convert_ids_to_tokens(ids))]
    data = b''.join(parts)
    m = re.search(rb'(?:^|\n|\\n)\s*Action:\s*(search|click|finish)\[(.*)\]\s*(?:<\|im_end\|>)?\s*$', data, re.S)
    labels = []
    offset = 0
    options = public.get('available_options', {})
    arg = m[2].decode('utf-8').strip().lower() if m else ''
    for i,b in zip(ids,parts):
        end = offset + len(b)
        label = 'unknown'
        if i in tokenizer.all_special_ids: label = 'format'
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
