"""First-command byte boundaries and single-token alternatives on masked supports."""
from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate
import re

from .parser import byte_decoder
from exploratory_shop_audit.grpo_forced_audit.common import MODEL, read


def vocabulary():
    reverse = byte_decoder()
    vocab = read(MODEL/'vocab.json')
    tokens = {i:bytes(reverse[ch] for ch in token) for token,i in vocab.items()}
    added = read(MODEL/'tokenizer_config.json')['added_tokens_decoder']
    tokens.update({int(i):t['content'].encode() for i,t in added.items()})
    eos = set(read(MODEL/'generation_config.json')['eos_token_id'])
    return tokens,eos


VERB = rb'[A-Za-z_][A-Za-z_0-9]*\['
COMMAND = re.compile(rb'(?<![a-zA-Z_0-9])'+VERB)
SEPARATOR = rb'(?:\s|\\n)*'
IMMEDIATE = re.compile(SEPARATOR+rb'(?:Action:'+SEPARATOR+rb')?'+VERB, re.I)
LATER_LINE = re.compile(rb'(?:\r?\n|\\n)'+SEPARATOR+rb'(?:Action:'+SEPARATOR+rb')?'+VERB, re.I)
NEWLINE = re.compile(rb'^[ \t]*(?:\r?\n|\\n)')
COMMAND_WORDS = (b'click',b'search',b'finish',b'Action:',b'search_products')


@dataclass(frozen=True)
class Position:
    index: int
    role: str
    prefix: bytes  # Bytes of the sampled token through its closing bracket.


def after_role(suffix):
    if not suffix.strip(b' \t'):
        return 'after_bare'
    return 'after_fused_newline' if NEWLINE.match(suffix) else 'after_other_suffix'


def locate(parts):
    data = b''.join(parts)
    m = COMMAND.search(data)
    if not m:
        return [],None,'no_command'
    depth = 1
    for j in range(m.end(),len(data)):
        depth += int(data[j]==91) - int(data[j]==93)
        if depth==0:
            break
    else:
        return [],None,'unclosed_first_command'
    after = data[j+1:]
    second = IMMEDIATE.match(after) or LATER_LINE.search(after)
    next_command = None
    if second:
        separator = after[:second.end()].replace(b'\\n',b'\n')
        next_command = 'newline' if b'\n' in separator else 'same_line'
    ends = list(accumulate(map(len,parts)))
    k = bisect_right(ends,j)
    prefix = data[ends[k-1] if k else 0:j+1]
    suffix = data[j+1:ends[k]]
    pos = [Position(k,'bracket',prefix)]
    if k+1<len(parts):
        pos.append(Position(k+1,after_role(suffix),b''))
    return pos,next_command,'located'


def candidate_classes(position,ids,tokens,eos):
    stop, cont = [], []
    for i,idx in enumerate(ids):
        word = tokens[int(idx)]
        if position.role=='bracket':
            if not word.startswith(position.prefix):
                continue
            tail = word[len(position.prefix):]
            if not tail.strip(b' \t'):
                stop.append(i)
            elif NEWLINE.match(tail):
                cont.append(i)
        elif idx in eos:
            stop.append(i)
        elif NEWLINE.match(word):
            cont.append(i)
        elif position.role=='after_fused_newline':
            # Newline is already in the prefix. Count possible command onset on this step.
            word = word.lstrip(b' \t')
            if word and (any(w.startswith(word) for w in COMMAND_WORDS) or COMMAND.match(word)):
                cont.append(i)
    return stop,cont
