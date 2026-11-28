import hashlib,json,unicodedata
from pathlib import Path
from tokenizers import Tokenizer
root=Path('/mnt/data/ForkAudit/math-opsd/models/Qwen3-1.7B')
path=root/'tokenizer.json'
t=Tokenizer.from_file(str(path))
vocab=t.get_vocab(with_added_tokens=True)
records=[]
for idx in sorted(vocab.values()):
    s=t.decode([idx],skip_special_tokens=False)
    if s and all(c.isspace() or unicodedata.category(c).startswith('P') or c in '#*_`>|-' for c in s.lstrip(' ')):
        records.append([idx,s])
out=dict(ids=[i for i,_ in records],decoded=records,tokenizer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),decoded_ids=len(vocab),unicode_version=unicodedata.unidata_version)
Path(__file__).with_name('format_tokens.json').write_text(json.dumps(out,ensure_ascii=False)+'\n')
