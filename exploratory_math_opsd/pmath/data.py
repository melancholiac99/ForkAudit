import argparse, glob, json, sys
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import ROOT,WORK,MODEL,save,json_save

def stream():
    data = [r for path in sorted((ROOT/'datasets/train/data').glob('*.parquet'))
            for r in pq.ParquetFile(path).read(columns=['problem','solution']).to_pylist()]
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(42)).tolist()
    # Check the recovered global batch against R's archived rank-zero generations.
    for file in (ROOT/'run/R/generations').glob('*.json'):
        for rec in json.loads(file.read_text())['generations']:
            problem=rec['prompt'].split('Problem: ',1)[1].rsplit('\n\nPlease reason',1)[0]
            step=rec['step']
            assert any(data[i]['problem']==problem for i in order[step*32:(step+1)*32])
    selected=order[3201:4701]
    assert len({data[i]['problem'] for i in selected})==1500
    assert not {data[i]['problem'] for i in selected}.intersection(data[i]['problem'] for i in order[:3200])
    return [(i,data[i]) for i in selected]

def main(lane,stage):
    WORK.mkdir(exist_ok=True)
    sys.path.insert(0,str(ROOT/'source'))
    from data_collator import SelfDistillationDataCollator
    jobs=stream()[lane::4]
    tok=AutoTokenizer.from_pretrained(MODEL)
    collate=SelfDistillationDataCollator(tok,max_length=20000,reason_first=False,student_thinking=False,teacher_thinking=True)
    path=WORK/f'rollouts-{lane}.pt'
    if stage=='sample':
        if path.exists(): return
        from vllm import LLM, SamplingParams
        llm=LLM(model=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,max_model_len=20000,seed=lane,max_num_seqs=8,gpu_memory_utilization=.6)
        batch=[collate([row]) for _,row in jobs]
        student=[x['student_prompts'][0].tolist() for x in batch]
        prompts=[tok.decode(ids,skip_special_tokens=False) for ids in student]
        generated=llm.generate(prompts,SamplingParams(n=1,temperature=1.1,top_k=20,top_p=.95,
                                                       repetition_penalty=1.,presence_penalty=0.,min_p=0.,max_tokens=1024))
        records=[]
        for (idx,_),x,ids,output in zip(jobs,batch,student,generated):
            assert ids==list(output.prompt_token_ids)
            response=list(output.outputs[0].token_ids)
            assert 0<len(response)<=1024
            records.append((idx,ids,x['teacher_prompts'][0].tolist(),response))
        assert len(records)==375
        save(path,records)
        return
    device='cuda'
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval()
    recs=torch.load(path,weights_only=True)
    with torch.inference_mode():
        for pos,(idx,s,t,y) in enumerate(recs):
            dest=WORK/f'features-{idx}.pt'
            if dest.exists():continue
            def hidden(prompt):
                ids=torch.tensor([prompt+y],device=device)
                return model.model(input_ids=ids,use_cache=False).last_hidden_state[0,len(prompt)-1:-1]
            hs,ht=hidden(s),hidden(t)
            pieces=[]
            for j in range(0,len(y),64):
                sl=model.lm_head(hs[j:j+64]).float()
                tl=model.lm_head(ht[j:j+64]).float()
                ids=torch.cat([sl.topk(32).indices,tl.topk(32).indices],1).sort().values
                valid=torch.ones_like(ids,dtype=torch.bool)
                valid[:,1:]=ids[:,1:]!=ids[:,:-1]
                a=sl.gather(1,ids)
                w=a.masked_fill(~valid,-torch.inf).softmax(1)
                d=tl.gather(1,ids)-a
                pieces.append((ids.int().cpu(),(d-(w*d).sum(1,keepdim=True)).cpu(),valid.cpu()))
            save(dest,dict(h=hs.cpu(),ids=torch.cat([p[0] for p in pieces]),d=torch.cat([p[1] for p in pieces]),valid=torch.cat([p[2] for p in pieces])))
        if lane==0: save(WORK/'head.pt',model.lm_head.weight.detach().cpu())

if __name__=='__main__':
    main(int(sys.argv[1]),sys.argv[2])
