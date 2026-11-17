"""Controlled fixed-prefix likelihood checks; no policy updates or new episodes."""
import argparse
import contextlib
import json
import time


def chunked_fp32_attention(module, query, key, value, attention_mask=None,
                           scaling=None, dropout=0., **kwargs):
    """Memory-bounded causal FP32 attention reference for diagnostic inference."""
    import torch
    assert dropout == 0
    groups=query.shape[1]//key.shape[1]
    with torch.autocast(query.device.type,enabled=False):
        q=query.float();k=key.float().repeat_interleave(groups,dim=1)
        v=value.float().repeat_interleave(groups,dim=1)
        length=q.shape[-2];total=k.shape[-2];chunks=[]
        for start in range(0,length,128):
            end=min(start+128,length)
            scores=(q[:,:,start:end]@k.transpose(-1,-2))*(scaling or query.shape[-1]**-.5)
            if attention_mask is not None:
                mask=attention_mask[:,:,start:end,:total]
                if mask.dtype==torch.bool:scores=scores.masked_fill(~mask,float('-inf'))
                else:scores=scores+mask.float()
            else:
                invalid=torch.arange(total,device=q.device)[None,:]>torch.arange(start,end,device=q.device)[:,None]+total-length
                scores=scores.masked_fill(invalid,float('-inf'))
            chunks.append((scores.softmax(-1)@v).to(query.dtype))
    return torch.cat(chunks,dim=-2).transpose(1,2).contiguous(),None


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from verl import DataProto
    from .common import DEFAULT_RUN, MODEL, save, utc
    p=argparse.ArgumentParser();p.add_argument('--cpu-fp32',action='store_true')
    p.add_argument('--residual-fp32',action='store_true');args=p.parse_args()
    run=DEFAULT_RUN.resolve();folder=run/'alignment_diagnostic/checkpoint_0002'
    output=folder/('root_cpu_fp32.json' if args.cpu_fp32 else 'root_gpu_residual_fp32.json' if args.residual_fp32 else 'root_gpu.json')
    if output.exists():raise FileExistsError(output)
    if not args.cpu_fp32:torch.cuda.set_per_process_memory_fraction(.25)
    torch.set_num_threads(16 if args.cpu_fp32 else 2)
    started=time.monotonic()
    packet=DataProto.load_from_disk(str(next((run/'native_batch_cache').glob('train_0002_sample_*.pkl'))));packet.to('cpu')
    dtype=torch.float32 if args.cpu_fp32 else torch.bfloat16
    device='cpu' if args.cpu_fp32 else 'cuda:0'
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),torch_dtype=dtype,
        attn_implementation='sdpa' if args.cpu_fp32 else 'flash_attention_2',device_map=device)
    model=PeftModel.from_pretrained(model,str(run/'training/checkpoints/global_step_2/actor/lora_adapter'),
                                    is_trainable=True,autocast_adapter_dtype=True)
    model.eval();tok=AutoTokenizer.from_pretrained(str(MODEL))
    if args.residual_fp32:
        assert not args.cpu_fp32
        for layer in model.get_base_model().model.layers:
            layer.register_forward_pre_hook(lambda module,inputs:(inputs[0].float(),*inputs[1:]))
    partial=output.with_suffix('.partial.json')
    records=json.loads(partial.read_text())['records'] if partial.exists() else []
    cases=[(508,2)] if args.residual_fp32 else [(508,2),(432,11),(327,2)]
    for row,pos in cases:
        prompt=packet.batch['prompts'][row];pmask=packet.batch['attention_mask'][row,:len(prompt)].bool()
        response=packet.batch['responses'][row];rmask=packet.batch['attention_mask'][row,-len(response):].bool()
        prefix=torch.cat([prompt[pmask],response[:pos]]).to(device)[None]
        full=torch.cat([prompt[pmask],response[rmask]]).to(device)[None]
        target=int(response[pos]);pred=prefix.shape[-1]-1
        def compute(name,ids,keep,cache=False,math_attention=False):
            if any(r['row']==row and r['mode']==name for r in records):return
            attn=contextlib.nullcontext()
            if math_attention:
                from torch.nn.attention import sdpa_kernel,SDPBackend
                attn=sdpa_kernel(SDPBackend.MATH)
            autocast=contextlib.nullcontext() if args.cpu_fp32 else torch.autocast('cuda',dtype=torch.bfloat16)
            with torch.no_grad(),autocast,attn:
                if cache:
                    first=model(input_ids=ids[:,:-1],use_cache=True,logits_to_keep=1)
                    out=model(input_ids=ids[:,-1:],past_key_values=first.past_key_values,use_cache=True,logits_to_keep=1)
                    del first
                else:out=model(input_ids=ids,use_cache=False,logits_to_keep=keep)
                logits=out.logits[0,-1 if isinstance(keep,int) else 0].float()
                logp=(logits/.7).log_softmax(-1)
                values,indices=logp.topk(5)
                result={'row':row,'position':pos,'mode':name,'token':tok.decode([target]),
                    'probability':float(logp[target].exp()),'target_logit':float(logits[target]),
                    'top5':[{'id':int(i),'text':tok.decode([int(i)]),'logit':float(logits[i]),'probability':float(v.exp())} for i,v in zip(indices,values)]}
                del out,logits,logp
            records.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
            save(output.with_suffix('.partial.json'),{'time':utc(),'seconds':time.monotonic()-started,'records':records})
        index=torch.tensor([pred],device=device)
        compute('full_sequence_target_projection',full,index)
        compute('prefix_target_projection',prefix,1)
        compute('cached_one_token_decode',prefix,1,cache=True)
        if not args.cpu_fp32:
            model.set_attn_implementation('sdpa')
            compute('sdpa_prefix',prefix,1)
            try:
                compute('sdpa_math_prefix',prefix,1,math_attention=True)
            except torch.OutOfMemoryError as exc:
                records.append({'row':row,'mode':'sdpa_math_prefix','error':'diagnostic_memory_limit',
                                'message':str(exc)})
                torch.cuda.empty_cache()
            import transformers.models.qwen2.modeling_qwen2 as qwen
            original=qwen.eager_attention_forward
            qwen.eager_attention_forward=chunked_fp32_attention
            model.set_attn_implementation('eager')
            old_tf32=torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32=False
            try:compute('chunked_fp32_attention_prefix',prefix,1)
            finally:
                qwen.eager_attention_forward=original
                torch.backends.cuda.matmul.allow_tf32=old_tf32
            model.set_attn_implementation('flash_attention_2')
            parameters=[p for p in model.parameters() if p.requires_grad]
            originals=[p.detach().cpu().clone() for p in parameters]
            for param in parameters:param.data=param.data.to(torch.bfloat16)
            compute('bf16_lora_prefix',prefix,1)
            for param,original in zip(parameters,originals):
                param.data=original.to(device)
            del originals
    save(output,{'time':utc(),'seconds':time.monotonic()-started,'device':device,'dtype':str(dtype),
                 'residual_fp32':args.residual_fp32,
                 'base_model':'same frozen BF16 checkpoint values, cast to float32 for CPU reference' if args.cpu_fp32 else 'BF16',
                 'records':records,'optimizer_updates':0,'generation_requests':0})


if __name__=='__main__':main()
