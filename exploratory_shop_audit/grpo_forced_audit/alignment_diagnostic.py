"""Read-only fixed-token comparison with saved behavior log probabilities."""
import argparse
import json
import time
from pathlib import Path


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from verl import DataProto
    from omegaconf import OmegaConf
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from .common import DEFAULT_RUN, MODEL, save, utc
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--microbatch-check', action='store_true')
    args = parser.parse_args()
    run = DEFAULT_RUN.resolve()
    folder = run/'alignment_diagnostic'/f'checkpoint_{args.checkpoint:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    if args.microbatch_check:
        assert args.checkpoint == 2, 'The saved diagnostic cases belong to checkpoint2'
    completion = folder/('microbatch_check.json' if args.microbatch_check else 'completed.json')
    if completion.exists():
        raise FileExistsError(f'Preserve completed diagnostic: {completion}')
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.set_num_threads(2)
    started = time.monotonic()
    packet = DataProto.load_from_disk(str(next((run/'native_batch_cache').glob(f'train_{args.checkpoint:04d}_sample_*.pkl'))))
    packet.to('cpu')
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16,
                                               attn_implementation='flash_attention_2', device_map='cuda:0')
    adapter = run/'training/checkpoints'/f'global_step_{args.checkpoint}'/'actor/lora_adapter'
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True, autocast_adapter_dtype=True)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    model.eval()
    if args.microbatch_check:
        from verl.models.transformers.monkey_patch import apply_monkey_patch
        apply_monkey_patch(model.get_base_model(), use_remove_padding=True, ulysses_sp_size=1,
                          use_fused_kernels=False)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    cfg = OmegaConf.load(run/'resolved_train_config.yaml').actor_rollout_ref.actor
    cfg.use_remove_padding = True
    actor = DataParallelPPOActor(cfg, model, None)
    if args.microbatch_check:
        records = []
        for indices, float_logits in [([508],False),([508,509,510,511],False),([508],True)]:
            micro = {k:packet.batch[k][indices].cuda() for k in ['responses','input_ids','attention_mask','position_ids']}
            hook = model.get_output_embeddings().register_forward_hook(lambda module, inputs, output:output.float()) if float_logits else None
            with torch.no_grad():
                _, values = actor._forward_micro_batch(micro,.7,False)
            records.append({'rows':indices,'float32_logits_before_temperature':float_logits,
                            'row508_position2_logp':float(values[0,2]),'probability':float(values[0,2].exp())})
            if hook:hook.remove()
            del micro,values
            torch.cuda.empty_cache()
        save(folder/'microbatch_check.json',{'time':utc(),'seconds':time.monotonic()-started,'records':records})
        print('MICROBATCH',json.dumps(records),flush=True)
        return
    width = packet.batch['responses'].shape[-1]
    mask = packet.batch['attention_mask'][:, -width:].bool().cpu()
    rollout = packet.batch['rollout_log_probs'].float().cpu()
    actor_probs = torch.full_like(rollout, float('nan'))
    count = min(len(packet), args.limit or len(packet))
    save(folder/'started.json', {'time':utc(), 'checkpoint':args.checkpoint, 'rows':count,
                                'microbatch':1, 'gpu_memory_limit_gib':20,
                                'resource_note':'Shares existing audit GPU; occupancy already accounted by audit phase.'})
    for i in range(count):
        micro = {k:packet.batch[k][i:i+1].cuda() for k in ['responses','input_ids','attention_mask','position_ids']}
        with torch.no_grad():
            _, values = actor._forward_micro_batch(micro, .7, False)
        actor_probs[i] = values[0].float().cpu()
        del micro, values
        if (i+1) % 25 == 0 or i+1 == count:
            np.savez_compressed(folder/'probabilities.npz', actor=actor_probs[:i+1].numpy(),
                                rollout=rollout[:i+1].numpy(), mask=mask[:i+1].numpy())
            save(folder/'progress.json', {'rows':i+1,'total':count,'seconds':time.monotonic()-started,'time':utc()})
            print('PROGRESS', i+1, count, round(time.monotonic()-started,1), flush=True)
    selected = (actor_probs[:count].exp()-rollout[:count].exp()).abs().masked_fill(~mask[:count], -1)
    top = selected.flatten().topk(min(40,int(mask[:count].sum()))).indices
    examples = []
    for idx in top.tolist():
        row, pos = divmod(idx,width)
        tokens = packet.batch['responses'][row][mask[row]].tolist()
        examples.append({'row':row,'position':pos,'token_id':tokens[pos],
                         'token':tokenizer.decode([tokens[pos]]),'response':tokenizer.decode(tokens),
                         'decision':str(packet.non_tensor_batch['shop_decision'][row]),
                         'trainer_logp':float(actor_probs[row,pos]),'sampler_logp':float(rollout[row,pos]),
                         'abs_probability_difference':float(selected[row,pos])})
    save(folder/'largest_differences.json', examples)
    save(folder/'completed.json', {'time':utc(),'rows':count,'seconds':time.monotonic()-started,
                                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                                 'optimizer_updates':0,'generation_requests':0})


if __name__ == '__main__':
    main()
