import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig,get_peft_model
from common import MODEL,WORK,projected_logits,json_save

@torch.inference_mode()
def check():
    initial=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda()
    model=get_peft_model(initial,LoraConfig(r=64,lora_alpha=128,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    _,s,_,y=torch.load(WORK/'rollouts-0.pt',weights_only=True)[0]
    ids=torch.tensor([s+y[:4]],device='cuda')
    mask=torch.ones_like(ids)
    state=torch.load(WORK/'projector.pt',map_location='cuda',weights_only=True)
    def fixed():
        with model.disable_adapter():
            return projected_logits(model.get_base_model(),ids,mask,len(s)-1,state)
    actor_before=model(ids).logits
    q=fixed()
    torch.manual_seed(42)
    for name,param in model.named_parameters():
        if 'lora_B' in name:param.normal_(std=.05)
    actor_after=model(ids).logits
    assert not torch.equal(actor_before,actor_after)
    assert torch.equal(q,fixed())
    with model.disable_adapter():
        base=model.get_base_model()
        original=base(ids,attention_mask=mask,use_cache=False).logits[:,len(s)-1:-1]
        zero=projected_logits(base,ids,mask,len(s)-1,dict(state,scale=0.))
        assert torch.equal(original,zero)
        outside=(q!=original).scatter_(-1,original.topk(64).indices,False)
        assert not outside.any()
    json_save(WORK/'target_check.json',dict(invariant=True,zero_exact=True,top64_only=True,
            max_actor_logit_change=(actor_after-actor_before).abs().max().item()))
if __name__=='__main__':check()
