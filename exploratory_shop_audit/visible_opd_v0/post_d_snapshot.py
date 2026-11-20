"""Audit the target's candidate anchor against an independent start-of-step actor forward."""
import gzip
import json
import time
import torch

from .c_targets import response
from .d_data import SMOKE, load_batch
from .post_d_common import ARMS, POST, configure, folder, start_at, train_model
from .score import support
from .s21_data import summarize
from .s22_report import checkpoint
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


def native_tape(k,arm):
    source = SMOKE if k==1 else folder(k,arm)
    records={}
    for rep in range(2):
        plain=source/f'rollout_{rep}.jsonl'
        stream=plain.open() if plain.exists() else gzip.open(plain.with_suffix('.jsonl.gz'),'rt')
        with stream:
            for line in stream:
                row=json.loads(line)
                if not any(row['na'].values()):
                    key=row['goal'],row['rep'],row['step']
                    assert key not in records
                    records[key]=(row['views']['none'],row['response'])
    return records


@torch.no_grad()
def audit(model,k,arm):
    dst=POST/'snapshot'/f'{k:02}_{arm}.json'
    if dst.exists():return
    start=time.monotonic()
    p=folder(k,arm);meta=read(p/'optimizer.json')
    when=read(p/'fp32.json')
    sha=start_at(model,k,arm)
    assert sha==meta['before_sha']==when['actor_before_sha']
    batch,q=load_batch(k,arm);indices=range(len(q))
    if k==1:
        saved_ref,lineage=checkpoint('bf16','initial')
        scorer_sha=read(SMOKE/'arm_R.json')['initial_adapter_sha256']
        assert all(x['weight_sha256']==sha for x in lineage)
        source='reconstructed stage-C candidate logits; independent archived BF16 initial forward and original saved student probabilities'
    else:
        staged=read(p/'target_manifest.json')
        file=p/'bf16_before.pt'
        assert digest_file(file)==staged['bf16_before_sha256']
        saved_ref=dict(enumerate(torch.load(file,weights_only=True)))
        scorer_sha=staged['scorer_sha']
        assert sha==read(folder(k-1,arm)/'optimizer.json')['after_sha']
        source='exact s saved in target construction before candidate softmax'
    tape=native_tape(k,arm)
    assert len(tape)==len(batch.rows)==len(saved_ref)
    total=positions=id_errors=mask_errors=raw_changed=0
    worst=error_sum=prob_worst=0.
    actor_logits={}
    for i,r in enumerate(batch.rows):
        assert tape.pop((r['goal'],r['rep'],r['step']))==(r['prompt'],r['answer'])
        _,raw=response(model,r['prompt'],r['answer'])
        ids,mask=support(raw,torch.tensor(r['answer'],device='cuda'))
        id_errors+=int((ids.cpu()!=r['ids']).any(1).sum())
        mask_errors+=int((mask.cpu()!=r['mask']).any(1).sum())
        m=r['mask'];v=raw.gather(1,r['ids'].cuda()).cpu()
        delta=(v-saved_ref[i])[m].double().abs()
        total+=delta.numel();positions+=len(r['answer'])
        raw_changed+=int((delta!=0).sum());worst=max(worst,float(delta.max()));error_sum+=float(delta.sum())
        probs=v.masked_fill(~m,-torch.inf).softmax(-1)
        prob_worst=max(prob_worst,float((probs-r['snapshot']).abs().max()))
        actor_logits[i]=v
    assert not tape
    strength=summarize(batch,saved_ref,saved_ref,q,indices)['kl_before']
    actor_kl=summarize(batch,actor_logits,actor_logits,q,indices)['kl_before']
    passed=(sha==scorer_sha and id_errors==mask_errors==raw_changed==0 and prob_worst<=1e-6 and
            abs(actor_kl-meta['bf16_fixed_target']['kl_before'])<1e-10)
    save(dst,dict(arm=arm,update=k,fp32_kl_before=when['fixed_target']['kl_before'],
         original_bf16_kl_before=meta['bf16_fixed_target']['kl_before'],recomputed_actor_bf16_kl=actor_kl,
         target_strength_bf16=strength,actor_sha=sha,scorer_sha=scorer_sha,records=len(q),positions=positions,
         valid_candidate_logits=total,raw_logits_differ=raw_changed,max_abs_logit_diff=worst,
         mean_abs_logit_diff=error_sum/total,probability_max_abs_diff=prob_worst,
         candidate_mismatch_positions=id_errors,mask_mismatch_positions=mask_errors,all_native_prefixes_match=True,
         anchor_evidence=source,passed=passed,seconds=time.monotonic()-start))
    print(k,arm,'snapshot',passed,flush=True)

if __name__=='__main__':
    rank=configure();model=train_model()
    for k in range(1,11):audit(model,k,ARMS[rank])
