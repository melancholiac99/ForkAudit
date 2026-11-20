"""Assemble the preregistered and post hoc measurements without changing either estimand."""
from .post_d_common import ARMS, CASES, POST, case_dir
from .post_d_precision import location
from .post_d_behavior_probs import MODELS, OFFSET
from .s22_report import tsv
from .s23_cross import EVIDENCE
from exploratory_shop_audit.grpo_forced_audit.common import read, save

OUTPUTS=('fp32_reanchored_history.tsv','fp32_reanchored_reset.tsv','behavior_logprob_agreement.tsv',
         'precision_supplement.json','POST_D_PRECISION.md')


def columns(row,label,metrics):
    assert abs(metrics['delta_kl']-metrics['alignment']+metrics['movement'])<1e-11
    for k in ('kl_before','kl_after','delta_kl','alignment','movement'):
        row[label+'_'+k]=metrics[k]


def report():
    history=[]
    for k in range(1,11):
        for arm in ARMS:
            d=read(location(k,arm)/'result.json')
            assert (d['arm'],d['update'])==(arm,k)
            s=read(POST/'snapshot'/f'{k:02}_{arm}.json')
            assert d['initial_sha']==s['actor_sha'] and s['passed']
            row=dict(arm=arm,update=k,kl_bf16_to_fp32=d['precision_kl_bf16_to_fp32'],
                bf16_target_strength=d['bf16_target_strength'],initial_sha=d['initial_sha'],final_sha=d['final_sha'],
                anchor_evidence=d['bf16_anchor_evidence'])
            columns(row,'original',d['original']);columns(row,'reanchored',d['reanchored'])
            history.append(row)
    resets=[]
    for k,a in CASES:
        d=read(case_dir(k,a)/'fp32.json')
        row=dict(arm=a,update=k,initial_sha=d['initial_sha'],reset_sha=d['reset_sha'])
        for field in ('historical','reset','historical_reanchored','reset_reanchored'):
            columns(row,field,d[field])
        resets.append(row)
    behavior=[];metadata={}
    for arm in MODELS:
        for dtype in ('bf16','fp32'):
            d=read(POST/'behavior_logprobs'/arm/f'{dtype}.json')
            assert d['metadata']['episodes']==32 and d['metadata']['model']==arm and d['rows']
            assert any(r['tokens']>0 and r['subset']=='all_replay' and r['pool']=='ALL' for r in d['rows'])
            if arm in metadata:assert metadata[arm]==d['metadata']
            metadata[arm]=d['metadata']
            behavior.extend(d['rows'])
    tsv(EVIDENCE/OUTPUTS[0],history);tsv(EVIDENCE/OUTPUTS[1],resets);tsv(EVIDENCE/OUTPUTS[2],behavior)
    summed={}
    for a in ARMS:
        h=[r for r in history if r['arm']==a]
        summed[a]={f:sum(r[f] for r in h) for f in ('original_alignment','reanchored_alignment','original_delta_kl','reanchored_delta_kl')}
        summed[a]['positive_after_reanchor']=sum(r['reanchored_alignment']>0 for r in h)
    extra=dict(post_hoc=True,history_cells=len(history),reset_cells=len(resets),behavior_models=len(metadata),
        behavior_episodes=sum(d['episodes'] for d in metadata.values()),behavior_batch_offset=OFFSET,behavior_provenance=metadata,
        sums=summed,precision_kl='KL(p_BF16 || p_FP32) on the historical candidate support',
        correction='log(q) minus normalized candidate log probabilities at the historical BF16 anchor; exact zero q remains zero',
        logprob_difference='full native vocabulary at T=1, sampled token, HF minus vLLM')
    save(EVIDENCE/OUTPUTS[3],extra)
    lines=['# 阶段 D 事后精度诊断','','四组十步保留原 BF16 候选与目标修正。每组每步使用同一轮初权重和相同前缀，'
           '将原目标相对 BF16 学生的对数概率修正，加到 FP32 学生 logits 上，再在候选集归一化。两套读数的移动量相同。','',
           '| 组 | 原累计对齐量 | 重锚累计对齐量 | 原累计 ΔKL | 重锚累计 ΔKL | 重锚对齐量为正的步数 |',
           '|---|---:|---:|---:|---:|---:|']
    for a,v in summed.items():
        lines.append('| '+a+' | '+' | '.join(f'{v[f]:+.7f}' for f in ('original_alignment','reanchored_alignment','original_delta_kl','reanchored_delta_kl'))+
                f" | {v['positive_after_reanchor']}/10 |")
    lines+=['','[40 个组次的 BF16 强度、同权重跨精度 KL 与重锚前后分解](fp32_reanchored_history.tsv) · '
            '[九次重放的历史与重置分支，两套目标逐格对照](fp32_reanchored_reset.tsv)','',
            '## 行为评测的原生 token 对数概率','',
            '每个策略重放同一批 32 条评测轨迹。在相同原生前缀和选中 token 上比较全词表、温度 1 的对数概率。'
            '下表取动作序列摘要与原始行为评测完全匹配的轨迹，单位为 nat/token。','',
            '| 模型 | 匹配原动作的轨迹 | token 数 | HF BF16 相对 vLLM 的平均绝对差 | HF FP32 相对 vLLM 的平均绝对差 |',
            '|---|---:|---:|---:|---:|']
    for a,m in metadata.items():
        values={r['precision']:r for r in behavior if r['model']==a and r['pool']=='ALL' and r['subset']=='archived_action_match'}
        if values:
            lines.append(f"| {a} | {m['archived_actions_matched']}/32 | {values['bf16']['tokens']} | "
                         f"{values['bf16']['mae']:.6f} | {values['fp32']['mae']:.6f} |")
        else:lines.append(f'| {a} | 0/32 | 0 | NA | NA |')
    lines+=['','[按动作类别、任务池、解码和匹配情况拆分的有符号误差及分位数](behavior_logprob_agreement.tsv) · '
            '[样本来源、签名与汇总](precision_supplement.json) · [原三项诊断](POST_D_DIAGNOSTICS.md)']
    (EVIDENCE/OUTPUTS[4]).write_text('\n'.join(lines)+'\n')
    return extra
