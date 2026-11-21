"""Write aggregate, reproducible stage-E artifacts and preserve all registered endpoints."""
import csv
from .e25_layout import Layout
from .e25_probe import RUN,PUBLIC
from .s22_report import tsv
from exploratory_shop_audit.grpo_forced_audit.common import read,read_rows,save,digest_file

def final_report():
    l=Layout('formal');completion=read(l.root/'complete.json')
    steps=[];monitors=[];staging={}
    for a,last in completion['endpoints'].items():
        for k in range(1,last+1):
            directory=l.folder(k,a);s=read(directory/'train_statistics.json')
            m=read(directory/'measurement.json');opt=read(directory/'optimizer.json')
            assert opt['before_sha']==s['before_sha'] and opt['after_sha']==s['after_sha']
            raw={f'{prec}_{field}':m[prec][field] for prec in ('bf16','fp32_reanchored') for field in ('kl_before','kl_after','delta_kl','alignment','movement')}
            steps.append(dict(arm=a,update=k,legal_rate=s['legal_rate'],records=s['records'],scored=s['scored'],
                 before_sha=s['before_sha'],after_sha=s['after_sha'],optimizer_steps=s['optimizer_steps'],
                 clip_count=s['clip_count'],projector_version=s['projection_version'],**raw))
            staging[f'{a}/{k}']=dict(classes=s['by_class'],candidate_mass=s.get('candidate_mass'))
            if k%10==0:
                v=read(l.root/'monitor'/f'{k:03}.json')[a]
                for group,x in v['stats'].items():
                    monitors.append(dict(arm=a,update=k,group=group,**x,projector_mse=v['projector_mse'],projector_cosine=v['projector_cosine']))
    tsv(PUBLIC/'stage_e_steps.tsv',steps);tsv(PUBLIC/'stage_e_monitors.tsv',monitors)
    save(PUBLIC/'stage_e_category_metrics.json',staging)
    behavior={str(k):read(l.root/'behavior'/f'{k:03}.json') for k in range(0,completion['last_step']+1,10)}
    save(PUBLIC/'stage_e_final_behavior.json',behavior[str(completion['last_step'])])
    # Keep all intermediate paired intervals locally. Publish the point estimates by checkpoint.
    table=[]
    for k,r in behavior.items():
        for key,x in r['summary'].items():
            if x['tasks']:
                a,pool,mode=key.split('/')
                table.append(dict(update=k,arm=a,checkpoint=r['endpoints'][a],pool=pool,mode=mode,
                    **{f:x[f] for f in ['tasks','episodes','reward','all_requirements','strict','decision_entropy','legal_rate','length']}))
    tsv(PUBLIC/'stage_e_behavior_by_checkpoint.tsv',table)
    actual=read_rows(RUN/'allocation.jsonl');costs={}
    for r in actual:costs[r['run']]=costs.get(r['run'],0.)+r['elapsed']*r['allocated_gpus']/3600
    check=l.root/'rematch_at_10.json'
    save(PUBLIC/'stage_e_completion.json',dict(completion=completion,actual_allocated_gpu_hours=costs,
        formal_settings=l.config,step10_matching=read(check) if check.exists() else None,
        offline_pilot_trigger=read(RUN/'offline/complete.json')['p_kl_pilot_after_E'],description='单种子探索性；沿修正方向反复推进。'))
    save(l.root/'local_delivery.json',dict(public_sha256={p.name:digest_file(p) for p in sorted(PUBLIC.iterdir()) if p.is_file()}))
