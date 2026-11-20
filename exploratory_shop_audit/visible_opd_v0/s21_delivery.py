"""Prepare de-identified diagnostic tables after all gated stages have stopped."""
import csv
import itertools
from pathlib import Path

from .s21_store import RUN, ARMS
from exploratory_shop_audit.grpo_forced_audit.common import digest_file,read,save

def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)

def build():
    dest=RUN/'delivery'
    dest.mkdir(exist_ok=True)
    grid=read(RUN/'step_size_diagnostics.json')
    chosen=read(RUN/'selection.json')
    orig=read(RUN/'original_replay_report.json')
    for name in ('baseline_optimizer_audit','step_size_diagnostics','selection','original_replay_report'):
        (dest/(name+'.json')).write_bytes((RUN/(name+'.json')).read_bytes())
    cells=[];directions=[];steps=[];comparisons=[]
    def add(report,phase):
        for k in report['grid']:
            key=str(k)
            for a in ARMS:
                v=report['endpoints'][a][key]
                cells.append(dict(phase=phase,arm=a,multiplier=k,learning_rate=k*1e-6,
                    **{s:v[s] for s in ('kl_before','kl_after','delta_kl','alignment','movement','identity_residual')},
                    lora_update_norm=report['geometry'][key]['norms'][a]))
                for row in report['optimization_steps'].get(key,{}).get(a,[]):
                    s=row['fixed_minibatch'];probe=row['fixed_probe']
                    steps.append(dict(phase=phase,arm=a,multiplier=k,step=row['optimizer_step'],rows=row['rows'],
                        kl_before=s['kl_before'],kl_after=s['kl_after'],delta_kl=s['delta_kl'],alignment=s['alignment'],
                        movement=s['movement'],identity_residual=s['identity_residual'],fixed_probe_movement=probe['movement']))
            for pair,cos in report['geometry'][key]['pairwise_cosines'].items():
                directions.append(dict(phase=phase,multiplier=k,pair=pair,cosine='NA' if cos is None else cos))
            for other in ('R-beta','P-shuffled','R'):
                left,right=report['endpoints']['P'][key],report['endpoints'][other][key]
                comparisons.append(dict(phase=phase,multiplier=k,contrast='P minus '+other,
                    **{v:left[v]-right[v] for v in ('delta_kl','alignment','movement')}))
    add(grid,'grid')
    matched={}
    if chosen['selected_multiplier'] is not None:
        smoke=read(RUN/'new_smoke_report.json');add(smoke,'rerun')
        (dest/'new_smoke_report.json').write_bytes((RUN/'new_smoke_report.json').read_bytes())
        k=str(chosen['selected_multiplier'])
        matched={a:{n:smoke['endpoints'][a][k][n] for n in ('grid_final_adapter_match','grid_final_logits_match')} for a in ARMS}
    table(dest/'step_size_diagnostics.tsv',cells)
    table(dest/'optimizer_steps.tsv',steps)
    table(dest/'lora_direction_cosines.tsv',directions)
    table(dest/'paired_contrasts.tsv',comparisons)
    artifacts={}
    for sub in ('original','zero','grid','rerun'):
        for f in sorted((RUN/sub).rglob('*')):
            if f.is_file():
                artifacts[str(f.relative_to(RUN))]=dict(bytes=f.stat().st_size,sha256=digest_file(f))
    totals={}
    for phase in ('grid','rerun'):
        totals[phase]=sum(read(p)['gpu_hours'] for p in (RUN/phase).glob('*/complete.json'))
    totals['original']=read(RUN/'original/smoke_report.json')['train_gpu_hours']
    save(dest/'private_artifacts.json',dict(server_directory=str(RUN),files=artifacts,initial='original/initial.pt',
        grid_archive_format='lossless byte-plane XOR against initial; restore via s21_store',shared_request_inputs='signed stage C student and target shards',
        fixed_support_order='SmokeBatch canonical sorted (goal,rep,step); position in answer',grid_zero_weights='original/initial.pt'))
    residual=max(abs(x['identity_residual']) for x in cells+steps)
    assert residual<1e-11
    save(dest/'acceptance.json',dict(grid_cells=len([r for r in cells if r['phase']=='grid']),rerun_matches=matched,
        steps_per_update=len(grid['batches']),batch_sizes=grid['batches'],batches_sha256=grid['minibatch_sha256'],
        identity_max_abs_residual=residual,numerical_controls=chosen['error'],old_single_step_final_hash_matches={a:orig['arms'][a]['final_hash_match'] for a in ARMS},
        selected_multiplier=chosen['selected_multiplier'],nominal_lr=1e-6,new_smoke_ran=bool(matched),
        measured_worker_gpu_hours=totals,stage_D_updates=0,stage_E_updates=0,V_reads=0,T_opd_eval_reads=0,
        reports={p.name:digest_file(p) for p in dest.iterdir() if p.is_file() and p.name!='acceptance.json'}))
if __name__=='__main__':build()
