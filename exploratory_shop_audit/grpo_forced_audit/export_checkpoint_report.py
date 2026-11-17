"""Export a completed milestone without publishing weights or raw trajectories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .common import DEFAULT_RUN, HERE, read, read_rows, save, utc


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,default=30);p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--output',type=Path);a=p.parse_args();run=a.run.resolve()
    output=a.output or HERE/'evidence'/f'checkpoint_{a.step:04d}_20261117'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.json').exists():raise FileExistsError('Preserve published snapshot; use a separate output directory')
    ledger=read_rows(run/'cost_ledger.jsonl')
    endings=[(i,r) for i,r in enumerate(ledger) if r['event']=='end' and r['stage']==f'train_{a.step:04d}' and r.get('exit_code')==0]
    assert endings,'Stage has not completed successfully'
    end_index,stage_end=endings[-1]
    ckpt=run/'training/checkpoints'/f'global_step_{a.step}'/'actor'
    fingerprints=[read(ckpt/f'live_fingerprint_rank_{i}.json') for i in range(4)]
    assert all(x==fingerprints[0] for x in fingerprints)
    file_checks={kind:[{'rank':i,'bytes':(ckpt/f'{kind}_world_size_4_rank_{i}.pt').stat().st_size,
                             'inode':(ckpt/f'{kind}_world_size_4_rank_{i}.pt').stat().st_ino} for i in range(4)]
                 for kind in ['model','optim','extra_state']}
    assert all(x['bytes']>0 for rows in file_checks.values() for x in rows)
    assert all(len({x['inode'] for x in file_checks[k]})==1 for k in ['model','optim'])
    assert (ckpt/'lora_adapter/adapter_model.safetensors').is_file()
    steps=sorted({0,*range(10,a.step+1,10)})
    inputs=[];evaluations={};table=[]
    expected={r['task_id'] for r in read(run/'splits.json')['V']}
    for step in steps:
        for mode in ['greedy','sample']:
            path=run/'evaluation'/f'checkpoint_{step:04d}_{mode}.json';raw=read(path);inputs.append(path)
            records=raw['all_unweighted']['records']
            assert len(records)==len(expected) and {r['task_id'] for r in records}==expected
            summary={key:({k:v for k,v in value.items() if k!='records'} if isinstance(value,dict) else value)
                     for key,value in raw.items()}
            evaluations[f'{step}:{mode}']=summary
            w=raw['weighted_all'];table.append({'step':step,'decode':mode,'tasks':len(expected),
                'reward':w['reward']['estimate'],'reward_ci95_low':w['reward']['ci95'][0],'reward_ci95_high':w['reward']['ci95'][1],
                'success':w['strict_success']['estimate'],'success_ci95_low':w['strict_success']['ci95'][0],
                'success_ci95_high':w['strict_success']['ci95'][1]})
    window=steps[-3:];checks={}
    for key,limit in [('reward',.03),('strict_success',.05),('L_click_plus_spec',.05)]:
        values=[evaluations[f'{step}:sample']['hard']['metrics'][key]['estimate'] for step in window]
        checks[key]={'values':values,'range':max(values)-min(values),'limit':limit,'passes':max(values)-min(values)<=limit}
    coverage=[evaluations[f'{step}:sample']['hard']['metrics']['funnel_coverage']['estimate'] for step in window]
    plateau=a.step>=30 and len(window)==3 and all(r['passes'] for r in checks.values()) and min(coverage)>=coverage[0]
    alignment=[]
    fields=['update','tokens','responses','trajectories','mean_abs_probability_difference','token_ratio_outside_08_12',
            'token_ess_fraction','response_ess_fraction','trajectory_ess_fraction','token_ratio_quantiles',
            'abs_probability_difference_quantiles','nonzero_advantage_ratio_outside_08_12']
    for row in read_rows(run/'alignment_live/metrics.jsonl'):
        if row['update']<=a.step:alignment.append({k:row[k] for k in fields if k in row})
    audit=read(run/'audit_base/metrics.json')
    result={'snapshot_time':utc(),'milestone':a.step,'full_experiment_complete':False,
            'stage_end':stage_end,'cumulative_gpu_hours_at_stage_end':sum(r['gpu_hours'] for r in ledger[:end_index+1] if r['event']=='end'),
            'checkpoint':{'all_rank_fingerprints_equal':True,'fingerprint':fingerprints[0],'files':file_checks,
                          'replicated_storage':read(ckpt/'replicated_checkpoint.json')},
            'validation':evaluations,'plateau':{'window':window,'metrics':checks,'coverage':coverage,'met':plateau},
            'alignment':alignment,'audit_base':{'tasks':len({r['task_id'] for r in audit['state_records']}),
                       'states':len(audit['state_records']),'pages':audit['pages']},
            'live_status_at_export':read(run/'live_status.json'),
            'limitations':['Periodic V validation, not final holdout performance.',
                          'Alignment remains unresolved; no retrospective likelihood correction or resampling.',
                          'ESS is likelihood-weight concentration, not fraction of training benefit retained.',
                          'Raw records and weights remain on the server.']}
    save(output/'metrics.json',result)
    with (output/'validation.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    save(output/'provenance.json',{'input_sha256':{str(p.relative_to(run)):sha(p) for p in inputs},
                                 'generator_sha256':sha(Path(__file__)),'run_id':run.name,'registration':read(run/'registration.json')})
    lines=[f'# 第{a.step}步阶段报告','',f'训练与验证、checkpoint保存已完成；整轮实验尚未完成。阶段退出时间：{stage_end["time"]}。',
           f'四rank保存指纹一致，模型、优化器、各rank RNG与LoRA文件齐全；模型和优化器的rank别名为硬链接。',
           '', '固定验证集200题；下表采用总体加权口径，95%区间见validation.csv和metrics.json。',
           '', '| 更新 | 解码 | 连续奖励 | 完全成功率 |','|---|---|---:|---:|']
    for row in table:lines.append(f'| {row["step"]} | {row["decode"]} | {row["reward"]:.4f} | {row["success"]:.2%} |')
    lines+=['',f'平台期检查窗口：{window}；判定：{"满足" if plateau else "未满足"}。按既有计划继续，最大100次更新，不因本报告另改规模。',
            f'困难组采样奖励极差 {checks["reward"]["range"]:.4f}（上限0.03）；成功率极差 {checks["strict_success"]["range"]:.4f}（上限0.05）；点击/规格残差极差 {checks["L_click_plus_spec"]["range"]:.4f}（上限0.05）。',
            f'覆盖率：{coverage}。判定按原代码浮点比较保留，未移动阈值。',
            '',f'基础分叉审计完成 {result["audit_base"]["tasks"]} 题、{result["audit_base"]["states"]} 个状态。',
            '', '训推一致性尚未修复。第11步起保存实际更新前概率；第20/23步的完整回合ESS约7.6%/3.6%，第23步最大token概率比约104.7。低平均差异不能掩盖尾部；这些不是训练收益损失比例。全部已保存摘要附在metrics.json，不按表现选取批次。',
            '验证点估计的上升不证明训推偏差无影响，区间和最终评估仍需报告。原因对照见 ../../ALIGNMENT_ROOT_CAUSE.md。',
            '',f'截至第{a.step}步阶段结束累计记录 {result["cumulative_gpu_hours_at_stage_end"]:.3f} GPU小时；用户已取消GPU小时上限，继续保留记账。',
            f'导出时实际阶段：{result["live_status_at_export"]["stage"]}。原始轨迹与权重不纳入Git。',
            '', '复算：`python -m exploratory_shop_audit.grpo_forced_audit.export_checkpoint_report --step 30 --output /path/to/new_snapshot`。需要服务器保存的输入；provenance.json列出校验和。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'step':a.step,'plateau':plateau,'live_stage':result['live_status_at_export']['stage']},ensure_ascii=False))


if __name__=='__main__':main()
