from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
r=json.loads((ROOT/'exploratory_P4/x.json').read_text());z=r['results'];I=z['I']
def f(v):
 if v['estimate'] is None:return 'undefined (n=0)'
 lo,hi=v['ci95'];return f"{100*v['estimate']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] pp; n={v['n']}, trajectories={v['n_traj']}"
lines=['P4-X CPU 正式运行报告','',
'仓库：melancholiac99/ForkAudit；保持运行前分支 main，未执行 checkout/switch。',
'git fetch 初次因TLS失败；指定现有HTTPS代理后成功。git merge --no-edit origin/main 快进合并，无冲突。',
'运行提交：'+r['commit'],
'包含 P4-X dd8b07973ca0d48607c8322b0c65f785352d4a8f 与 P5 85f0864c440b53f1aa00794dc2cfacc2d8f1f391。',
'本轮未运行E2、未调用GPU、未采集轨迹或训练模型、未推送远端。原始数据、分析脚本、预注册和P4/E1结果保持原样。',
'', '正式读法：'+r['reading'],
'按预注册表序：PX-1/PX-2成立，且探索类内收益95%CI上界<0，命中R-X-MISDIRECT。',
'允许的表述：在本设置的信息受限候选层Fork上，探索选项内部的采样偏好也系统性偏向较差选项。',
'该读法是E1结果触发后的探索性稳健性检查，不修改P4原判定R-MIXED，不证明更强模型的结果，也不启动E3。',
'', '参数：seed=30102026；10,000次轨迹cluster bootstrap；原1478个Live点的同一轨迹集；q=库1+2，Gold第二读出。',
'依赖与路径检查：preflight.json；必需文件10735个齐全，Gold保留300点；自检selftest passed。',
'一致性校验：logp_sum复算I层G_u = '+repr(r['consistency_gate']['recomputed'])+'，E1 = '+repr(r['consistency_gate']['e1'])+'，绝对差为0。',
'', 'I层主人群 X1–X7；数值为百分点，方括号为95%CI。']
for name,label in [('X1_logp_mean','X1 长度归一化logp_mean'),('X2_drafts_batch0','X2 首批草稿边际'),('X2b_drafts_all','X2 全批次敏感性'),('X3_chosen','X3 实际chosen动作')]:
 lines.append(label)
 for src in ['12','gold']:
  for metric in ['G_u','G_c']:lines.append('  '+src+' '+metric+': '+f(I[src][name][metric]))
lines.append('X4 探索分解（脚本只输出库1+2）：')
for name,v in I['X4'].items():lines.append('  '+name+': '+f(v))
lines.append('  P_E是首批32份草稿中有效且合法的动作内E类的比例；并非以固定32作为分母。其余选臂概率限制在五臂内并加0.5平滑。')
lines.append('X5 循环（库1+2）：')
for name,v in I['X5'].items():lines.append('  '+name+': '+f(v))
lines.append('X6 首个Live I点：55条轨迹各一个点，其中27个Fork。')
for name in ['X2_drafts_batch0','X3_chosen']:lines.append('  '+name+': '+f(z['X6'][name]))
lines.append('X7 留一轨迹（估计范围，非CI）：')
for name in ['loto_X2','loto_X3']:
 v=z['X7'][name];lines.append(f"  {name}: [{v['min']*100:+.2f}, {v['max']*100:+.2f}] pp; {v['n_traj']} trajectories")
lines.append('X7 任务类型：n / X2 / X3（点估计，非显著性判定）')
for task,v in z['X7']['by_task_type'].items():lines.append(f"  {task}: {v['n']} / {100*v['X2']:+.2f} / {100*v['X3']:+.2f} pp")
lines+=['','预测：']+[f'  {key}: {value}' for key,value in r['predictions'].items()]
lines+=['','重要限定：',
'X1、X2在I层Gold上仍显著为负；X3 Gold为−5.65 pp [−10.95,+0.95]，CI含0，不能称其在Gold上显著复现。',
'X6方向未翻转，X7留一轨迹范围未跨0；五个已出现任务类型点估计均负，但热处理任务仅4点，不能声称各类型都显著。',
'原脚本仅为X1–X3及E1对照提供Gold读出；X4/X5为库1+2，X6/X7仅输出I层库1+2。未修改脚本扩展这些读出，也不声称X4–X7有Gold复现。',
'X4 d_EX需同时具有E/N类（181点）；类内收益需至少两个E臂（112点）。样本减少由预注册条件造成，不是文件缺失。N层无目标揭示臂，revealing_arms_E_share为undefined。',
'', 'U、N对照层（同原脚本全部输出）：']
for layer in ['U','N']:
 lines.append(layer)
 for source in ['12','gold']:
  for sel,metrics in z[layer][source].items():
   for metric,v in metrics.items():lines.append('  '+source+'/'+sel+'/'+metric+': '+f(v))
 for check in ['X4','X5']:
  for name,v in z[layer][check].items():lines.append('  '+check+'/'+name+': '+f(v))
lines+=['','执行命令（仓库根目录）：',
'git -c http.proxy="$HTTPS_PROXY" -c http.version=HTTP/1.1 fetch origin',
'git merge --no-edit origin/main',
"CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/m0/bin/python exploratory_P4/p4_x_robust.py --selftest > exploratory_P4/p4_x_run/selftest.log 2>&1",
"CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/m0/bin/python exploratory_P4/p4_x_robust.py > exploratory_P4/p4_x_run/run.log 2>&1",
'','文件：exploratory_P4/x.json；exploratory_P4/p4_x_run/{selftest.log,run.log,preflight.json,verification.json,REPORT.txt}。',
'原脚本记录的正式分析耗时：'+str(r['seconds'])+'秒。']
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
